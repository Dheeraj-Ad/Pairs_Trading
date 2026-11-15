"""Learn from completed paper trades using only features captured before entry."""
import json
import math

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from ml.research import FEATURES

MIN_TRADES = 40


def dataset(trades):
    rows = []
    for trade in trades:
        entry = (trade.get('decision') or {}).get('features', {})
        if not all(isinstance(entry.get(k), (int, float)) and math.isfinite(entry[k]) for k in FEATURES):
            continue
        if trade['closed'] < trade['opened']:
            continue
        rows.append({**{k: entry[k] for k in FEATURES}, 'opened': trade['opened'],
                     'closed': trade['closed'], 'target': int(trade['pnl'] > 0), 'id': trade['id']})
    return sorted(rows, key=lambda r: (r['opened'], r['id']))


def train_trades(trades):
    rows = dataset(trades)
    if len(rows) < MIN_TRADES:
        raise ValueError(f'Need {MIN_TRADES} completed trades with entry features; currently {len(rows)}. Keep paper trading first.')
    a, b = int(len(rows) * .6), int(len(rows) * .8)
    # Labels must have resolved before the next evaluation period starts.
    train = [r for r in rows[:a] if r['closed'] < rows[a]['opened']]
    valid = [r for r in rows[a:b] if r['closed'] < rows[b]['opened']]
    test = rows[b:]
    if len(train) < 12 or min(len(valid), len(test)) < 6 or len({r['target'] for r in train}) < 2:
        raise ValueError('Need more non-overlapping completed trades and both winning and losing training examples.')
    def xy(part):
        df = pd.DataFrame(part)
        return df[FEATURES], df.target
    model = RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=4, random_state=19, n_jobs=1)
    xt, yt = xy(train)
    model.fit(xt, yt)
    def metrics(part):
        x, y = xy(part)
        scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
        return dict(rows=len(part), accuracy=float(accuracy_score(y, scores >= .5)),
                    balanced_accuracy=float(balanced_accuracy_score(y, scores >= .5)),
                    auc=float(roc_auc_score(y, scores)) if y.nunique() == 2 else None,
                    majority_baseline=float(max(y.mean(), 1 - y.mean())))
    validation, holdout = metrics(valid), metrics(test)
    report = dict(algorithm='Paper outcome random forest', target='Closed paper trade net P&L > 0',
        rows=len(rows), train_rows=len(train), purged_rows=a-len(train)+(b-a)-len(valid),
        validation=validation, test=holdout, first_entry=rows[0]['opened'], last_exit=max(r['closed'] for r in rows),
        eligible=(validation['auc'] or 0) > .5 and validation['balanced_accuracy'] > .5,
        feature_importance=dict(zip(FEATURES, map(float, model.feature_importances_))))
    return model, report


class PaperLearner:
    def __init__(self, app):
        self.app = app
        self.models, self.reports = {}, {}
        self.enabled = set()
        with app.book.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS paper_models(account TEXT PRIMARY KEY, report TEXT)')
            for account, report in db.execute('SELECT account,report FROM paper_models'):
                self.reports[account] = json.loads(report)

    def train(self):
        account = self.app.account_key
        model, report = train_trades(self.app.book.read(account)['trades'])
        report.update(account=account, trained_at=self.app.market_now().isoformat())
        joblib.dump(model, self.app.directory / ('paper-model-' + account.replace(':', '-') + '.joblib'))
        with self.app.book.connect() as db:
            db.execute('INSERT OR REPLACE INTO paper_models VALUES(?,?)', (account, json.dumps(report, allow_nan=False)))
        self.models[account], self.reports[account] = model, report
        self.enabled.discard(account)

    def score(self, entry):
        model = self.models.get(self.app.account_key)
        if model is None or not all(k in entry for k in FEATURES):
            return None
        return float(model.predict_proba(pd.DataFrame([entry], columns=FEATURES))[0, list(model.classes_).index(1)])

    def configure(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Specify whether the paper-trained filter is enabled.')
        key = self.app.account_key
        if enabled and (key not in self.models or not self.reports[key]['eligible']):
            raise ValueError('Train a paper model that passes chronological validation before enabling its entry filter.')
        if enabled: self.enabled.add(key)
        else: self.enabled.discard(key)

    def view(self):
        trades = self.app.book.read(self.app.account_key)['trades']
        rows = dataset(trades)
        return dict(rows=len(rows), excluded=len(trades)-len(rows), minimum=MIN_TRADES,
                    wins=sum(r['target'] for r in rows), losses=sum(1-r['target'] for r in rows),
                    report=self.reports.get(self.app.account_key), loaded=self.app.account_key in self.models,
                    enabled=self.app.account_key in self.enabled)
