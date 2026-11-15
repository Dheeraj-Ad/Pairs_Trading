from copy import deepcopy
from datetime import datetime, timedelta, timezone
import tempfile
import time
import unittest
import uuid

import numpy as np
import pandas as pd

from app.server import Application
from app.market import fresh, PAIRS
from ml.paper_model import train_trades, dataset
from ml.research import FEATURES


class LocalWorkflowTests(unittest.TestCase):
    def test_expanding_saved_universe_preserves_prices_clock_and_ledger(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            app=Application(directory)
            app.advance_local(2)
            key=app.account_key
            old=deepcopy(app.local.state(key))
            old['bars']={p:bars for p,bars in old['bars'].items() if p in ('banks','it','auto')}
            with app.book.connect() as db:
                db.execute('UPDATE local_markets SET body=? WHERE account=?',(json.dumps(old),key))
            reopened=Application(directory)
            state=reopened.local.state(key)
            self.assertEqual(state['step'],old['step'])
            self.assertEqual(reopened.market_now(),app.market_now())
            for pair,bars in old['bars'].items(): self.assertEqual(state['bars'][pair],bars)
            self.assertEqual(len(state['bars']),len(PAIRS))
            self.assertEqual(len(reopened.quotes),20)
            for pair in PAIRS:
                bars=state['bars'][pair['id']]
                for leg in ('y','x'):
                    self.assertEqual(bars[leg][-1]['time']+300,int(reopened.market_now().timestamp()))
                    self.assertEqual(bars[leg][-1]['close'],reopened.quotes[pair[leg]]['price'])
            self.assertEqual(reopened.book.read(key),app.book.read(key))
            reopened.advance_local()
            self.assertEqual(len(reopened.quotes),20)

    def test_background_batch_can_pause_and_blocks_switching(self):
        from unittest.mock import patch
        import threading
        with tempfile.TemporaryDirectory() as directory:
            app=Application(directory)
            entered,release=threading.Event(),threading.Event()
            original=app.advance_local
            def advance():
                entered.set()
                release.wait(5)
                original()
            with patch.object(app,'advance_local',side_effect=advance):
                app.start_local_batch(75)
                try:
                    self.assertTrue(entered.wait(2))
                    with self.assertRaisesRegex(ValueError,'batch'):
                        app.terminal.can_switch('demo:BSE')
                    with self.assertRaises(ValueError): app.terminal.scan()
                    # Cancellation is checked between bars, never halfway through a pair fill.
                    app.local.remaining=1
                finally:
                    release.set()
                deadline=time.monotonic()+5
                while app.local.batch_active and time.monotonic()<deadline: time.sleep(.01)
                self.assertFalse(app.local.batch_active)
            self.assertEqual(app.local.state(app.account_key)['step'],1)
            self.assertEqual(app.local.remaining,0)
            app.terminal.can_switch('demo:BSE')

    def test_local_clock_prices_fills_and_restart_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(directory)
            start = app.market_now()
            app.action('/api/open', dict(account_key=app.account_key, pair='banks', side='long',
                                        budget=100000, request_id=str(uuid.uuid4())))
            opened = app.book.read(app.account_key)['positions'][0]
            self.assertEqual(opened['opened'], start.isoformat())
            app.advance_local(3)
            state = app.snapshot()
            self.assertGreater(app.market_now(), start)
            self.assertAlmostEqual(state['performance']['overall']['total'], state['account']['total_pnl'])
            self.assertEqual(len(state['transactions']), 2)
            self.assertGreater(state['account']['positions'][0]['borrow'], 0)
            app.action('/api/close', dict(account_key=app.account_key, position_id=opened['id'], request_id=str(uuid.uuid4())))
            state = app.snapshot()
            self.assertEqual(len(state['transactions']), 4)
            self.assertAlmostEqual(sum(f['fee'] for f in state['transactions']), state['account']['trades'][0]['fees'])
            self.assertAlmostEqual(sum(r['total'] for r in state['performance']['methods']), state['account']['total_pnl'])
            self.assertAlmostEqual(sum(r['total'] for r in state['performance']['pairs']), state['account']['total_pnl'])
            reopened = Application(directory)
            self.assertEqual(reopened.market_now(), app.market_now())
            self.assertEqual(reopened.quotes, app.quotes)
            self.assertEqual(reopened.snapshot()['account'], state['account'])
            self.assertFalse(reopened.local.running)
            for pair in state['pairs']:
                bars = app.local.candles(app.account_key, pair['id'])
                for i, leg in enumerate(('y','x')):
                    self.assertEqual(bars[leg][-1]['close'], pair['quotes'][i]['price'])
                    self.assertTrue(fresh(pair['quotes'][i], app.market_now()))
            before = app.local.state(app.account_key)['step']
            app.mode='live'
            with self.assertRaises(ValueError): app.advance_local()
            app.mode='demo'
            self.assertEqual(app.local.state(app.account_key)['step'], before)

    def test_real_local_model_generates_auto_trades_without_forced_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(directory)
            app.terminal.scan()
            deadline = time.monotonic()+30
            while app.terminal.job['running'] and time.monotonic()<deadline:
                time.sleep(.05)
            self.assertFalse(app.terminal.job['running'])
            self.assertTrue(any(r['eligible'] for r in app.terminal.screen))
            app.terminal.configure({'enabled': True})
            for _ in range(3):
                app.advance_local(75)
            state = app.snapshot()
            auto = [t for t in state['account']['trades'] if t['origin']=='auto']
            self.assertTrue(auto, 'Qualified local model must complete at least one forward simulated trade')
            for trade in auto:
                self.assertEqual(set(trade['decision']['features']), set(FEATURES))
                self.assertGreaterEqual(trade['closed'], trade['opened'])
                self.assertEqual(len(trade['entry_legs']), 2)
                self.assertEqual(len(trade['exit_legs']), 2)
            self.assertEqual(state['paper_model']['rows'], len(auto))
            self.assertAlmostEqual(state['performance']['overall']['total'], state['account']['total_pnl'])
            self.assertEqual(app.book.read('live:NSE')['trades'], [])


class PaperLearningTests(unittest.TestCase):
    def examples(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        rng=np.random.default_rng(33)
        rows=[]
        for i in range(100):
            z=float(rng.normal(0,2))
            entry=dict(zscore=z, abs_zscore=abs(z), spread_change=float(rng.normal()), spread_volatility=float(rng.uniform(.2,1)))
            opened=start+timedelta(hours=i)
            rows.append(dict(id=str(i), opened=opened.isoformat(),closed=(opened+timedelta(minutes=25)).isoformat(),
                             pnl=100 if abs(z)>1.5 else -50, decision={'features':entry}))
        return rows

    def test_holdout_outcomes_do_not_change_training_or_validation(self):
        rows=self.examples()
        model,report=train_trades(rows)
        changed=deepcopy(rows)
        for trade in changed[80:]: trade['pnl']*=-1
        other,other_report=train_trades(changed)
        probe=pd.DataFrame([t['decision']['features'] for t in rows],columns=FEATURES)
        np.testing.assert_array_equal(model.predict_proba(probe),other.predict_proba(probe))
        self.assertEqual(report['validation'],other_report['validation'])
        self.assertEqual(report['eligible'],other_report['eligible'])
        self.assertNotEqual(report['test']['accuracy'],other_report['test']['accuracy'])

    def test_insufficient_data_legacy_exclusion_and_overlap_purge(self):
        rows=self.examples()
        self.assertFalse(dataset([{**rows[0], 'decision':None}]))
        with self.assertRaisesRegex(ValueError,'40'): train_trades(rows[:20])
        rows[59]['closed']=rows[65]['closed']
        rows[79]['closed']=rows[85]['closed']
        _,report=train_trades(rows)
        self.assertEqual(report['purged_rows'],2)
        self.assertEqual(report['train_rows'],59)

    def test_training_and_filter_are_account_scoped_and_restart_pauses_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            app=Application(directory)
            account=app.book.read(app.account_key)
            account['trades']=self.examples()
            import json
            with app.book.connect() as db:
                db.execute('INSERT INTO accounts VALUES(?,?)',(app.account_key,json.dumps(account)))
            app.paper_learner.train()
            self.assertTrue(app.paper_learner.view()['loaded'])
            app.paper_learner.configure(True)
            self.assertTrue(app.paper_learner.view()['enabled'])
            app.mode='live'
            self.assertEqual(app.paper_learner.view()['rows'],0)
            self.assertIsNone(app.paper_learner.view()['report'])
            with self.assertRaises(ValueError): app.paper_learner.configure(True)
            reopened=Application(directory)
            self.assertFalse(reopened.paper_learner.view()['loaded'])
            self.assertFalse(reopened.paper_learner.view()['enabled'])
            self.assertIsNotNone(reopened.paper_learner.view()['report'])


if __name__=='__main__': unittest.main()
