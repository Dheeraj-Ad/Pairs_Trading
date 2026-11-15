"""Daily research: temporal isolation, accounting, provenance and actual attribution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from ml.report.data import prepare
from ml.report.discovery import discover
from ml.report.features import engineer
from ml.report.models import train_pair, weighted_loss, thresholds
from ml.report.backtest import backtest
from ml.report.pipeline import run
from app.report_lab import ReportLab
from scripts.run_daily_report import synthetic_prices


class DailyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = synthetic_prices()

    def test_future_prices_cannot_change_formation_discovery(self):
        original = prepare(self.prices)
        changed = self.prices.copy()
        cut = int(len(changed) * .3)
        changed.iloc[cut:, 0] *= np.linspace(.8, 1.5, len(changed)-cut)
        altered = prepare(changed)
        self.assertEqual(original[3], altered[3])
        self.assertEqual(discover(original[0], original[3]), discover(altered[0], altered[3]))

    def test_rolling_features_do_not_use_future_prices(self):
        pair = dict(y='HDFCBANK', x='ICICIBANK')
        changed = self.prices.copy()
        changed.iloc[550:, 0] *= 1.5
        original = engineer(self.prices, pair)
        altered = engineer(changed, pair)
        pd.testing.assert_frame_equal(original[0].iloc[:550], altered[0].iloc[:550])
        pd.testing.assert_frame_equal(original[4].iloc[:550], altered[4].iloc[:550])
        # Future outcomes are labels only; their information boundary is t+6.
        pd.testing.assert_series_equal(original[1].iloc[:544], altered[1].iloc[:544])

    def test_daily_contract_rejects_intraday_duplicates_and_short_history(self):
        for bad in [self.prices.iloc[:599], pd.concat([self.prices, self.prices.iloc[:1]])]:
            with self.assertRaises(ValueError): prepare(bad)
        bad = self.prices.copy()
        bad.index = pd.date_range('2024-01-01', periods=len(bad), freq='5min', tz='UTC')
        with self.assertRaisesRegex(ValueError, 'daily'): prepare(bad)

    def test_invalid_dates_and_path_like_symbols_are_rejected(self):
        bad = self.prices.rename(columns={'HDFCBANK': '../outside'})
        with self.assertRaisesRegex(ValueError, 'Stock symbols'): prepare(bad)
        bad = self.prices.copy()
        dates = list(bad.index)
        dates[0] = pd.NaT
        bad.index = dates
        with self.assertRaisesRegex(ValueError, 'valid dates'): prepare(bad)

    def test_loss_penalty_and_thin_thresholds(self):
        probs = np.array([[.5, .25, .25], [.25, .25, .5]])
        self.assertAlmostEqual(weighted_loss([-1, 1], probs), -2*np.log(.5))
        self.assertEqual(thresholds(np.array([1, 0]), probs, np.array([-2.5, 2.5])),
                         dict(long=1.01, short=1.01))


class DailyExecutionTests(unittest.TestCase):
    def test_next_close_fills_integer_neutrality_costs_and_reconciliation(self):
        prices = pd.DataFrame({'Y': [100.]*50, 'X': [50.]*50},
                              index=pd.date_range('2025-01-01', periods=50, freq='B', tz='UTC'))
        prices.loc[prices.index[42]:, 'Y'] = 110.
        signal = dict(action='Long', side='long', zscore=-2.5, score=.8, features={}, high_volatility=False)
        signals = {40: signal, 41: signal, 42: {**signal, 'action': 'Hold', 'zscore': 0}}
        bundle = dict(report=dict(y='Y', x='X'), signals=signals)
        result = backtest(prices, {'pair': bundle})
        trade = result['trades'][0]
        self.assertEqual((trade['signal_index'], trade['entry_index'], trade['exit_index']), (40, 41, 43))
        self.assertEqual(trade['quantity'], [749, -1498])
        self.assertEqual(trade['dollar_imbalance'], 0)
        entry = [100*1.0007, 50*.9993]
        exit_ = [110*.9993, 50*1.0007]
        fee = .0003 * (749*(entry[0]+exit_[0]) + 1498*(entry[1]+exit_[1]))
        days = (prices.index[43]-prices.index[41]).days
        borrow = 1498*entry[1]*.1*days/365
        expected = 749*(exit_[0]-entry[0])-1498*(exit_[1]-entry[1])-fee-borrow
        self.assertAlmostEqual(trade['pnl'], expected)
        self.assertAlmostEqual(result['metrics']['net_pnl'], expected)
        self.assertEqual(result['equity'][-1]['gross'], 0)
        json.dumps(result, allow_nan=False)

    def test_no_signals_means_no_trades_or_fabricated_metrics(self):
        result = backtest(synthetic_prices(), {})
        self.assertEqual(result['trades'], [])
        self.assertEqual(result['metrics']['net_pnl'], 0)
        self.assertIsNone(result['metrics']['sharpe'])
        self.assertIsNone(result['metrics']['breakeven_all_in_bps'])

    def test_artifacts_are_account_scoped_and_no_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            app = SimpleNamespace(directory=Path(directory), account_key='demo:NSE')
            lab = ReportLab(app)
            lab.results['demo:NSE'] = dict(run_id='run', artifacts=['daily_prices.csv'])
            self.assertEqual(lab.artifact('run', 'daily_prices.csv').name, 'daily_prices.csv')
            for name in ['../../.env', 'missing.csv']:
                with self.assertRaises(ValueError): lab.artifact('run', name)
            app.account_key = 'demo:BSE'
            self.assertIsNone(lab.view()['result'])
            with self.assertRaises(ValueError): lab.artifact('run', 'daily_prices.csv')


class DailyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.prices = synthetic_prices()
        cls.result = run(cls.prices, cls.temp.name, 'Synthetic test data')

    def test_full_report_exports_and_reconciles_actual_trades(self):
        result = self.result
        self.assertEqual(result['possible_pairs'], 190)
        self.assertEqual(result['verified'], 2)
        self.assertEqual(result['modeled'], 1)
        self.assertEqual(result['failures'][0]['pair'], 'HCLTECH__WIPRO')
        self.assertIn('Insufficient causal samples', result['failures'][0]['error'])
        self.assertGreater(len(result['backtest']['trades']), 0)
        self.assertAlmostEqual(sum(t['pnl'] for t in result['backtest']['trades']),
                               result['backtest']['metrics']['net_pnl'])
        for name in result['artifacts'] + ['report.json']:
            self.assertTrue(Path(self.temp.name, name).is_file(), name)
        json.dumps(result, allow_nan=False)

    def test_shap_explains_every_executed_entry_score(self):
        trades = self.result['backtest']['trades']
        explanations = {(e['pair'], e['signal_index']): e for e in self.result['explanations']}
        for trade in trades:
            e = explanations[(trade['pair'], trade['signal_index'])]
            self.assertAlmostEqual(e['base_value'] + sum(e['contributions'].values()), trade['score'], places=6)

    def test_test_data_cannot_change_model_selection_or_thresholds(self):
        pair = next(p for p in self.result['candidates'] if p['id']=='HDFCBANK__ICICIBANK')
        original = next(p for p in self.result['pair_models'] if p['pair']==pair['id'])
        changed = self.prices.copy()
        cut = int(len(changed)*.8)
        changed.iloc[cut:, 0] *= np.linspace(.99, 1.03, len(changed)-cut)
        altered = train_pair(changed, pair)['report']
        for key in ['selected_regression', 'lasso_coefficients', 'retained_features', 'regime_cutoff']:
            self.assertEqual(original[key], altered[key])
        for key in ['model', 'parameters', 'thresholds', 'validation', 'train_loss']:
            self.assertEqual(original['selected'][key], altered['selected'][key])


if __name__ == '__main__':
    unittest.main()
