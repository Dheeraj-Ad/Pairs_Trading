from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.market import PAIRS, IST, demo_candles, candle_frame, normalize_candles, Upstox
from app.server import Application
from ml.profit_model import train_profit, features, net_targets
from ml.research import baseline


class CandleTests(unittest.TestCase):
    def test_slow_quote_fetch_does_not_block_state_or_cross_accounts(self):
        with tempfile.TemporaryDirectory() as directory:
            app=Application(directory)
            app.mode='live';app.provider.token='test'
            started,release=threading.Event(),threading.Event()
            def quote_fetch(exchange):
                started.set();release.wait(5)
                return {'SHOULD_NOT_APPEAR':{'price':99}}
            with patch.object(app.provider,'quotes',side_effect=quote_fetch):
                worker=threading.Thread(target=app.poll);worker.start()
                try:
                    self.assertTrue(started.wait(2))
                    before=time.monotonic();app.snapshot()
                    self.assertLess(time.monotonic()-before,1)
                    with app.lock:
                        app.mode='demo';app.quotes={};app.terminal.reset()
                finally:
                    release.set();worker.join(5)
            self.assertEqual(app.quotes,{})

    def test_invalid_ohlc_never_reaches_chart(self):
        for row in [['2026-11-16T09:15:00+05:30',100,90,80,100,10],
                    ['2026-11-16T09:15:00',100,110,90,100,10],
                    ['2026-11-16T09:15:00+05:30',100,110,90,float('nan'),10]]:
            with self.assertRaises(ValueError): normalize_candles([row])

    def test_incomplete_candles_are_not_model_inputs_and_gaps_fail(self):
        bars=demo_candles(0)
        final=bars['y'][-1]['time']
        cutoff=datetime.fromtimestamp(final+150,timezone.utc)
        frame=candle_frame(bars,completed_at=cutoff)
        self.assertEqual(int(frame.index[-1].timestamp()),final-300)
        bars['x'].pop(10)
        with self.assertRaises(ValueError): candle_frame(bars,completed_at=cutoff)

    def test_provider_merges_history_and_intraday_and_preserves_ohlcv(self):
        u=Upstox('test'); pair=PAIRS[0]
        historic=['2026-11-13T15:20:00+05:30',100,105,98,104,1234]
        today=['2026-11-16T09:15:00+05:30',104,110,103,109,6789]
        paths=[]
        def get(path):
            paths.append(path)
            return {'candles':[today] if '/intraday/' in path else [historic]}
        with patch.object(u,'keys',return_value={pair['y']:'NSE_EQ|Y',pair['x']:'NSE_EQ|X'}),patch.object(u,'get',side_effect=get):
            bars=u.candles('NSE',pair)
        self.assertEqual(len(paths),4)
        self.assertEqual(bars['y'][-1]['volume'],6789)
        self.assertEqual(bars['y'][-1]['high'],110)
        self.assertEqual(len(bars['y']),2)


class ProfitModelTests(unittest.TestCase):
    def test_test_prices_do_not_change_model_or_validation_selection(self):
        frame=candle_frame(demo_candles(0))
        model,report=train_profit(frame)
        changed=frame.copy()
        changed.loc[changed.index[int(len(frame)*.8):],['y','y_open']]*=1.05
        other,other_report=train_profit(changed)
        probe=features(frame,report['baseline']).dropna().iloc[:10]
        np.testing.assert_array_equal(model.predict_proba(probe),other.predict_proba(probe))
        self.assertEqual(report['threshold'],other_report['threshold'])
        self.assertEqual(report['validation_pass'],other_report['validation_pass'])
        self.assertEqual(report['purge_sessions'],6)

    def test_profit_target_uses_next_open_and_both_legs_costs(self):
        dates=pd.date_range('2026-11-16T09:15:00+05:30',periods=12,freq='5min')
        frame=pd.DataFrame({'y':999.,'x':999.,'y_open':100.,'x_open':50.},index=dates)
        xx=pd.DataFrame({'zscore':-2.},index=dates)
        net=net_targets(frame,{'beta':2.},xx)
        entry_y,exit_y=100*1.0005,100*.9995
        entry_x,exit_x=50*.9995,50*1.0005
        expected=((exit_y-entry_y)-2*(exit_x-entry_x)-.0003*(entry_y+exit_y+2*(entry_x+exit_x))-2*entry_x*.1*(25/(365*24*60)))/200
        self.assertAlmostEqual(net.iloc[0],expected)
        self.assertLess(net.iloc[0],0)
        self.assertTrue(net.iloc[-1:].isna().all())
        shifted=frame.copy();shifted.index=list(dates[:5])+list(dates[5:]+pd.Timedelta(days=1))
        self.assertTrue(pd.isna(net_targets(shifted,{'beta':2.},pd.DataFrame({'zscore':-2.},index=shifted.index)).iloc[0]))


class AutomationTests(unittest.TestCase):
    def setUp(self):
        self.fixed=datetime(2026,11,16,10,0,tzinfo=IST).astimezone(timezone.utc)
        for name in ['app.market.now','app.paper.now','app.terminal.now']:
            patcher=patch(name,return_value=self.fixed);patcher.start();self.addCleanup(patcher.stop)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.app=Application(self.temp.name);self.app.mode='live';self.app.error=None
        self.t=self.app.terminal;self.pair=PAIRS[0]
        bars=demo_candles(0)
        # Align the fixture's final completed bar with this market session.
        delta=int(self.fixed.timestamp())-300-bars['y'][-1]['time']
        for leg in ('y','x'):
            for bar in bars[leg]:bar['time']+=delta
        self.t.candles={'banks':bars}
        frame=candle_frame(bars)
        info=baseline(frame[['y','x']])
        self.t.reports={'banks':{'baseline':info,'threshold':.65,'trained_at':self.fixed.isoformat()}}
        s=frame.y-info['alpha']-info['beta']*frame.x
        x=float(frame.x.iloc[-1]);y=info['alpha']+info['beta']*x+float(s.iloc[-60:].mean())-2.5*float(s.iloc[-60:].std())
        self.app.quotes={self.pair['y']:{'price':y,'time':self.fixed.isoformat()},self.pair['x']:{'price':x,'time':self.fixed.isoformat()}}
        self.t.screen=[{'id':'banks','eligible':True}]
        self.t.signals={'banks':dict(score=.9,zscore=-2.5,side='long',fresh=True,bar_time=int(self.fixed.timestamp())-300,threshold=.65,trained_at=self.fixed.isoformat())}
        self.t.configure({'enabled':True,'allocation':100000.})

    def account(self):return self.app.book.read('live:NSE')

    def test_auto_entry_is_atomic_and_only_once_per_signal(self):
        self.t.manage();self.t.manage()
        p=self.account()['positions'][0]
        self.assertEqual(len(self.account()['positions']),1)
        self.assertEqual(p['origin'],'auto');self.assertEqual(p['decision']['score'],.9)
        self.t.close(p,'Test exit')
        self.t.manage()
        self.assertFalse(self.account()['positions'])
        self.assertEqual(len(self.account()['trades']),1)
        self.assertEqual(self.account()['trades'][0]['exit_reason'],'Test exit')

    def test_no_entry_on_rejected_low_score_expired_or_stale_signals(self):
        cases=[('rejected',lambda:self.t.screen[0].update(eligible=False)),
               ('low score',lambda:self.t.signals['banks'].update(score=.4)),
               ('old candle',lambda:self.t.signals['banks'].update(fresh=False)),
               ('aged cached signal',lambda:self.t.signals['banks'].update(bar_time=int(self.fixed.timestamp())-900)),
               ('expired model',lambda:self.t.signals['banks'].update(trained_at=(self.fixed-timedelta(days=2)).isoformat())),
               ('quote error',lambda:setattr(self.app,'error','Feed failed'))]
        for name,change in cases:
            with self.subTest(name=name):
                oldscreen,oldsignal=deepcopy(self.t.screen),deepcopy(self.t.signals)
                change();self.t.manage();self.assertFalse(self.account()['positions'])
                self.t.screen,self.t.signals,self.app.error=oldscreen,oldsignal,None

    def test_stop_loss_remains_active_when_entries_paused(self):
        self.t.manage();p=self.account()['positions'][0]
        self.t.configure({'enabled':False})
        gross=sum(abs(l['qty'])*l['fill'] for l in p['legs'])
        self.app.quotes[self.pair['y']]['price']-=gross*.02/p['legs'][0]['qty']
        self.t.manage()
        self.assertFalse(self.account()['positions'])
        self.assertEqual(self.account()['trades'][0]['exit_reason'],'Stop loss')
        self.assertLess(self.account()['trades'][0]['pnl'],0)

    def test_profit_target_records_positive_net_result(self):
        self.t.manage();p=self.account()['positions'][0]
        gross=sum(abs(l['qty'])*l['fill'] for l in p['legs'])
        self.app.quotes[self.pair['y']]['price']+=gross*.03/p['legs'][0]['qty']
        self.t.manage()
        trade=self.account()['trades'][0]
        self.assertEqual(trade['exit_reason'],'Profit target');self.assertGreater(trade['pnl'],0)

    def test_feed_failure_defers_exit_until_fresh_quotes_recover(self):
        self.t.manage();p=self.account()['positions'][0]
        self.t.enabled=False
        self.app.quotes[self.pair['y']]['price']*=.8
        self.app.error='Provider down';self.t.manage();self.assertEqual(len(self.account()['positions']),1)
        self.app.error=None;self.app.quotes[self.pair['y']]['time']=(self.fixed-timedelta(minutes=2)).isoformat()
        self.t.manage();self.assertEqual(len(self.account()['positions']),1)
        self.app.quotes[self.pair['y']]['time']=self.fixed.isoformat();self.t.manage()
        self.assertFalse(self.account()['positions'])

    def test_candle_feed_error_blocks_entries_but_not_quote_driven_exits(self):
        self.t.feed_error='Candles unavailable'
        self.t.manage();self.assertFalse(self.account()['positions'])
        self.t.feed_error=None;self.t.manage()
        self.t.feed_error='Candles unavailable';self.app.quotes[self.pair['y']]['price']*=.95
        self.t.manage();self.assertFalse(self.account()['positions'])
        self.assertEqual(self.account()['trades'][0]['exit_reason'],'Stop loss')

    def test_daily_halt_is_persistent_and_not_overridden_by_enable(self):
        self.t.configure({'enabled':True,'daily_loss':100.})
        self.t.manage();self.app.quotes[self.pair['y']]['price']*=.95
        self.t.manage();self.assertFalse(self.t.enabled);self.assertTrue(self.t.day_state()['halted'])
        with self.assertRaisesRegex(ValueError,'Daily loss'):
            self.t.configure({'enabled':True})

    def test_account_switch_blocked_with_automated_positions(self):
        self.t.manage()
        with self.assertRaisesRegex(ValueError,'Close automated'):
            self.app.action('/api/connection',{'mode':'demo','exchange':'BSE'})
        self.assertEqual(self.app.account_key,'live:NSE')

    def test_local_can_enable_auto_and_restart_never_enables_entries(self):
        self.t.enabled=False;self.app.mode='demo'
        self.app.poll()
        self.t.configure({'enabled':True})
        self.assertTrue(self.t.enabled)
        reopened=Application(self.temp.name)
        self.assertFalse(reopened.terminal.enabled)
        self.assertFalse(reopened.local.running)

    def test_reconnect_same_account_with_open_positions_and_persisted_risk(self):
        self.t.configure({'enabled':True,'daily_loss':3456.})
        self.t.manage()
        reopened=Application(self.temp.name)
        reopened.mode='live';reopened.terminal.load_config()
        self.assertEqual(reopened.terminal.config['daily_loss'],3456.)
        with patch.object(reopened.provider,'quotes',return_value=deepcopy(self.app.quotes)):
            reopened.action('/api/connection',{'mode':'live','exchange':'NSE','token':'replacement-token'})
        self.assertEqual(len(reopened.book.read('live:NSE')['positions']),1)
        self.assertFalse(reopened.terminal.enabled)


if __name__=='__main__':unittest.main()
