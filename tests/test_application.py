"""Behavioral tests for money, data provenance, chronology, and local HTTP access."""
from copy import deepcopy
from datetime import timedelta
import gzip
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import numpy as np

from app.market import PAIRS, Upstox, demo_history, now
from app.paper import PaperBook, CAPITAL, FEE, SLIPPAGE
from app.server import Application, handler_for
from ml.research import baseline, train


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.book = PaperBook(Path(self.temp.name)/"paper.db")
        self.pair = PAIRS[0]
        self.quotes = {self.pair["y"]: {"price": 100., "time": now().isoformat()},
                       self.pair["x"]: {"price": 50., "time": now().isoformat()}}

    def open(self, **kwargs):
        args = dict(mode="demo:NSE", quotes=self.quotes, request_id=str(uuid.uuid4()), pair=self.pair,
                    side="long", budget=10000., beta=2.)
        args.update(kwargs)
        self.book.trade(**args)
        return self.book.read("demo:NSE")

    def test_round_trip_costs_and_cash_reconcile(self):
        account = self.open()
        p = account["positions"][0]
        self.assertEqual([l["qty"] for l in p["legs"]], [49, -99])
        expected_entry_fee = sum(abs(l["qty"])*l["fill"] for l in p["legs"])*FEE
        self.assertAlmostEqual(account["balance"], CAPITAL-expected_entry_fee)
        self.quotes[self.pair["y"]]["price"] = 110.
        self.quotes[self.pair["x"]]["price"] = 48.
        marked = self.book.mark(account, self.quotes)
        expected_mark = sum(l["qty"]*(self.quotes[l["symbol"]]["price"]-l["fill"]) for l in p["legs"])
        self.assertAlmostEqual(marked["unrealized"], expected_mark-marked["positions"][0]["borrow"])
        self.book.trade("demo:NSE", self.quotes, str(uuid.uuid4()), close_id=p["id"])
        final = self.book.read("demo:NSE")
        self.assertFalse(final["positions"])
        trade = final["trades"][0]
        gross = sum(l["qty"]*(e["fill"]-l["fill"]) for l,e in zip(p["legs"],trade["exit_legs"]))
        self.assertAlmostEqual(trade["pnl"], gross-trade["fees"]-trade["borrow"])
        self.assertAlmostEqual(final["balance"]-CAPITAL, trade["pnl"])
        self.assertEqual(PaperBook(self.book.path).read("demo:NSE"), final)

    def test_short_spread_has_reversed_legs_and_pays_round_trip_costs(self):
        a = self.open(side="short")
        p = a["positions"][0]
        self.assertLess(p["legs"][0]["qty"], 0)
        self.assertGreater(p["legs"][1]["qty"], 0)
        self.book.trade("demo:NSE", self.quotes, str(uuid.uuid4()), close_id=p["id"])
        self.assertLess(self.book.read("demo:NSE")["trades"][0]["pnl"], 0)

    def test_duplicate_request_never_duplicates_a_fill(self):
        request_id = str(uuid.uuid4())
        first = self.open(request_id=request_id)
        second = self.open(request_id=request_id)
        self.assertEqual(first, second)
        close_id, request_id = first["positions"][0]["id"], str(uuid.uuid4())
        self.book.trade("demo:NSE", self.quotes, request_id, close_id=close_id)
        self.book.trade("demo:NSE", self.quotes, request_id, close_id=close_id)
        self.assertEqual(len(self.book.read("demo:NSE")["trades"]), 1)

    def test_stale_missing_future_and_out_of_sync_quotes_reject_atomic_fill(self):
        for delta in [-90, 30, -20]:
            with self.subTest(delta=delta):
                self.quotes[self.pair["x"]]["time"]=(now()+timedelta(seconds=delta)).isoformat()
                with self.assertRaises(ValueError):
                    self.open()
                self.assertEqual(self.book.read("demo:NSE")["balance"], CAPITAL)
                self.assertFalse(self.book.read("demo:NSE")["positions"])
        self.quotes.pop(self.pair["x"])
        with self.assertRaises(ValueError):
            self.open()

    def test_limits_and_invalid_budgets(self):
        for budget in [float('nan'), float('inf'), -1, 1, 600000, "10000"]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.open(budget=budget)
        self.open()
        with self.assertRaises(ValueError):
            self.open()

    def test_accounts_are_isolated(self):
        self.open()
        for key in ["live:NSE", "demo:BSE", "live:BSE"]:
            self.assertEqual(self.book.read(key)["balance"], CAPITAL)
            self.assertFalse(self.book.read(key)["positions"])

    def test_stale_close_preserves_position(self):
        p = self.open()["positions"][0]
        self.quotes[self.pair["y"]]["time"]=(now()-timedelta(minutes=5)).isoformat()
        with self.assertRaises(ValueError):
            self.book.trade("demo:NSE", self.quotes, str(uuid.uuid4()), close_id=p["id"])
        self.assertEqual(len(self.book.read("demo:NSE")["positions"]), 1)

    def test_concurrent_duplicate_is_atomic(self):
        request = str(uuid.uuid4())
        errors = []
        def submit():
            try:
                self.open(request_id=request)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=submit) for _ in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertFalse(errors)
        self.assertEqual(len(self.book.read("demo:NSE")["positions"]), 1)


class ResearchTests(unittest.TestCase):
    def test_future_test_prices_cannot_change_fit_or_threshold(self):
        frame = demo_history(0)
        model, report = train(frame)
        changed = frame.copy()
        changed.iloc[int(len(frame)*.8):, 0] *= 1.05
        other_model, other = train(changed)
        probes = np.array([[0,0,0,.2], [2,2,-.1,.3], [-2,2,.2,.5]])
        import pandas as pd
        probes = pd.DataFrame(probes, columns=model.feature_names_in_)
        np.testing.assert_equal(model.predict_proba(probes), other_model.predict_proba(probes))
        self.assertEqual(report["threshold"], other["threshold"])
        self.assertEqual(report["baseline"]["beta"], other["baseline"]["beta"])
        self.assertEqual([report[k] for k in ["train_rows", "validation_rows", "test_rows"]], [211,139,139])

    def test_history_rejects_missing_duplicate_nonpositive_and_short_inputs(self):
        frame = demo_history(0)
        invalids = [frame.iloc[:100], frame.iloc[list(range(719))+[718]]]
        for bad in [np.nan, 0, -1, np.inf]:
            changed = frame.copy()
            changed.iloc[20,0] = bad
            invalids.append(changed)
        for frame in invalids:
            with self.assertRaises(ValueError): baseline(frame)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app=Application(self.temp.name)

    def test_no_silent_live_fallback_and_no_secret_in_state(self):
        self.app.provider.token='test-secret'
        with patch.object(self.app.provider, 'quotes', side_effect=ValueError('Rejected')):
            with self.assertRaises(ValueError):
                self.app.action('/api/connection', {'mode':'live','exchange':'NSE','token':'new-secret'})
        self.assertEqual(self.app.mode,'demo')
        self.assertEqual(self.app.provider.token,'test-secret')
        self.assertNotIn('test-secret',json.dumps(self.app.snapshot()))
        self.app.mode='live'
        with patch.object(self.app.provider,'quotes',side_effect=ValueError('Unavailable')):
            self.app.poll()
        self.assertEqual(self.app.mode,'live')
        self.assertTrue(self.app.error)
        with self.assertRaises(ValueError):
            self.app.action('/api/open', {'account_key':'live:NSE','pair':'banks','side':'long','budget':100000,'request_id':str(uuid.uuid4())})

    def test_mode_switch_and_old_tab_rejection(self):
        self.app.action('/api/connection',{'mode':'demo','exchange':'BSE'})
        with self.assertRaises(ValueError):
            self.app.action('/api/open', {'account_key':'demo:NSE','pair':'banks'})
        self.assertEqual(self.app.snapshot()['account_key'],'demo:BSE')

    def test_provider_parses_documented_quotes_and_rejects_missing_leg(self):
        provider=Upstox('test')
        keys={p[l]: 'NSE_EQ|'+p['isin_'+l] for p in PAIRS for l in ('y','x')}
        data={s:{'instrument_token':k,'last_price':100.,'last_trade_time':str(int(now().timestamp()*1000)), 'prev_close_price':99.} for s,k in keys.items()}
        with patch.object(provider,'keys',return_value=keys), patch.object(provider,'get',return_value=data):
            result=provider.quotes('NSE')
            self.assertEqual(len(result),len(keys))
            self.assertEqual(result['HDFCBANK']['price'],100.)
            data.pop('HDFCBANK')
            with self.assertRaises(ValueError): provider.quotes('NSE')

    def test_bse_master_resolves_equity_group_a_by_isin(self):
        rows=[{'isin':p['isin_'+leg], 'instrument_key':'BSE_EQ|'+p['isin_'+leg],
               'segment':'BSE_EQ','instrument_type':'A'} for p in PAIRS for leg in ('y','x')]
        compressed=gzip.compress(json.dumps(rows).encode())
        with patch('urllib.request.urlopen',return_value=io.BytesIO(compressed)):
            keys=Upstox().keys('BSE')
        self.assertEqual(keys['HDFCBANK'],'BSE_EQ|INE040A01034')
        self.assertEqual(len(keys),len({p[leg] for p in PAIRS for leg in ('y','x')}))

    def test_http_security_and_full_paper_roundtrip(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.app))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base='http://127.0.0.1:'+str(server.server_port)
        def request(path,data=None,csrf=True,origin=None):
            headers={}
            if data is not None:
                headers['Content-Type']='application/json'
                if csrf: headers['X-CSRF-Token']=self.app.csrf
            if origin: headers['Origin']=origin
            req=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
            with urllib.request.urlopen(req,timeout=10) as response: return response.read()
        self.assertIn(b'Pairwise',request('/'))
        self.assertIn(b'Content', request('/app.js'))
        state=json.loads(request('/api/state'))
        payload={'account_key':state['account_key'],'pair':'banks','side':'long','budget':100000,'request_id':str(uuid.uuid4())}
        for csrf,origin in [(False,None),(True,'https://evil.example')]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                request('/api/open',payload,csrf,origin)
            self.assertEqual(error.exception.code,403)
        opened=json.loads(request('/api/open',payload))
        self.assertEqual(len(opened['account']['positions']),1)
        closed=json.loads(request('/api/close',{'account_key':state['account_key'],'position_id':opened['account']['positions'][0]['id'],'request_id':str(uuid.uuid4())}))
        self.assertEqual(len(closed['account']['trades']),1)
        self.assertIn(b'banks', request('/api/export'))
        report=json.loads(request('/api/train',{'account_key':state['account_key'],'pair':'banks'}))
        self.assertIsNotNone(report['pairs'][0]['report'])
        self.assertTrue((Path(self.temp.name)/'model-demo-banks.joblib').exists())


if __name__ == '__main__':
    unittest.main()
