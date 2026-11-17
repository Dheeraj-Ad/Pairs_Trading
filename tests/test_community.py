from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from app.community import parse_snapshots, fetch_snapshots, validate_url
from app.market import PAIRS
from app.server import Application


def fixture(exchange='NSE'):
    suffix='.NS' if exchange=='NSE' else '.BO'
    return dict(status='success',stocks=[dict(symbol=p[leg],exchange=exchange,ticker=p[leg]+suffix,
        last_price=100.+i,percent_change=1.,volume=1000) for i,p in enumerate(PAIRS) for leg in ('y','x')])


def wait_for(job):
    deadline=time.monotonic()+10
    while job.running and time.monotonic()<deadline:time.sleep(.01)
    if job.running:raise AssertionError('Worker did not finish')


class CommunityContractTests(unittest.TestCase):
    def test_only_requested_exchange_valid_numeric_prices_are_accepted(self):
        body=fixture()
        body['stocks'][0]['last_price']=float('nan')
        body['stocks'][1]['ticker']='ICICIBANK.BO'
        body['stocks'][2]['last_price']='1,000'
        rows,missing=parse_snapshots(body,'NSE')
        self.assertEqual(len(rows),17)
        self.assertEqual(len(missing),3)
        self.assertTrue(all(row['exchange_trade_time'] is None for row in rows.values()))

    def test_duplicate_symbol_cannot_override_a_price(self):
        body=fixture();body['stocks'].append(deepcopy(body['stocks'][0]))
        rows,missing=parse_snapshots(body,'NSE')
        self.assertNotIn('HDFCBANK',rows)
        self.assertIn('HDFCBANK',missing)

    def test_url_credentials_and_redirects_are_not_accepted(self):
        for url in ['file:///etc/passwd','https://user:secret@example.org','https://example.org/?token=x','http://example.org']:
            with self.assertRaises(ValueError):validate_url(url)
        self.assertEqual(validate_url('http://127.0.0.1:8787/'),'http://127.0.0.1:8787')

    def test_request_does_not_send_broker_secrets(self):
        opener=Mock();opener.open.return_value=io.BytesIO(json.dumps(fixture('BSE')).encode())
        with patch('app.community.urllib.request.build_opener',return_value=opener):
            rows,missing=fetch_snapshots('https://example.org','BSE')
        request=opener.open.call_args.args[0]
        self.assertNotIn('Authorization',request.headers)
        self.assertNotIn('Cookie',request.headers)
        self.assertIn('res=num',request.full_url)
        self.assertIn('M%26M.BO',request.full_url)
        self.assertEqual(len(rows),20);self.assertEqual(missing,[])


class CommunityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.app=Application(self.temp.name)

    def test_snapshots_preserve_trading_quotes_ledger_and_source(self):
        app=self.app;before=deepcopy(app.quotes);account=app.book.read(app.account_key)
        with patch('app.community.fetch_snapshots',return_value=parse_snapshots(fixture(),'NSE')):
            app.action('/api/community',dict(account_key=app.account_key,base_url='http://127.0.0.1:8787',enabled=True,refresh=True))
            wait_for(app.community)
        self.assertEqual(app.quotes,before)
        self.assertEqual(app.book.read(app.account_key),account)
        self.assertEqual(app.account_key,'demo:NSE')
        self.assertEqual(len(app.community.view()['result']['rows']),20)
        reopened=Application(self.temp.name)
        self.assertEqual(len(reopened.community.view()['result']['rows']),20)
        self.assertFalse(reopened.community.enabled)
        reopened.exchange='BSE'
        self.assertIsNone(reopened.community.view()['result'])

    def test_failure_preserves_saved_observation_without_claiming_freshness(self):
        app=self.app
        with patch('app.community.fetch_snapshots',return_value=parse_snapshots(fixture(),'NSE')):
            app.community.start();wait_for(app.community)
        before=deepcopy(app.community.view()['result'])
        app.community.next_poll=0
        with patch('app.community.fetch_snapshots',side_effect=ValueError('Provider unavailable')):
            app.community.start();wait_for(app.community)
        after=app.community.view()['result']
        self.assertEqual(after['observed_at'],before['observed_at'])
        self.assertEqual(after['rows'],before['rows'])
        self.assertEqual(after['error'],'Provider unavailable')
        self.assertIsNone(app.error)

    def test_slow_feed_does_not_lock_state_or_cross_exchanges(self):
        app=self.app;entered=threading.Event();release=threading.Event()
        def fetch(*args):
            entered.set();release.wait(5)
            return parse_snapshots(fixture(),'NSE')
        with patch('app.community.fetch_snapshots',side_effect=fetch):
            app.community.start()
            try:
                self.assertTrue(entered.wait(2))
                started=time.monotonic();app.snapshot()
                self.assertLess(time.monotonic()-started,1)
                app.exchange='BSE'
            finally:release.set();wait_for(app.community)
        self.assertIsNone(app.community.view()['result'])
        app.exchange='NSE'
        self.assertEqual(len(app.community.view()['result']['rows']),20)

    def test_demo_controls_cannot_switch_accounts_mid_run(self):
        app=self.app
        app.demo_session.job['running']=True
        with self.assertRaises(ValueError):app.action('/api/connection',dict(mode='live',exchange='NSE'))
        with self.assertRaises(ValueError):app.start_oauth({'exchange':'NSE'},app.oauth.config.origin)
        app.action('/api/demo/stop',dict(account_key=app.account_key))
        self.assertTrue(app.demo_session.cancelled)


if __name__=='__main__':unittest.main()
