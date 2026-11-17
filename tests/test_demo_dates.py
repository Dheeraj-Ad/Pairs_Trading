from copy import deepcopy
from datetime import datetime, timedelta
import json
import tempfile
import unittest
from unittest.mock import patch
import uuid

from app.market import IST, now
from app.server import Application


class DemoDateTests(unittest.TestCase):
    def test_future_calendar_migrates_all_saved_dates_without_changing_money(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(directory)
            key = app.account_key
            app.action('/api/open', dict(account_key=key, pair='banks', side='long',
                                        budget=100000, request_id=str(uuid.uuid4())))
            position = app.book.read(key)['positions'][0]
            app.advance_local(3)
            app.action('/api/close', dict(account_key=key, position_id=position['id'], request_id=str(uuid.uuid4())))
            app.action('/api/open', dict(account_key=key, pair='it', side='short',
                                        budget=100000, request_id=str(uuid.uuid4())))
            market = deepcopy(app.local.state(key))
            ledger = app.book.read(key)
            offset = timedelta(weeks=10)
            for pair in market['bars'].values():
                for bars in pair.values():
                    for bar in bars:
                        bar['time'] += int(offset.total_seconds())
            for trade in ledger['trades'] + ledger['positions']:
                for field in ('opened', 'closed'):
                    if field in trade:
                        trade[field] = (datetime.fromisoformat(trade[field]) + offset).isoformat()
            old_clock = datetime.fromtimestamp(market['bars']['banks']['y'][-1]['time'] + 300, IST)
            old_bar = int(old_clock.timestamp()) - 300
            ledger['trades'][0]['decision'] = {'bar_time': old_bar, 'trained_at': old_clock.isoformat()}
            ledger['trades'][0]['origin'] = 'auto'
            old_request = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{key}/banks/{old_bar}'))
            live_body = json.dumps(ledger)
            with app.book.connect() as db:
                db.execute('UPDATE local_markets SET body=? WHERE account=?', (json.dumps(market), key))
                db.execute('UPDATE accounts SET body=? WHERE id=?', (json.dumps(ledger), key))
                db.execute('INSERT INTO accounts VALUES(?,?)', ('live:NSE', live_body))
                db.execute('INSERT INTO requests VALUES(?,?)', (old_request, key))
                db.execute('INSERT INTO automation_events(account,time,kind,pair,message) VALUES(?,?,?,?,?)',
                           (key, old_clock.isoformat(), 'entry', 'banks', f'completed candle {old_bar}'))
                for day in (old_clock.date(), old_clock.date() - timedelta(weeks=2)):
                    db.execute('INSERT INTO automation_days VALUES(?,?,?,?)', (key, day.isoformat(), 1000000, 1))
                db.execute('INSERT INTO paper_models VALUES(?,?)', (key, json.dumps({'trained_at': old_clock.isoformat()})))
            before_mark = app.book.mark(ledger, app.quotes, old_clock)
            reopened = Application(directory)
            new_clock = reopened.market_now()
            shift = old_clock - new_clock
            self.assertLessEqual(new_clock, now())
            self.assertGreater(shift.total_seconds(), 0)
            self.assertEqual(shift.total_seconds() % (7 * 86400), 0)
            updated = reopened.book.read(key)
            after_mark = reopened.book.mark(updated, reopened.quotes, new_clock)
            self.assertEqual(before_mark['equity'], after_mark['equity'])
            for old, new in zip(ledger['trades'] + ledger['positions'], updated['trades'] + updated['positions']):
                for field in ('opened', 'closed'):
                    if field in old:
                        self.assertEqual(datetime.fromisoformat(old[field]) - datetime.fromisoformat(new[field]), shift)
                for field in ('pnl', 'fees', 'borrow', 'legs', 'entry_legs', 'exit_legs', 'id'):
                    self.assertEqual(old.get(field), new.get(field))
            self.assertEqual(updated['trades'][0]['decision']['bar_time'], old_bar - shift.total_seconds())
            with reopened.book.connect() as db:
                new_request = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{key}/banks/{old_bar - int(shift.total_seconds())}'))
                self.assertIsNone(db.execute('SELECT 1 FROM requests WHERE id=?', (old_request,)).fetchone())
                self.assertIsNotNone(db.execute('SELECT 1 FROM requests WHERE id=? AND account=?', (new_request, key)).fetchone())
                self.assertEqual(db.execute('SELECT body FROM accounts WHERE id=?', ('live:NSE',)).fetchone()[0], live_body)
                event = db.execute('SELECT time,message FROM automation_events WHERE account=?', (key,)).fetchone()
                self.assertEqual(datetime.fromisoformat(event[0]), new_clock)
                self.assertEqual(event[1], f'completed candle {old_bar - int(shift.total_seconds())}')
                days = db.execute('SELECT day,halted FROM automation_days WHERE account=? ORDER BY day', (key,)).fetchall()
                self.assertEqual(len(days), 2)
                self.assertTrue(all(halted == 1 and datetime.fromisoformat(day).date() <= now().date() for day, halted in days))
                report = json.loads(db.execute('SELECT report FROM paper_models WHERE account=?', (key,)).fetchone()[0])
                self.assertEqual(datetime.fromisoformat(report['trained_at']), new_clock)
            again = Application(directory)
            self.assertEqual(again.local.state(key), reopened.local.state(key))
            self.assertEqual(again.book.read(key), updated)

    def test_advance_stops_at_present_without_mutating_candles(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(directory)
            key = app.account_key
            self.assertLess(app.market_now(), now() - timedelta(days=20))
            before = deepcopy(app.local.state(key))
            clock = app.market_now()
            app.local.running = True
            app.local.remaining = 75
            with patch('app.local_market.now', return_value=clock):
                with self.assertRaisesRegex(ValueError, 'reached the present'):
                    app.advance_local()
            self.assertEqual(app.local.state(key), before)
            self.assertFalse(app.local.running)
            self.assertEqual(app.local.remaining, 0)
            with patch('app.local_market.now', return_value=clock + timedelta(minutes=5)):
                app.advance_local()
            self.assertEqual(app.market_now(), clock + timedelta(minutes=5))


if __name__ == '__main__':
    unittest.main()
