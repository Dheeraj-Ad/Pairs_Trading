"""Persistent synthetic OHLCV stream. No broker, network or future price lookup."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import re
import time
import uuid

import numpy as np

from app.market import PAIRS, IST, DEMO_PARAMETERS, demo_candles, now


def shift_demo_dates(value, seconds):
    """Move synthetic timestamps together, preserving prices and elapsed durations."""
    if isinstance(value, list):
        return [shift_demo_dates(item, seconds) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in ('opened', 'closed', 'trained_at') and isinstance(item, str):
            result[key] = (datetime.fromisoformat(item) - timedelta(seconds=seconds)).isoformat()
        elif key in ('time', 'bar_time') and isinstance(item, (int, float)):
            result[key] = item - seconds
        else:
            result[key] = shift_demo_dates(item, seconds)
    return result


class LocalMarket:
    def __init__(self, book):
        self.book = book
        self.running = False  # Restart never advances an unattended simulation.
        self.remaining = 0
        self.batch_active = False
        self.last_step = time.monotonic()
        self.cache = {}
        with book.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS local_markets(account TEXT PRIMARY KEY, body TEXT NOT NULL)')
        self.normalize_dates()

    def normalize_dates(self):
        """Atomically repair legacy future demo calendars before caches are loaded.

        Whole weeks preserve weekdays, trading times, holding periods and P&L.
        Live accounts and real observation timestamps are never rewritten.
        """
        with self.book.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            cutoff = now().timestamp()
            for account, body in db.execute("SELECT account,body FROM local_markets WHERE account LIKE 'demo:%'").fetchall():
                market = json.loads(body)
                latest = max(bar['time'] + 300 for pair in market['bars'].values() for bars in pair.values() for bar in bars)
                if latest <= cutoff:
                    continue
                seconds = math.ceil((latest - cutoff) / (7 * 86400)) * 7 * 86400
                market = shift_demo_dates(market, seconds)
                market['calendar_shift_seconds'] = market.get('calendar_shift_seconds', 0) + seconds
                db.execute('UPDATE local_markets SET body=? WHERE account=?', (json.dumps(market), account))
                row = db.execute('SELECT body FROM accounts WHERE id=?', (account,)).fetchone()
                if row:
                    original = json.loads(row[0])
                    requests = []
                    for trade in original['trades'] + original['positions']:
                        bar = (trade.get('decision') or {}).get('bar_time')
                        if trade.get('origin') != 'auto' or not isinstance(bar, (int, float)):
                            continue
                        old_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{account}/{trade['pair']}/{bar}"))
                        new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{account}/{trade['pair']}/{bar - seconds}"))
                        if db.execute('SELECT 1 FROM requests WHERE id=? AND account=?', (old_id, account)).fetchone():
                            requests.append((old_id, new_id))
                    # Remove old keys first so overlapping calendar ranges cannot collide.
                    db.executemany('DELETE FROM requests WHERE id=? AND account=?', [(old, account) for old, new in requests])
                    db.executemany('INSERT INTO requests VALUES(?,?)', [(new, account) for old, new in requests])
                    ledger = shift_demo_dates(original, seconds)
                    db.execute('UPDATE accounts SET body=? WHERE id=?', (json.dumps(ledger), account))
                if 'automation_events' in tables:
                    for event_id, stamp, message in db.execute('SELECT id,time,message FROM automation_events WHERE account=?', (account,)).fetchall():
                        stamp = (datetime.fromisoformat(stamp) - timedelta(seconds=seconds)).isoformat()
                        message = re.sub(r'completed candle (\d+)', lambda m: 'completed candle ' + str(int(m[1]) - seconds), message)
                        db.execute('UPDATE automation_events SET time=?,message=? WHERE id=?', (stamp, message, event_id))
                if 'automation_days' in tables:
                    days = db.execute('SELECT day,equity,halted FROM automation_days WHERE account=?', (account,)).fetchall()
                    db.execute('DELETE FROM automation_days WHERE account=?', (account,))
                    for day, equity, halted in days:
                        day = (datetime.fromisoformat(day) - timedelta(seconds=seconds)).date().isoformat()
                        db.execute('INSERT INTO automation_days VALUES(?,?,?,?)', (account, day, equity, halted))
                if 'paper_models' in tables:
                    row = db.execute('SELECT report FROM paper_models WHERE account=?', (account,)).fetchone()
                    if row:
                        db.execute('UPDATE paper_models SET report=? WHERE account=?',
                                   (json.dumps(shift_demo_dates(json.loads(row[0]), seconds)), account))

    def state(self, account):
        if account not in self.cache:
            with self.book.connect() as db:
                row = db.execute('SELECT body FROM local_markets WHERE account=?', (account,)).fetchone()
            self.cache[account] = json.loads(row[0]) if row else {
                'step': 0, 'bars': {p['id']: demo_candles(i) for i, p in enumerate(PAIRS)}}
            # Extend existing simulations without rewinding their clock or replacing
            # any candles/prices already used for fills and model decisions.
            state = self.cache[account]
            reference = state['bars']['banks']['y'][-720:]
            changed = False
            for i, pair in enumerate(PAIRS):
                if pair['id'] in state['bars']: continue
                bars = demo_candles(i)
                for leg in ('y', 'x'):
                    bars[leg] = bars[leg][-len(reference):]
                    for bar, old in zip(bars[leg], reference): bar['time'] = old['time']
                state['bars'][pair['id']] = bars
                changed = True
            if not row or changed:
                self.save(account)
        return self.cache[account]

    def save(self, account):
        with self.book.connect() as db:
            db.execute('INSERT OR REPLACE INTO local_markets VALUES(?,?)',
                       (account, json.dumps(self.cache[account], allow_nan=False)))

    def clock(self, account):
        bars = self.state(account)['bars']['banks']['y']
        return datetime.fromtimestamp(bars[-1]['time'] + 300, timezone.utc)

    def candles(self, account, pair):
        return deepcopy(self.state(account)['bars'][pair])

    def quotes(self, account):
        state, stamp = self.state(account), self.clock(account).isoformat()
        quotes = {}
        for pair in PAIRS:
            for leg in ('y', 'x'):
                rows = state['bars'][pair['id']][leg]
                quotes[pair[leg]] = dict(price=rows[-1]['close'], previous=rows[-2]['close'], time=stamp)
        return quotes

    def advance(self, account):
        state = self.state(account)
        opening = self.clock(account).astimezone(IST)
        if opening.hour * 60 + opening.minute >= 15 * 60 + 30:
            opening = (opening + timedelta(days=1)).replace(hour=9, minute=15)
            while opening.weekday() >= 5:
                opening += timedelta(days=1)
        if opening + timedelta(minutes=5) > now():
            self.running = False
            self.remaining = 0
            raise ValueError('Local simulation has reached the present. Wait for another completed market interval before advancing.')
        for i, pair in enumerate(PAIRS):
            seed = 100000 + state['step'] * 3 + i if i < 3 else np.random.SeedSequence([100000, i, state['step']])
            rng = np.random.default_rng(seed)
            bars = state['bars'][pair['id']]
            py, px = bars['y'][-1]['close'], bars['x'][-1]['close']
            base, beta = DEMO_PARAMETERS[i]
            x = max(100, px + rng.normal(0, base * .004))
            residual = .87 * (py - 80 - beta * px) + rng.normal(0, base * .006)
            y = max(100, 80 + beta * x + residual)
            for leg, previous, close in [('y', py, y), ('x', px, x)]:
                wiggle = close * rng.uniform(.0001, .002)
                bars[leg].append(dict(time=int(opening.timestamp()), open=previous,
                    high=max(previous, close) + wiggle, low=min(previous, close) - wiggle,
                    close=float(close), volume=int(rng.integers(1000, 20000))))
                bars[leg] = bars[leg][-3000:]
        state['step'] += 1
        self.save(account)

    def view(self, account):
        return dict(running=self.running, remaining=self.remaining, batch_active=self.batch_active, time=self.clock(account).isoformat(),
                    steps=self.state(account)['step'], seconds_per_bar=2,
                    source='Generated synthetic prices; not historical exchange data')
