"""Persistent synthetic OHLCV stream. No broker, network or future price lookup."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import time

import numpy as np

from app.market import PAIRS, IST, demo_candles


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

    def state(self, account):
        if account not in self.cache:
            with self.book.connect() as db:
                row = db.execute('SELECT body FROM local_markets WHERE account=?', (account,)).fetchone()
            self.cache[account] = json.loads(row[0]) if row else {
                'step': 0, 'bars': {p['id']: demo_candles(i) for i, p in enumerate(PAIRS)}}
            if not row:
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
        for i, pair in enumerate(PAIRS):
            rng = np.random.default_rng(100000 + state['step'] * 3 + i)
            bars = state['bars'][pair['id']]
            py, px = bars['y'][-1]['close'], bars['x'][-1]['close']
            base, beta = [1300, 3000, 2800][i], [1.2, .45, 3.3][i]
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
