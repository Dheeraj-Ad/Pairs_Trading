"""Upstox market-data GET requests only. No broker execution endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import json
import math
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
PAIRS = [
    {"id": "banks", "name": "Banking", "y": "HDFCBANK", "x": "ICICIBANK", "isin_y": "INE040A01034", "isin_x": "INE090A01021"},
    {"id": "it", "name": "Information technology", "y": "INFY", "x": "TCS", "isin_y": "INE009A01021", "isin_x": "INE467B01029"},
    {"id": "auto", "name": "Automobiles", "y": "MARUTI", "x": "M&M", "isin_y": "INE585B01010", "isin_x": "INE101A01026"},
]


def now():
    return datetime.now(timezone.utc)


def fresh(quote, as_of=None):
    try:
        age = ((as_of or now()) - datetime.fromisoformat(quote["time"])).total_seconds()
        return -5 <= age <= 60 and math.isfinite(quote["price"]) and quote["price"] > 0
    except (ValueError, KeyError, TypeError):
        return False


class Upstox:
    def __init__(self, token=""):
        self.token = token
        self.instruments = {}

    def get(self, path):
        if not self.token:
            raise ValueError("Connect an Upstox access token in Data Connection first.")
        req = urllib.request.Request("https://api.upstox.com" + path,
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"Upstox returned HTTP {exc.code}. Check the token, data access, or rate limit.") from None
        except (urllib.error.URLError, TimeoutError):
            raise ValueError("Cannot reach Upstox. Check your connection and try again.") from None
        if body.get("status") != "success":
            raise ValueError("Upstox did not return a successful data response.")
        return body["data"]

    def keys(self, exchange):
        # Resolve both exchanges from the current official master by stable ISIN.
        if exchange not in self.instruments:
            url = f"https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz"
            try:
                with urllib.request.urlopen(url, timeout=25) as response:
                    records = json.loads(gzip.decompress(response.read()))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                raise ValueError("Could not load the Upstox instrument master.") from None
            # BSE equity types are trading groups (A, B, etc.), not NSE's EQ code.
            self.instruments[exchange] = {r.get("isin"): r["instrument_key"] for r in records
                if r.get("segment") == exchange + "_EQ"
                and (exchange == "BSE" or r.get("instrument_type") == "EQ")}
        result = {}
        for pair in PAIRS:
            for leg in ("y", "x"):
                key = self.instruments[exchange].get(pair["isin_" + leg])
                if not key:
                    raise ValueError(f"{pair[leg]} was not found in the current {exchange} equity master.")
                result[pair[leg]] = key
        return result

    def quotes(self, exchange):
        if not self.token:
            raise ValueError("Connect your Upstox access token to start live polling.")
        keys = self.keys(exchange)
        data = self.get("/v3/market-quote/quotes?" + urllib.parse.urlencode({"instrument_key": ",".join(keys.values())}))
        by_key = {v["instrument_token"]: v for v in data.values()}
        result = {}
        for symbol, key in keys.items():
            if key not in by_key:
                raise ValueError(f"Upstox returned no quote for {symbol}; paper fills paused.")
            q = by_key[key]
            try:
                price = float(q["last_price"])
                stamp = datetime.fromtimestamp(float(q["last_trade_time"])/1000, timezone.utc).isoformat()
                if not math.isfinite(price) or price <= 0:
                    raise ValueError()
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                raise ValueError(f"Invalid price or trade timestamp for {symbol}.") from None
            result[symbol] = {"price": price, "time": stamp, "previous": q.get("prev_close_price"), "key": key}
        return result

    def history(self, exchange, pair):
        keys = self.keys(exchange)
        end = datetime.now(IST).date() - timedelta(days=1)
        start = end - timedelta(days=1100)
        series = []
        for leg in ("y", "x"):
            key = urllib.parse.quote(keys[pair[leg]], safe="")
            rows = self.get(f"/v3/historical-candle/{key}/days/1/{end}/{start}")["candles"]
            dates = [pd.Timestamp(r[0]).date().isoformat() for r in rows]
            series.append(pd.Series([float(r[4]) for r in rows], index=dates, name=leg))
        frame = pd.concat(series, axis=1).sort_index()
        if frame.isna().any().any():
            raise ValueError("The legs have missing or unmatched sessions. History was not forward-filled.")
        if frame.empty or (end - pd.Timestamp(frame.index[-1]).date()).days > 7:
            raise ValueError("Historical candles are missing or stale.")
        return frame

    def candles(self, exchange, pair, intraday_only=False):
        """Actual exchange OHLCV, five-minute intervals; never synthesize live candles."""
        keys = self.keys(exchange)
        today = datetime.now(IST).date()
        result = {}
        for leg in ("y", "x"):
            key = urllib.parse.quote(keys[pair[leg]], safe="")
            rows = []
            if not intraday_only:
                end, start = today-timedelta(days=1), today-timedelta(days=27)
                rows += self.get(f"/v3/historical-candle/{key}/minutes/5/{end}/{start}")["candles"]
            rows += self.get(f"/v3/historical-candle/intraday/{key}/minutes/5")["candles"]
            result[leg] = normalize_candles(rows)
        return result


def normalize_candles(rows):
    result = {}
    for row in rows:
        stamp = pd.Timestamp(row[0])
        if stamp.tzinfo is None:
            raise ValueError("Candle timestamps must include timezone information.")
        values = [float(v) for v in row[1:6]]
        o, h, l, c, v = values
        if not all(math.isfinite(x) for x in values) or min(o,h,l,c) <= 0 or v < 0 or not l <= min(o,c) <= max(o,c) <= h:
            raise ValueError("Provider returned invalid OHLCV data.")
        epoch = int(stamp.timestamp())
        result[epoch] = dict(time=epoch, open=o, high=h, low=l, close=c, volume=v)
    return [result[t] for t in sorted(result)]


def candle_frame(candles, completed_at=None):
    cutoff = (completed_at or now()).timestamp()
    series = []
    for leg in ("y", "x"):
        data = {r["time"]: r for r in candles[leg] if r["time"]+300 <= cutoff}
        frame = pd.DataFrame.from_dict(data, orient="index")
        if frame.empty:
            raise ValueError("No completed five-minute candles are available.")
        frame.index = pd.to_datetime(frame.index, unit="s", utc=True)
        series.append(frame[["open", "close"]].rename(columns={"open":leg+"_open", "close":leg}))
    frame = pd.concat(series, axis=1).sort_index()
    if frame.isna().any().any() or frame.index.has_duplicates:
        raise ValueError("Pair candles are not aligned. Missing legs are not forward-filled.")
    return frame


def demo_candles(pair_index):
    """Explicitly synthetic five-minute OHLCV for offline tests and demo mode."""
    frame = demo_history(pair_index)
    # Reuse a reproducible correlated process on a synthetic intraday calendar.
    times = []
    for date in pd.bdate_range(end=datetime.now(IST).date()-timedelta(days=1), periods=10):
        start = datetime.combine(date.date(), datetime.min.time(), tzinfo=IST).replace(hour=9, minute=15)
        times.extend(int((start+timedelta(minutes=5*i)).timestamp()) for i in range(72))
    rng = np.random.default_rng(200+pair_index)
    out = {}
    for leg in ("y", "x"):
        rows = []
        for i, (stamp, close) in enumerate(zip(times, frame[leg])):
            opening = float(frame[leg].iloc[max(0, i-1)])
            wiggle = float(close)*rng.uniform(.0001,.002)
            rows.append(dict(time=stamp, open=opening, high=max(opening,close)+wiggle,
                             low=min(opening,close)-wiggle, close=float(close), volume=int(rng.integers(1000,20000))))
        out[leg] = rows
    return out


def demo_history(pair_index):
    rng = np.random.default_rng(42 + pair_index)
    n = 720
    base = [1300, 3000, 2800][pair_index]
    beta = [1.2, .45, 3.3][pair_index]
    x = base + np.cumsum(rng.normal(0, base*.004, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = .87 * residual[i-1] + rng.normal(0, base*.006)
    y = 80 + beta*x + residual
    dates = pd.bdate_range(end=datetime.now(IST).date()-timedelta(days=1), periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"y": y, "x": x}, index=dates)
