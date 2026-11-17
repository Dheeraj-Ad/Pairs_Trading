"""Groww authentication and read-only CASH market data; no broker order routes."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import csv
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from app.market import IST, PAIRS, USER_AGENT, normalize_candles, now
from app.oauth import NoRedirect


@dataclass(repr=False)
class GrowwConfig:
    auth_mode: str = 'approval'
    api_key: str = ''
    api_secret: str = ''
    totp_secret: str = ''
    access_token: str = ''

    @classmethod
    def load(cls, root):
        names = {name: 'GROWW_' + name.upper() for name in cls.__dataclass_fields__}
        values = {}
        path = Path(root) / '.env'
        if path.is_file():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                if key.strip() in names.values():
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    values[key.strip()] = value
        values.update({key: os.environ[key] for key in names.values() if key in os.environ})
        return cls(**{name: values[key] for name, key in names.items() if key in values})

    @property
    def configured(self):
        return bool(self.access_token or (self.api_key and (
            self.api_secret if self.auth_mode == 'approval' else self.totp_secret)))


def totp(secret, timestamp=None):
    """Standard RFC 6238 SHA-1, six digits, 30 seconds; never reinterpret bad secrets."""
    try:
        value = secret.strip().upper()
        key = base64.b32decode(value + '=' * ((-len(value)) % 8))
        if not key:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('GROWW_TOTP_SECRET must be the Base32 secret from the Groww TOTP setup, not an API Secret.') from None
    counter = int(time.time() if timestamp is None else timestamp) // 30
    digest = hmac.new(key, struct.pack('>Q', counter), hashlib.sha1).digest()
    offset = digest[-1] & 15
    return f'{(struct.unpack(">I", digest[offset:offset+4])[0] & 0x7fffffff) % 1000000:06d}'


class Groww:
    def __init__(self, config=None, token=''):
        self.config = config or GrowwConfig()
        self.token = token or self.config.access_token
        self.instruments = {}
        self._rate_lock = threading.Lock()
        self._next_request = 0.
        self.blocked = False

    def _request(self, path, token, body=None):
        # Fixed origin + no redirects prevent forwarding credentials to another host.
        allowed = ('/v1/live-data/quote?', '/v1/historical/candles?')
        if not (body is None and path.startswith(allowed)) and not (body is not None and path == '/v1/token/api/access'):
            raise ValueError('Only Groww authentication and market-data requests are allowed.')
        with self._rate_lock:
            time.sleep(max(0., self._next_request-time.monotonic()))
            self._next_request = time.monotonic() + .25
        headers = {'Authorization': 'Bearer '+token, 'Accept': 'application/json',
                   'X-API-VERSION': '1.0', 'User-Agent': USER_AGENT}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request('https://api.groww.in'+path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403) and body is None:
                self.blocked = True
            messages = {
                401: 'Groww rejected authentication. Check that the API key and secret belong to the same login method, or use a fresh access token.',
                403: 'Groww denied access. Check daily API-key approval and the market-data subscription in Groww.',
                429: 'Groww rate limit reached. Wait before reconnecting.',
            }
            raise ValueError(messages.get(exc.code, f'Groww returned HTTP {exc.code}. Check the credentials and API access.')) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValueError('Cannot reach Groww. Check the connection and retry.') from None
        except (ValueError, TypeError):
            raise ValueError('Groww returned an invalid response.') from None
        if not isinstance(data, dict):
            raise ValueError('Groww returned an invalid response.')
        if body is not None:
            token = data.get('token')
            if not isinstance(token, str) or not token or data.get('isActive') is False:
                raise ValueError('Groww did not issue an access token. Check the key type, matching secret and daily approval.')
            return token
        if data.get('status') != 'SUCCESS' or not isinstance(data.get('payload'), dict):
            raise ValueError('Groww did not return successful market data. Check your data subscription and credentials.')
        return data['payload']

    def authenticate(self):
        if self.token:
            return
        cfg = self.config
        if not cfg.api_key:
            raise ValueError('Configure Groww in .env or enter a Groww access token in Data connection.')
        if cfg.auth_mode == 'approval' and cfg.api_secret:
            timestamp = str(int(time.time()))
            body = dict(key_type='approval', timestamp=timestamp,
                        checksum=hashlib.sha256((cfg.api_secret+timestamp).encode()).hexdigest())
        elif cfg.auth_mode == 'totp' and cfg.totp_secret:
            body = dict(key_type='totp', totp=totp(cfg.totp_secret))
        else:
            raise ValueError('Set GROWW_AUTH_MODE to approval with API Secret, or totp with TOTP Secret.')
        self.token = self._request('/v1/token/api/access', cfg.api_key, body)
        self.blocked = False

    def get(self, path, params):
        if not self.token or self.blocked:
            raise ValueError('Reconnect Groww in Data connection before fetching market data.')
        return self._request(path+'?'+urllib.parse.urlencode(params), self.token)

    def keys(self, exchange):
        if exchange not in ('NSE', 'BSE'):
            raise ValueError('Choose NSE or BSE.')
        if exchange not in self.instruments:
            request = urllib.request.Request('https://growwapi-assets.groww.in/instruments/instrument.csv',
                headers={'User-Agent': USER_AGENT})
            try:
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
                    records = csv.DictReader(io.TextIOWrapper(response, encoding='utf-8-sig'))
                    by_isin = {}
                    for row in records:
                        if row.get('exchange') == exchange and row.get('segment') == 'CASH' and row.get('isin'):
                            if exchange == 'NSE' and row.get('series') != 'EQ':
                                continue
                            by_isin.setdefault(row['isin'], []).append(row)
            except (urllib.error.URLError, OSError, ValueError):
                raise ValueError('Could not load the Groww instrument master.') from None
            resolved = {}
            for pair in PAIRS:
                for leg in ('y', 'x'):
                    matches = by_isin.get(pair['isin_'+leg], [])
                    if len(matches) != 1 or not matches[0].get('trading_symbol') or not matches[0].get('groww_symbol'):
                        raise ValueError(f'Groww has no unique {exchange} equity match for {pair[leg]}.')
                    resolved[pair[leg]] = matches[0]
            self.instruments[exchange] = resolved
        return self.instruments[exchange]

    def quote(self, exchange, trading_symbol):
        data = self.get('/v1/live-data/quote', dict(exchange=exchange, segment='CASH', trading_symbol=trading_symbol))
        try:
            price, stamp = data['last_price'], data['last_trade_time']
            if isinstance(price, bool) or isinstance(stamp, bool):
                raise ValueError()
            price, stamp = float(price), float(stamp)
            if not math.isfinite(price) or price <= 0 or not math.isfinite(stamp) or stamp <= 0:
                raise ValueError()
            stamp = datetime.fromtimestamp(stamp/1000, timezone.utc).isoformat()
        except (KeyError, ValueError, TypeError, OverflowError, OSError):
            raise ValueError(f'Groww returned an invalid price or exchange trade time for {trading_symbol}.') from None
        return dict(price=price, time=stamp, previous=None, key=exchange+'-'+trading_symbol)

    def quotes(self, exchange):
        return {symbol: self.quote(exchange, row['trading_symbol']) for symbol, row in self.keys(exchange).items()}

    def _candles(self, exchange, symbol, start, end, interval):
        data = self.get('/v1/historical/candles', dict(exchange=exchange, segment='CASH',
            groww_symbol=symbol, start_time=start.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=end.strftime('%Y-%m-%d %H:%M:%S'), candle_interval=interval))
        rows = data.get('candles')
        if not isinstance(rows, list):
            raise ValueError('Groww returned no candle array.')
        normalized = []
        for row in rows:
            try:
                stamp = pd.Timestamp(row[0])
                # Groww returns exchange-local timestamps without an offset.
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize(IST)
                normalized.append([stamp.isoformat(), *row[1:6]])
            except (ValueError, TypeError, IndexError):
                raise ValueError('Groww returned an invalid candle timestamp.') from None
        return normalize_candles(normalized)

    def candles(self, exchange, pair, intraday_only=False):
        keys = self.keys(exchange)
        end = now().astimezone(IST)
        start = end.replace(hour=0, minute=0, second=0, microsecond=0)
        if not intraday_only:
            start -= timedelta(days=27)
        return {leg: self._candles(exchange, keys[pair[leg]]['groww_symbol'], start, end, '5minute')
                for leg in ('y', 'x')}

    def history(self, exchange, pair):
        keys = self.keys(exchange)
        end = now().astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(seconds=1)
        start = end-timedelta(days=1100)
        series = []
        for leg in ('y', 'x'):
            cursor, candles = start, {}
            while cursor < end:
                boundary = min(end, cursor+timedelta(days=179))
                rows = self._candles(exchange, keys[pair[leg]]['groww_symbol'], cursor, boundary, '1day')
                candles.update({datetime.fromtimestamp(r['time'], IST).date().isoformat(): r['close'] for r in rows})
                cursor = boundary
            series.append(pd.Series(candles, name=leg, dtype=float))
        frame = pd.concat(series, axis=1).sort_index()
        if frame.empty or frame.isna().any().any() or (end.date()-pd.Timestamp(frame.index[-1]).date()).days > 7:
            raise ValueError('Groww daily candles are missing, unmatched or stale. No prices were filled in.')
        return frame
