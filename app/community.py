"""Optional 0xramm API snapshots. These are never executable broker quotes."""
from copy import deepcopy
import json
import math
import threading
import time
from urllib.parse import urlsplit, urlencode
import urllib.request
import urllib.error

from app.market import PAIRS, USER_AGENT, now

DEFAULT_URL = 'http://65.0.104.9'
NOTICE = ('Third-party Yahoo-based snapshots. Exchange trade time and candle history are unavailable. '
          'Observations are for comparison only; they never price paper fills or train the trading model.')


def validate_url(value):
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError('Enter a valid community API base URL.')
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError('Use an HTTP(S) base URL without credentials, query parameters or fragments.')
    if parsed.scheme == 'http' and parsed.hostname not in ('65.0.104.9', '127.0.0.1', 'localhost'):
        raise ValueError('Use HTTPS for other deployments, or HTTP localhost for a self-hosted API.')
    return value.strip().rstrip('/')


def number(value, positive=False):
    # Numeric v3 contract. Never infer units from formatted display strings.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and (not positive or value > 0) else None


def parse_snapshots(body, exchange):
    if not isinstance(body, dict) or body.get('status') != 'success' or not isinstance(body.get('stocks'), list):
        raise ValueError('Community API did not return its documented batch-stock response.')
    if len(body['stocks']) > 100:
        raise ValueError('Community API returned an oversized stock list.')
    symbols = {p[leg] for p in PAIRS for leg in ('y', 'x')}
    suffix = '.NS' if exchange == 'NSE' else '.BO'
    rows, rejected = {}, set()
    for row in body['stocks']:
        if not isinstance(row, dict): continue
        symbol = row.get('symbol')
        if not isinstance(symbol, str) or symbol not in symbols: continue
        if symbol in rows or symbol in rejected:
            rows.pop(symbol, None); rejected.add(symbol); continue
        price = number(row.get('last_price'), positive=True)
        if row.get('error') or row.get('exchange') != exchange or row.get('ticker') != symbol+suffix or price is None:
            rejected.add(symbol); continue
        rows[symbol] = dict(symbol=symbol, price=price, change=number(row.get('change')),
                            percent_change=number(row.get('percent_change')), volume=number(row.get('volume')),
                            exchange_trade_time=None)
    if not rows:
        raise ValueError('Community API returned no valid prices for the requested exchange.')
    return rows, sorted(symbols-set(rows))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_snapshots(base_url, exchange):
    suffix = '.NS' if exchange == 'NSE' else '.BO'
    symbols = [p[leg]+suffix for p in PAIRS for leg in ('y', 'x')]
    url = validate_url(base_url)+'/stock/list?'+urlencode({'symbols':','.join(symbols), 'res':'num'})
    # Deliberately no Upstox token, API key, cookies or secret headers.
    request = urllib.request.Request(url, headers={'User-Agent':USER_AGENT, 'Accept':'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000: raise ValueError('Community API response exceeded the size limit.')
        body = json.loads(raw)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'Community API returned HTTP {exc.code}. No new snapshot was recorded.') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError('Community API is unreachable or timed out. Local demo and Upstox are unaffected.') from None
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError('Community API did not return valid JSON.') from None
    return parse_snapshots(body, exchange)


class CommunityFeed:
    def __init__(self, app):
        self.app = app
        self.base_url = DEFAULT_URL
        self.enabled = False  # Restarts never start third-party polling unattended.
        self.running = False
        self.next_poll = 0.
        self.results = {}
        self.config_path = app.directory/'community-feed.json'
        if self.config_path.exists():
            try: self.base_url = validate_url(json.loads(self.config_path.read_text())['base_url'])
            except (OSError, ValueError, KeyError, TypeError): pass
        with app.book.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS community_snapshots(endpoint TEXT, exchange TEXT, body TEXT, PRIMARY KEY(endpoint,exchange))')
            for endpoint, exchange, body in db.execute('SELECT endpoint,exchange,body FROM community_snapshots'):
                try: self.results[(endpoint,exchange)] = json.loads(body)
                except ValueError: pass

    def configure(self, data):
        if self.running: raise ValueError('Wait for the current community request to finish.')
        url = validate_url(data.get('base_url', self.base_url))
        enabled = data.get('enabled', self.enabled)
        if not isinstance(enabled, bool): raise ValueError('Choose whether to poll the community feed.')
        self.base_url, self.enabled = url, enabled
        self.config_path.write_text(json.dumps({'base_url':url}), encoding='utf-8')

    def start(self):
        if self.running: return False
        if time.monotonic() < self.next_poll:
            raise ValueError('Community requests are limited to once per 60 seconds. Wait before retrying.')
        endpoint, exchange = self.base_url, self.app.exchange
        self.running = True
        self.next_poll = time.monotonic()+60
        def worker():
            try:
                rows, missing = fetch_snapshots(endpoint, exchange)
                observed = now().isoformat()
                with self.app.lock:
                    old = self.results.get((endpoint,exchange), {})
                    history = deepcopy(old.get('history', {}))
                    for symbol, row in rows.items():
                        history[symbol] = (history.get(symbol, [])+[dict(observed_at=observed, price=row['price'])])[-120:]
                    result = dict(rows=rows, missing=missing, observed_at=observed, error=None, history=history)
                    self.results[(endpoint,exchange)] = result
                    with self.app.book.connect() as db:
                        db.execute('INSERT OR REPLACE INTO community_snapshots VALUES(?,?,?)',
                                   (endpoint,exchange,json.dumps(result,allow_nan=False)))
            except Exception as exc:
                with self.app.lock:
                    old = deepcopy(self.results.get((endpoint,exchange), dict(rows={}, history={}, missing=[])))
                    old['error'] = str(exc) if isinstance(exc,ValueError) else 'Community snapshot request failed.'
                    self.results[(endpoint,exchange)] = old
            finally:
                with self.app.lock: self.running = False
        threading.Thread(target=worker, daemon=True).start()
        return True

    def tick(self):
        with self.app.lock:
            if self.enabled and not self.running and time.monotonic() >= self.next_poll:
                self.start()

    def view(self):
        return dict(base_url=self.base_url, enabled=self.enabled, running=self.running, exchange=self.app.exchange,
                    notice=NOTICE, interval_seconds=60, retry_seconds=max(0,int(self.next_poll-time.monotonic())+1),
                    result=self.results.get((self.base_url,self.app.exchange)))
