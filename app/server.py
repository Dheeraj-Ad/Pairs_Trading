"""Run with python -m app.server. Local browser UI + read-only market data."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlparse, parse_qs

import joblib
import pandas as pd

from app.market import PAIRS, Upstox, demo_history, fresh, now
from app.paper import PaperBook
from app.local_market import LocalMarket
from app.reporting import summary, transactions
from ml.paper_model import PaperLearner
from app.report_lab import ReportLab
from app.community import CommunityFeed
from app.demo_session import DemoSession
from app.groww import Groww, GrowwConfig
from ml.research import baseline, train
from app.terminal import Terminal
from app.oauth import OAuthConfig, UpstoxOAuth, OAuthError, CALLBACK, login_cookie

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"


class LocalHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let a second process share a listening port,
    # splitting OAuth state and callbacks between two application instances.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Application:
    def __init__(self, directory=None, initial_mode='demo', oauth_config=None):
        self.directory = Path(directory or ROOT / "runtime")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.book = PaperBook(self.directory / "paper.sqlite3")
        self.local = LocalMarket(self.book)
        self.provider = Upstox(os.environ.get("UPSTOX_ACCESS_TOKEN", ""))
        self.upstox = self.provider
        self.groww_status = dict(configured=GrowwConfig.load(ROOT).configured, error=None, connecting=False)
        self.lock = threading.RLock()
        self.csrf = secrets.token_urlsafe(32)
        self.oauth = UpstoxOAuth(oauth_config or OAuthConfig.load(ROOT))
        self.oauth_user = None
        self.mode, self.exchange = initial_mode, "NSE"
        self.quotes, self.histories, self.baselines, self.reports = {}, {}, {}, {}
        self.error, self.last_poll = None, None
        for i, pair in enumerate(PAIRS):
            frame = demo_history(i)
            self.histories["demo:"+pair["id"]] = frame
            self.baselines["demo:"+pair["id"]] = baseline(frame)
        # Only JSON metadata is loaded. Serialized models are never unpickled here.
        for file in self.directory.glob("report-*.json"):
            try:
                report = json.loads(file.read_text(encoding="utf-8"))
                self.reports[report["key"]] = report
            except (OSError, ValueError, KeyError):
                continue
        self.poll()
        self.terminal = Terminal(self)
        self.paper_learner = PaperLearner(self)
        self.report_lab = ReportLab(self)
        self.community = CommunityFeed(self)
        self.demo_session = DemoSession(self)

    def market_now(self):
        return self.local.clock(self.account_key) if self.mode == 'demo' else now()

    def advance_local(self, steps=1):
        if self.mode != 'demo':
            raise ValueError('Local simulation controls require the local data source.')
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 75:
            raise ValueError('Advance between 1 and 75 candles.')
        if self.terminal.job['running']:
            raise ValueError('Wait for model training to finish before advancing local data.')
        for _ in range(steps):
            self.local.advance(self.account_key)
            self.poll()
            for pair_id in self.terminal.candles:
                self.terminal.candles[pair_id] = self.local.candles(self.account_key, pair_id)
            self.terminal.candle_time = self.market_now().isoformat()
            self.terminal.update_signals()
            self.terminal.manage()

    def start_local_batch(self, steps):
        if self.local.batch_active:
            raise ValueError('A local batch is already advancing. Pause it or wait for completion.')
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 75:
            raise ValueError('Advance between 1 and 75 candles.')
        self.local.running = False
        self.local.remaining, self.local.batch_active = steps, True
        def work():
            try:
                while True:
                    with self.lock:
                        if not self.local.remaining: break
                        self.advance_local()
                        self.local.remaining -= 1
                    time.sleep(.01)  # Let state requests and pause controls acquire the lock.
            except Exception as exc:
                with self.lock:
                    self.terminal.enabled = False
                    self.terminal.feed_error = str(exc) if isinstance(exc, ValueError) else 'Local advancement failed. Entries paused; review positions before restarting.'
            finally:
                with self.lock:
                    self.local.remaining, self.local.batch_active = 0, False
        threading.Thread(target=work, daemon=True).start()

    @property
    def account_key(self):
        return self.mode + ":" + self.exchange

    def research_key(self, pair):
        return ("demo" if self.mode == "demo" else self.account_key) + ":" + pair["id"]

    @property
    def provider_name(self):
        return 'Groww' if self.mode == 'groww' else 'Upstox'

    def connect_groww(self, data):
        """Validate credentials and data before committing a source switch."""
        with self.lock:
            exchange = data.get('exchange')
            if exchange not in ('NSE', 'BSE'):
                raise ValueError('Choose NSE or BSE.')
            self.terminal.can_switch('groww:'+exchange)
            if self.groww_status['connecting']:
                raise ValueError('A Groww connection check is already running.')
            token = data.get('token', '')
            if not isinstance(token, str) or len(token) > 8192 or any(c.isspace() for c in token):
                raise ValueError('Invalid Groww access-token format.')
            config = GrowwConfig.load(ROOT)
            candidate = Groww(config, token)
            generation = self.terminal.generation
            self.groww_status.update(configured=config.configured, error=None, connecting=True)
        try:
            candidate.authenticate()
            # Probe access before downloading the full master or 20 quotes.
            candidate.quote(exchange, 'HDFCBANK')
            quotes = candidate.quotes(exchange)
            with self.lock:
                if generation != self.terminal.generation:
                    raise ValueError('The account changed while Groww connected. Retry the connection.')
                self.terminal.can_switch('groww:'+exchange)
                self.provider = candidate
                self.mode, self.exchange, self.quotes = 'groww', exchange, quotes
                self.oauth.cancel()
                self.terminal.reset()
                self.error, self.last_poll = None, now().isoformat()
                self.groww_status['error'] = None
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else 'Groww connection failed. Retry after checking API access.'
            with self.lock:
                self.groww_status['error'] = message
            raise ValueError(message) from None
        finally:
            with self.lock:
                self.groww_status['connecting'] = False
        return self.snapshot()

    def oauth_status(self):
        expired = bool(self.oauth_user and now() >= datetime.fromisoformat(self.oauth_user['expires_at']))
        return dict(configured=self.oauth.config.configured, redirect_uri=self.oauth.config.redirect_uri,
                    app_url=self.oauth.config.origin, user=self.oauth_user, expired=expired,
                    last_error=self.oauth.error_status())

    def start_oauth(self, data, origin):
        with self.lock:
            exchange = data.get('exchange', self.exchange)
            if exchange not in ('NSE','BSE'):
                raise ValueError('Choose NSE or BSE.')
            self.terminal.can_switch('live:'+exchange)
            url,cookie = self.oauth.begin(origin,exchange,self.terminal.generation)
            if self.terminal.enabled:
                self.terminal.enabled=False
                self.terminal.event('control','','Entries paused while reconnecting to Upstox. Exits remain active.')
            return {'authorization_url':url},cookie

    def finish_oauth(self, params, cookie, origin):
        self.oauth.config.validate(origin)
        code,pending = self.oauth.consume(params,cookie)
        def check_context():
            if self.terminal.generation != pending['generation'] or not self.oauth.is_current(pending):
                raise OAuthError('changed','The connection changed during sign-in. Connect again.')
            try:
                self.terminal.can_switch('live:'+pending['exchange'])
            except ValueError:
                raise OAuthError('changed','The application changed during sign-in. Connect again.') from None
        with self.lock:
            check_context()
        authorized = self.oauth.exchange(code)
        with self.lock:
            check_context()
            self.provider = self.upstox
            self.provider.token = authorized['token']
            self.oauth_user = {k:authorized[k] for k in ('user_name','expires_at')}
            self.mode,self.exchange,self.quotes='live',pending['exchange'],{}
            self.last_poll=None
            self.error='Upstox connected. Waiting for the first market-data refresh.'
            self.terminal.reset()
            self.oauth.cancel()
        # Keep a successfully authorized token even if quote access is temporarily unavailable.
        self.poll()

    def poll(self):
        # Fetch outside the state lock, so a slow provider cannot freeze the UI.
        with self.lock:
            if self.mode == 'demo':
                # Local prices and their clock commit together; a concurrent poll
                # must never restore quotes from the previous simulation step.
                self.quotes = self.local.quotes(self.account_key)
                self.error, self.last_poll = None, self.market_now().isoformat()
                return
            if self.mode=='live' and self.oauth_status()['expired']:
                self.error='Your Upstox session expired. Use Connect with Upstox to reconnect.'
                if hasattr(self,'terminal'): self.terminal.enabled=False
                return
            mode,exchange,token=self.mode,self.exchange,self.provider.token
            provider=self.provider
            generation=self.terminal.generation if hasattr(self,'terminal') else 0
        def same_source():
            return (mode,exchange,token)==(self.mode,self.exchange,self.provider.token) and generation==(self.terminal.generation if hasattr(self,'terminal') else 0)
        try:
            quotes = provider.quotes(exchange)
            with self.lock:
                if not same_source(): return
                self.quotes=quotes
                self.error, self.last_poll = None, now().isoformat()
        except Exception as exc:
            with self.lock:
                if not same_source(): return
                self.error = str(exc) if isinstance(exc, ValueError) else "Market-data refresh failed. Try reconnecting."

    def snapshot(self):
        with self.lock:
            clock = self.market_now()
            account = self.book.mark(self.book.read(self.account_key), self.quotes, clock)
            pairs = []
            for pair in PAIRS:
                item = {k: v for k, v in pair.items() if not k.startswith("isin")}
                key = self.research_key(pair)
                info = self.baselines.get(key)
                qy, qx = self.quotes.get(pair["y"]), self.quotes.get(pair["x"])
                item.update(history=info, report=self.reports.get(key), quotes=[qy, qx], zscore=None, signal="Load history")
                if info and qy and qx:
                    z = (qy["price"]-info["alpha"]-info["beta"]*qx["price"]-info["mean"])/info["std"]
                    item["zscore"] = z
                    item["signal"] = "Stop zone" if abs(z) >= 4 else "Long spread" if z <= -2 else "Short spread" if z >= 2 else "Near mean" if abs(z) <= .5 else "Wait"
                item["fresh"] = bool(qy and qx and fresh(qy, clock) and fresh(qx, clock) and not self.error)
                pairs.append(item)
            return {"mode": self.mode, "exchange": self.exchange, "account_key": self.account_key,
                    "connected": bool(self.provider.token) and not getattr(self.provider, 'blocked', False) and (self.mode != 'live' or not self.oauth_status()['expired']), "error": self.error, "last_poll": self.last_poll,
                    "provider_name": self.provider_name, "groww": dict(self.groww_status),
                    "oauth": self.oauth_status(),
                    "csrf": self.csrf, "pairs": pairs, "account": account,
                    "market_time": clock.isoformat(), "performance": summary(account),
                    "transactions": transactions(account), "paper_model": self.paper_learner.view(),
                    "daily_report": self.report_lab.view(),
                    "community": self.community.view(), "demo_session": dict(self.demo_session.job),
                    "local": self.local.view(self.account_key) if self.mode == 'demo' else None,
                    "terminal": self.terminal.view(),
                    "source": "Synthetic demo · not market prices" if self.mode == "demo" else self.provider_name + " · quotes refreshed after each 5-second polling wait",
                    "costs": {"fee_bps": 3, "slippage_bps": 5, "annual_borrow_percent": 10}}

    def action(self, route, data):
        if route == '/api/connection' and data.get('mode') == 'groww':
            return self.connect_groww(data)
        with self.lock:
            if self.demo_session.job['running'] and route not in ('/api/demo/stop','/api/community'):
                raise ValueError('Stop the automated demo or wait for it to finish before changing trading controls.')
            if route == "/api/connection":
                mode, exchange = data.get("mode"), data.get("exchange")
                if mode not in ("demo", "live") or exchange not in ("NSE", "BSE"):
                    raise ValueError("Choose demo/live mode and NSE/BSE exchange.")
                self.terminal.can_switch(mode+':'+exchange)
                token = data.get("token", "").strip()
                candidate = self.upstox
                old_token = candidate.token
                if token:
                    if len(token) > 8192 or any(c.isspace() for c in token):
                        raise ValueError("Invalid access-token format.")
                    candidate.token = token
                if mode == "live" and not candidate.token:
                    raise ValueError("An Upstox access token is required for live data.")
                # Validate before switching accounts; failure preserves the active account.
                try:
                    quotes = candidate.quotes(exchange) if mode == "live" else {}
                except Exception:
                    candidate.token = old_token
                    raise
                self.provider = candidate
                self.mode, self.exchange, self.quotes = mode, exchange, quotes
                self.oauth.cancel()
                if token: self.oauth_user=None
                self.terminal.reset()
                self.error = None
                if mode == "demo":
                    self.poll()
                else:
                    self.last_poll = now().isoformat()
            elif route == "/api/disconnect":
                self.terminal.can_switch()
                self.provider.token = ""
                self.oauth.cancel()
                self.oauth_user=None
                self.mode, self.quotes = ('groww' if self.mode == 'groww' else 'live'), {}
                self.terminal.reset()
                self.poll()
            else:
                if data.get("account_key") != self.account_key:
                    raise ValueError("The active account changed. Refresh and try again.")
                pair = next((p for p in PAIRS if p["id"] == data.get("pair")), None)
                if route == '/api/scan':
                    self.terminal.scan()
                elif route == '/api/report/run':
                    self.report_lab.start()
                elif route == '/api/community':
                    self.community.configure(data)
                    if data.get('refresh', False): self.community.start()
                elif route == '/api/demo/run':
                    self.demo_session.start()
                elif route == '/api/demo/stop':
                    self.demo_session.cancelled = True
                    self.demo_session.job['message'] = 'Stopping new demo entries; normal risk exits will finish remaining positions.'
                elif route == '/api/local':
                    if self.mode != 'demo': raise ValueError('Select local simulation first.')
                    if self.terminal.job['running']: raise ValueError('Wait for model training to finish.')
                    if 'running' in data:
                        if not isinstance(data['running'], bool): raise ValueError('Invalid playback setting.')
                        if data['running'] and self.local.batch_active: raise ValueError('Wait for the current batch or pause it first.')
                        if not data['running']: self.local.remaining = 0
                        self.local.running = data['running']
                        self.local.last_step = time.monotonic()
                    if 'steps' in data:
                        self.start_local_batch(data['steps'])
                elif route == '/api/paper/train':
                    self.paper_learner.train()
                    self.terminal.update_signals()
                elif route == '/api/paper/filter':
                    self.paper_learner.configure(data.get('enabled'))
                elif route == '/api/automation':
                    self.terminal.configure(data)
                elif route in ("/api/history", "/api/train"):
                    if not pair:
                        raise ValueError("Unknown pair.")
                    key = self.research_key(pair)
                    if route == "/api/history":
                        frame = self.histories[key] if self.mode == "demo" else self.provider.history(self.exchange, pair)
                        info = baseline(frame)
                        self.histories[key], self.baselines[key] = frame, info
                        frame.to_csv(self.directory / ("history-"+key.replace(":", "-")+".csv"))
                    else:
                        if key not in self.histories:
                            raise ValueError("Load daily history for this pair before training.")
                        model, report = train(self.histories[key])
                        report.update(key=key, source=self.mode, exchange=self.exchange, trained_at=now().isoformat())
                        stem = key.replace(":", "-")
                        joblib.dump(model, self.directory / ("model-"+stem+".joblib"))
                        target = self.directory / ("report-"+stem+".json")
                        temp = target.with_suffix(".tmp")
                        temp.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
                        temp.replace(target)
                        self.reports[key] = report
                elif route in ("/api/open", "/api/close"):
                    if self.error:
                        raise ValueError("Market-data connection has an error. Fills paused until it recovers.")
                    if route == "/api/close":
                        self.book.trade(self.account_key, self.quotes, data.get("request_id"), close_id=data.get("position_id"), as_of=self.market_now())
                    else:
                        if not pair:
                            raise ValueError("Unknown pair.")
                        report = self.terminal.reports.get(pair['id'])
                        info = report['baseline'] if report else self.baselines.get(self.research_key(pair))
                        if not info:
                            raise ValueError("Load pair history in ML Lab before opening a position.")
                        age = (now().date() - pd.Timestamp(info["history_end"]).date()).days
                        if age > 7 and self.mode != 'demo' and not report:
                            raise ValueError("The signal history is stale. Reload it in ML Lab.")
                        self.book.trade(self.account_key, self.quotes, data.get("request_id"), pair=pair,
                                        side=data.get("side"), budget=data.get("budget"), beta=info["beta"],
                                        decision=self.terminal.entry_decision(pair['id']), as_of=self.market_now())
                else:
                    raise ValueError("Unknown action.")
            return self.snapshot()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log access tokens or request bodies.

        def send(self, status, body, content_type="application/json", headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            for key,value in (headers or {}).items():
                self.send_header(key,value)
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Local access only."})
            route = urlparse(self.path).path
            if route == CALLBACK:
                result='connected'
                try:
                    if len(self.path)>8192: raise ValueError('Invalid callback.')
                    params=parse_qs(urlparse(self.path).query,keep_blank_values=True,max_num_fields=16)
                    app.finish_oauth(params,self.headers.get('Cookie'),'http://'+self.headers['Host'])
                except OAuthError as exc:
                    result=exc.code
                    app.oauth.record_error(result)
                except Exception:
                    result='failed'
                    app.oauth.record_error(result)
                return self.send(303,b'',headers={'Location':'/?upstox='+result,'Set-Cookie':login_cookie(clear=True)})
            if route == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            if route == "/api/state":
                return self.send(200, app.snapshot())
            if route == '/api/report/artifact':
                try:
                    params=parse_qs(urlparse(self.path).query,max_num_fields=4)
                    with app.lock:path=app.report_lab.artifact(params.get('run',[''])[0],params.get('name',[''])[0])
                    return self.send(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                except (ValueError,OSError):return self.send(404,{'error':'Report artifact not found.'})
            if route == "/api/export":
                with app.lock:
                    rows = app.book.read(app.account_key)["trades"]
                buffer = io.StringIO()
                fields = ["id", "pair", "side", "opened", "closed", "pnl", "fees", "borrow", "entry_legs", "exit_legs", "origin", "decision", "exit_reason"]
                writer = csv.DictWriter(buffer, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                return self.send(200, buffer.getvalue(), "text/csv; charset=utf-8")
            if route == '/api/transactions/export':
                with app.lock:
                    rows = transactions(app.book.read(app.account_key))
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=['trade_id','time','event','pair','origin','symbol','side','quantity','price','value','fee'])
                writer.writeheader()
                writer.writerows(rows)
                return self.send(200, buffer.getvalue(), 'text/csv; charset=utf-8')
            name = {"/": "terminal.html", "/app.js": "terminal.js", "/local.js": "local.js", "/report.js": "report.js", "/community.js": "community.js", "/style.css": "terminal.css"}.get(route)
            if name:
                return self.send(200, (STATIC/name).read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain")+"; charset=utf-8")
            self.send(404, {"error": "Not found."})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16384:
                    return self.send(400, {"error": "Invalid request size."})
                raw = self.rfile.read(size)
            except ValueError:
                return self.send(400, {"error": "Invalid request size."})
            origin = self.headers.get("Origin")
            allowed = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
            if not self.valid_host() or (origin and origin not in allowed) or self.headers.get("X-CSRF-Token") != app.csrf:
                return self.send(403, {"error": "Invalid local session. Reload the application."})
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if urlparse(self.path).path == '/api/auth/upstox/start':
                    result,cookie=app.start_oauth(data,'http://'+self.headers['Host'])
                    return self.send(200,result,headers={'Set-Cookie':cookie})
                result = app.action(urlparse(self.path).path, data)
                self.send(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                self.send(500, {"error": "Operation failed. No successful fill was reported. Refresh before retrying."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--live', action='store_true', help='Start disconnected from Upstox instead of local simulation')
    args = parser.parse_args()
    app = Application(initial_mode='live' if args.live else 'demo',oauth_config=OAuthConfig.load(ROOT,args.port))
    if app.mode == 'demo' and not any(p.get('origin')=='auto' for p in app.book.read(app.account_key)['positions']):
        app.terminal.scan()
    stop = threading.Event()
    def refresh():
        while not stop.wait(5):
            app.poll()
    def community_poll():
        while not stop.wait(5):
            app.community.tick()
    def automate():
        while not stop.wait(5):
            try:
                app.terminal.tick()
            except Exception:
                with app.lock:
                    app.terminal.enabled=False
                    app.terminal.feed_error='Automation processing failed; entries paused. Review positions and rescan.'
    def execute():
        # Risk checks run separately: slow candle downloads must not block exits.
        while not stop.wait(1):
            try:
                with app.lock:
                    if app.mode == 'demo' and app.local.running and not app.terminal.job['running'] and time.monotonic()-app.local.last_step >= 2:
                        app.advance_local()
                        app.local.last_step = time.monotonic()
                    app.terminal.manage()
            except Exception as exc:
                with app.lock:
                    app.terminal.enabled=False
                    app.terminal.feed_error=str(exc) if isinstance(exc, ValueError) else 'Paper execution failed; entries paused. Review positions before restarting.'
    try:
        server = LocalHTTPServer(("127.0.0.1", args.port), handler_for(app))
    except OSError:
        parser.exit(2, f'Cannot listen on 127.0.0.1:{args.port}. Another app instance may already be running; use that window or stop it before restarting.\n')
    threading.Thread(target=refresh, daemon=True).start()
    threading.Thread(target=community_poll, daemon=True).start()
    threading.Thread(target=automate, daemon=True).start()
    threading.Thread(target=execute, daemon=True).start()
    source = 'local simulation' if app.mode == 'demo' else 'Upstox data'
    print(f"Pairwise terminal: http://127.0.0.1:{args.port} - {source}, paper execution only.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
