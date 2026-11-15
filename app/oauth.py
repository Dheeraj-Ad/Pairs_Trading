"""Upstox authorization-code login. Secrets and access tokens stay server-side."""
from dataclasses import dataclass
from datetime import timedelta
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlencode, urlparse
import urllib.error
import urllib.request

from app.market import IST, now

CALLBACK = '/auth/upstox/callback'
COOKIE = 'pairwise_upstox_login'
KEYS = ('UPSTOX_CLIENT_ID', 'UPSTOX_CLIENT_SECRET', 'UPSTOX_REDIRECT_URI')


class OAuthError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(repr=False)
class OAuthConfig:
    client_id: str = ''
    client_secret: str = ''
    redirect_uri: str = 'http://127.0.0.1:8765/auth/upstox/callback'

    @classmethod
    def load(cls, root, port=8765):
        values = {}
        path = Path(root) / '.env'
        if path.is_file():
            # Deliberately no shell expansion, interpolation, or execution.
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                if key.strip() in KEYS:
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    values[key.strip()] = value
        values.update({k: os.environ[k] for k in KEYS if k in os.environ})
        return cls(values.get(KEYS[0], ''), values.get(KEYS[1], ''),
                   values.get(KEYS[2]) or f'http://127.0.0.1:{port}{CALLBACK}')

    @property
    def configured(self):
        return bool(self.client_id.strip() and self.client_secret.strip())

    @property
    def origin(self):
        parsed = urlparse(self.redirect_uri)
        return f'{parsed.scheme}://{parsed.netloc}'

    def validate(self, request_origin):
        if not self.configured:
            raise ValueError('Upstox app setup is required. Add your API key and secret to .env using .env.example, then restart.')
        parsed = urlparse(self.redirect_uri)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or parsed.path != CALLBACK or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError('This local app requires an HTTP localhost or 127.0.0.1 callback ending in /auth/upstox/callback.')
        if self.origin != request_origin:
            raise ValueError('Open the app at the origin configured in UPSTOX_REDIRECT_URI; its host and port must match this server.')


def login_cookie(binding='', clear=False):
    cookie = SimpleCookie()
    cookie[COOKIE] = binding
    cookie[COOKIE]['path'] = CALLBACK
    cookie[COOKIE]['httponly'] = True
    cookie[COOKIE]['samesite'] = 'Lax'
    cookie[COOKIE]['max-age'] = 0 if clear else 600
    return cookie.output(header='').strip()


def browser_binding(header):
    cookie = SimpleCookie()
    try:
        cookie.load(header or '')
        return cookie[COOKIE].value if COOKIE in cookie else ''
    except Exception:
        return ''


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UpstoxOAuth:
    def __init__(self, config):
        self.config = config
        self.pending = None
        self.revision = 0
        self.lock = threading.Lock()

    def cancel(self):
        with self.lock:
            self.pending = None
            self.revision += 1

    def is_current(self, pending):
        with self.lock:
            return pending['revision'] == self.revision

    def begin(self, origin, exchange, generation):
        self.config.validate(origin)
        if exchange not in ('NSE', 'BSE'):
            raise ValueError('Choose NSE or BSE.')
        state, binding = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with self.lock:
            self.revision += 1
            # One local user: a newer login replaces an older pending attempt.
            self.pending = dict(state=state, binding=binding, expires=time.monotonic()+600,
                                exchange=exchange, generation=generation, revision=self.revision)
        query = urlencode(dict(response_type='code', client_id=self.config.client_id,
                               redirect_uri=self.config.redirect_uri, state=state))
        return 'https://api.upstox.com/v2/login/authorization/dialog?' + query, login_cookie(binding)

    def consume(self, params, cookie_header):
        state = params.get('state', [])
        binding = browser_binding(cookie_header)
        with self.lock:
            pending = self.pending
            if (not pending or len(state) != 1 or not binding
                    or not secrets.compare_digest(state[0], pending['state'])
                    or not secrets.compare_digest(binding, pending['binding'])):
                raise OAuthError('invalid_state', 'This login does not belong to this browser. Start Connect with Upstox again.')
            self.pending = None  # Consume before any network call, including failed exchanges.
            if time.monotonic() > pending['expires']:
                raise OAuthError('expired', 'Login timed out. Start Connect with Upstox again.')
        if 'error' in params:
            raise OAuthError('cancelled', 'Upstox login was cancelled or denied.')
        codes = params.get('code', [])
        if len(codes) != 1 or not codes[0] or len(codes[0]) > 4096:
            raise OAuthError('failed', 'Upstox did not return a valid authorization code.')
        return codes[0], pending

    def exchange(self, code):
        payload = urlencode(dict(code=code, client_id=self.config.client_id,
            client_secret=self.config.client_secret, redirect_uri=self.config.redirect_uri,
            grant_type='authorization_code')).encode()
        request = urllib.request.Request('https://api.upstox.com/v2/login/authorization/token',
            data=payload, headers={'Accept':'application/json', 'Content-Type':'application/x-www-form-urlencoded'}, method='POST')
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
                result = json.loads(response.read(1_000_000))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            # Do not forward provider bodies, which may contain credentials or personal data.
            raise OAuthError('failed', 'Could not complete Upstox authorization. Check the app configuration and reconnect.') from None
        token = result.get('access_token') if isinstance(result, dict) else None
        if not isinstance(token, str) or not token or len(token)>8192 or any(c.isspace() for c in token):
            raise OAuthError('failed', 'Upstox did not issue a valid access token.')
        local = now().astimezone(IST)
        expires = local.replace(hour=3, minute=30, second=0, microsecond=0)
        if expires <= local:
            expires += timedelta(days=1)
        return dict(token=token, user_name=str(result.get('user_name') or 'Upstox user')[:100],
                    expires_at=expires.isoformat())
