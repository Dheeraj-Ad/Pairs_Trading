"""OAuth tests never contact Upstox or use real credentials."""
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
import http.cookiejar
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
from urllib.parse import parse_qs, urlparse, urlencode
import urllib.error
import urllib.request

from app.oauth import OAuthConfig, UpstoxOAuth, OAuthError, CALLBACK, browser_binding
from app.market import IST, now
from app.server import Application, handler_for, LocalHTTPServer


class OAuthFlowTests(unittest.TestCase):
    def setUp(self):
        self.config=OAuthConfig('test-key','test-secret')
        self.flow=UpstoxOAuth(self.config)

    def begin(self):
        url,cookie=self.flow.begin('http://127.0.0.1:8765','NSE',7)
        params=parse_qs(urlparse(url).query)
        return params,cookie

    def test_authorization_url_cookie_and_one_use(self):
        params,cookie=self.begin()
        self.assertEqual(params['redirect_uri'],[self.config.redirect_uri])
        self.assertNotIn('test-secret',str(params))
        self.assertIn('HttpOnly',cookie);self.assertIn('SameSite=Lax',cookie)
        self.assertIn('Path='+CALLBACK,cookie)
        code,pending=self.flow.consume({'state':params['state'],'code':['one-use-code']},cookie)
        self.assertEqual(code,'one-use-code');self.assertEqual(pending['generation'],7)
        with self.assertRaises(OAuthError):
            self.flow.consume({'state':params['state'],'code':['one-use-code']},cookie)

    def test_bad_state_and_wrong_browser_do_not_consume_valid_login(self):
        params,cookie=self.begin()
        for query,header in [({'state':['bad'],'code':['x']},cookie),
                             ({'state':params['state'],'code':['x']},''),
                             ({'state':params['state']*2,'code':['x']},cookie)]:
            with self.assertRaises(OAuthError):self.flow.consume(query,header)
        self.assertEqual(self.flow.consume({'state':params['state'],'code':['ok']},cookie)[0],'ok')

    def test_expiry_denial_and_duplicate_code_are_rejected(self):
        params,cookie=self.begin();self.flow.pending['expires']=0
        with self.assertRaises(OAuthError) as e:self.flow.consume({'state':params['state'],'code':['x']},cookie)
        self.assertEqual(e.exception.code,'expired')
        params,cookie=self.begin()
        with self.assertRaises(OAuthError) as e:self.flow.consume({'state':params['state'],'error':['access_denied']},cookie)
        self.assertEqual(e.exception.code,'cancelled')
        params,cookie=self.begin()
        with self.assertRaises(OAuthError):self.flow.consume({'state':params['state'],'code':['a','b']},cookie)

    def test_new_attempt_and_disconnect_invalidate_pending_context(self):
        params,cookie=self.begin()
        _,pending=self.flow.consume({'state':params['state'],'code':['a']},cookie)
        self.begin();self.assertFalse(self.flow.is_current(pending))
        params,cookie=self.begin();self.flow.cancel()
        with self.assertRaises(OAuthError):self.flow.consume({'state':params['state'],'code':['a']},cookie)

    def test_missing_config_wrong_host_and_external_callback_fail(self):
        with self.assertRaises(ValueError):OAuthConfig().validate('http://127.0.0.1:8765')
        with self.assertRaises(ValueError):self.config.validate('http://localhost:8765')
        for callback in ['https://evil.example/auth/upstox/callback','http://127.0.0.1:8765/auth/upstox/callback?x=y',
                         'http://user:secret@127.0.0.1:8765/auth/upstox/callback']:
            with self.assertRaises(ValueError):OAuthConfig('key','secret',callback).validate('http://127.0.0.1:8765')

    def test_token_exchange_is_server_side_and_expiry_is_0330_ist(self):
        for hour,expected_day in [(2,15),(10,16)]:
            current=datetime(2026,11,15,hour,0,tzinfo=IST)
            opener=Mock();opener.open.return_value=io.BytesIO(json.dumps({'access_token':'test-access-token','user_name':'Test User'}).encode())
            with patch('app.oauth.urllib.request.build_opener',return_value=opener),patch('app.oauth.now',return_value=current):
                result=self.flow.exchange('code')
            req=opener.open.call_args.args[0]
            self.assertEqual(req.full_url,'https://api.upstox.com/v2/login/authorization/token')
            self.assertEqual(req.method,'POST')
            self.assertEqual(req.get_header('User-agent'),'PairwisePaperTrading/1.0')
            self.assertEqual(parse_qs(req.data.decode())['client_secret'],['test-secret'])
            self.assertNotIn('test-secret',req.full_url)
            expiry=datetime.fromisoformat(result['expires_at'])
            self.assertEqual((expiry.day,expiry.hour,expiry.minute),(expected_day,3,30))

    def test_provider_errors_and_invalid_tokens_never_echo_secret(self):
        opener=Mock();opener.open.side_effect=urllib.error.URLError('test-secret should not leak')
        with patch('app.oauth.urllib.request.build_opener',return_value=opener),self.assertRaises(OAuthError) as e:
            self.flow.exchange('secret-code')
        self.assertNotIn('test-secret',str(e.exception));self.assertNotIn('secret-code',str(e.exception))
        for result in [{}, {'access_token':'bad token'}, []]:
            opener=Mock();opener.open.return_value=io.BytesIO(json.dumps(result).encode())
            with patch('app.oauth.urllib.request.build_opener',return_value=opener),self.assertRaises(OAuthError):self.flow.exchange('code')

    def test_dotenv_has_no_shell_expansion_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'.env').write_text('UPSTOX_CLIENT_ID="file-key"\nUPSTOX_CLIENT_SECRET=literal-$value\nIGNORED=123\n',encoding='utf-8')
            with patch.dict(os.environ,{'UPSTOX_CLIENT_ID':'env-key'},clear=True):
                config=OAuthConfig.load(directory,9999)
            self.assertEqual(config.client_id,'env-key');self.assertEqual(config.client_secret,'literal-$value')
            self.assertEqual(config.redirect_uri,'http://127.0.0.1:9999'+CALLBACK)

    def test_provider_error_codes_are_actionable_without_echoing_body(self):
        for provider_code, expected in [('UDAPI100069','credentials'),('UDAPI100070','redirect'),
                                        ('UDAPI100057','auth_code'),('UNKNOWN','failed')]:
            body=json.dumps({'errors':[{'errorCode':provider_code,'message':'test-secret private-code'}]}).encode()
            opener=Mock()
            opener.open.side_effect=urllib.error.HTTPError('https://api.upstox.com',400,'Bad Request',{},io.BytesIO(body))
            with patch('app.oauth.urllib.request.build_opener',return_value=opener),self.assertRaises(OAuthError) as caught:
                self.flow.exchange('private-code')
            self.assertEqual(caught.exception.code,expected)
            self.assertNotIn('test-secret',str(caught.exception))
            self.assertNotIn('private-code',str(caught.exception))

    def test_error_status_is_sanitized_and_cleared_by_new_login(self):
        self.flow.record_error('credentials')
        self.assertEqual(self.flow.error_status()['code'],'credentials')
        self.flow.record_error('private-secret')
        self.assertNotIn('private-secret',json.dumps(self.flow.error_status()))
        self.begin()
        self.assertIsNone(self.flow.error_status())

    def test_cloudflare_block_is_not_misreported_as_bad_credentials(self):
        body=json.dumps({'cloudflare_error':True,'error_code':1010,
                         'detail':'private-code test-secret must never be echoed'}).encode()
        opener=Mock()
        opener.open.side_effect=urllib.error.HTTPError('https://api.upstox.com',403,'Forbidden',{},io.BytesIO(body))
        with patch('app.oauth.urllib.request.build_opener',return_value=opener),self.assertRaises(OAuthError) as caught:
            self.flow.exchange('private-code')
        self.assertEqual(caught.exception.code,'client_blocked')
        self.assertIn('Cloudflare 1010',str(caught.exception))
        self.assertNotIn('test-secret',str(caught.exception))
        self.assertNotIn('private-code',str(caught.exception))
        self.assertEqual(opener.open.call_count,1)


class OAuthApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.app=Application(self.temp.name,oauth_config=OAuthConfig('test-key','test-secret'))
        self.origin='http://127.0.0.1:8765'
        self.authorized=dict(token='private-access-token',user_name='Test User',expires_at=(now()+timedelta(hours=8)).isoformat())
        self.quotes=dict(self.app.quotes)

    def begin(self):
        result,cookie=self.app.start_oauth({'exchange':'NSE'},self.origin)
        params=parse_qs(urlparse(result['authorization_url']).query)
        return {'state':params['state'],'code':['private-code']},cookie

    def test_success_connects_without_returning_token_or_enabling_trading(self):
        params,cookie=self.begin()
        with patch.object(self.app.oauth,'exchange',return_value=self.authorized),patch.object(self.app.provider,'quotes',return_value=self.quotes):
            self.app.finish_oauth(params,cookie,self.origin)
        state=self.app.snapshot()
        self.assertEqual(state['account_key'],'live:NSE');self.assertTrue(state['connected'])
        self.assertEqual(state['oauth']['user']['user_name'],'Test User');self.assertFalse(state['terminal']['enabled'])
        for private in ['private-access-token','private-code','test-secret']:
            self.assertNotIn(private,json.dumps(state))
        self.assertEqual(self.app.provider.token,'private-access-token')

    def test_failed_or_cancelled_login_preserves_existing_connection(self):
        self.app.provider.token='existing-token'
        params,cookie=self.begin()
        with patch.object(self.app.oauth,'exchange',side_effect=OAuthError('failed','Provider rejected code')):
            with self.assertRaises(OAuthError):self.app.finish_oauth(params,cookie,self.origin)
        self.assertEqual(self.app.provider.token,'existing-token')
        params,cookie=self.begin();params['error']=['denied']
        with self.assertRaises(OAuthError):self.app.finish_oauth(params,cookie,self.origin)
        self.assertEqual(self.app.provider.token,'existing-token')

    def test_quote_outage_keeps_new_token_and_shows_feed_error(self):
        params,cookie=self.begin()
        with patch.object(self.app.oauth,'exchange',return_value=self.authorized),patch.object(self.app.provider,'quotes',side_effect=ValueError('Quotes unavailable')):
            self.app.finish_oauth(params,cookie,self.origin)
        self.assertEqual(self.app.provider.token,'private-access-token');self.assertEqual(self.app.error,'Quotes unavailable')

    def test_changed_source_during_exchange_cannot_overwrite_connection(self):
        self.app.provider.token='old-token';params,cookie=self.begin()
        def exchange(code):
            self.app.terminal.reset()
            return self.authorized
        with patch.object(self.app.oauth,'exchange',side_effect=exchange),self.assertRaises(OAuthError):
            self.app.finish_oauth(params,cookie,self.origin)
        self.assertEqual(self.app.provider.token,'old-token')

    def test_expired_token_pauses_entries_and_requests_reconnect(self):
        self.app.mode='live';self.app.provider.token='expired-token'
        self.app.oauth_user={'user_name':'Test','expires_at':(now()-timedelta(seconds=1)).isoformat()}
        self.app.terminal.enabled=True
        with patch.object(self.app.provider,'quotes') as quotes:self.app.poll()
        quotes.assert_not_called();self.assertFalse(self.app.terminal.enabled)
        self.assertFalse(self.app.snapshot()['connected']);self.assertIn('expired',self.app.error)

    def test_http_callback_cookie_roundtrip_clean_redirect_and_replay(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.app))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base='http://127.0.0.1:'+str(server.server_port)
        self.app.oauth.config.redirect_uri=base+CALLBACK
        jar=http.cookiejar.CookieJar()
        client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        request=urllib.request.Request(base+'/api/auth/upstox/start',data=b'{"exchange":"NSE"}',
            headers={'X-CSRF-Token':self.app.csrf,'Content-Type':'application/json'})
        with client.open(request) as response:result=json.load(response)
        params=parse_qs(urlparse(result['authorization_url']).query)
        callback=base+CALLBACK+'?'+urlencode({'state':params['state'][0],'code':'one-use'})
        with patch.object(self.app.oauth,'exchange',return_value=self.authorized) as exchange,patch.object(self.app.provider,'quotes',return_value=self.quotes):
            with client.open(callback) as response:
                self.assertEqual(response.url,base+'/?upstox=connected')
                self.assertNotIn(b'private-access-token',response.read())
            with client.open(callback) as response:self.assertEqual(response.url,base+'/?upstox=invalid_state')
            self.assertEqual(exchange.call_count,1)
            self.assertEqual(self.app.snapshot()['oauth']['last_error']['code'],'invalid_state')
        self.assertFalse(list(jar))

    def test_second_server_cannot_share_oauth_callback_port(self):
        first=LocalHTTPServer(('127.0.0.1',0),handler_for(self.app))
        self.addCleanup(first.server_close)
        with self.assertRaises(OSError):
            second=LocalHTTPServer(('127.0.0.1',first.server_port),handler_for(self.app))
            self.addCleanup(second.server_close)


if __name__=='__main__':unittest.main()
