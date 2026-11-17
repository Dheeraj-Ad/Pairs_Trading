"""Test real browser redirects/cookies against mocked Upstox responses only."""
from datetime import timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
from urllib.parse import parse_qs, urlencode, urlparse
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from app.market import now
from app.oauth import OAuthConfig, CALLBACK
from app.server import Application, handler_for


def main():
    output=Path(__file__).resolve().parents[1]/'outputs'/'oauth-preview'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        app=Application(directory,oauth_config=OAuthConfig())
        quotes=dict(app.quotes);app.provider.token=''
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
        base=f'http://127.0.0.1:{server.server_port}'
        app.oauth.config.redirect_uri=base+CALLBACK
        threading.Thread(target=server.serve_forever,daemon=True).start()
        authorized={'token':'browser-private-token','user_name':'Test Account','expires_at':(now()+timedelta(hours=8)).isoformat()}
        try:
            with patch.object(app.oauth,'exchange',return_value=authorized) as exchange,patch.object(app.provider,'quotes',return_value=quotes),sync_playwright() as pw:
                browser=pw.chromium.launch(channel='chrome',headless=True)
                context=browser.new_context(viewport={'width':1440,'height':1000})
                page=context.new_page();errors=[];deny=False
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
                def authorization(route):
                    params=parse_qs(urlparse(route.request.url).query)
                    assert params['client_id']==['test-browser-key']
                    assert 'browser-app-secret' not in route.request.url
                    response={'state':params['state'][0]}
                    response.update({'error':'access_denied'} if deny else {'code':'browser-code'})
                    route.fulfill(status=302,headers={'Location':base+CALLBACK+'?'+urlencode(response)},body='')
                page.route('https://api.upstox.com/v2/login/authorization/dialog?*',authorization)
                page.goto(base)
                page.locator('#connection-button').click()
                expect(page.locator('#mode')).to_have_value('demo')
                expect(page.locator('#oauth-main')).to_be_visible()
                expect(page.locator('#oauth-setup')).to_be_visible()
                expect(page.locator('#oauth-start')).to_be_disabled()
                page.screenshot(path=str(output/'developer-setup.png'))
                page.goto(base+'/?connect=upstox')
                expect(page.locator('#connection')).to_be_visible()
                expect(page.locator('#oauth-start')).to_be_visible()
                assert app.mode=='demo', 'Opening sign-in must not switch the account.'
                app.oauth.config.client_id='test-browser-key';app.oauth.config.client_secret='browser-app-secret'
                expect(page.locator('#oauth-start')).to_be_enabled(timeout=10000)
                expect(page.locator('#oauth-setup')).to_be_hidden()
                page.locator('#exchange').select_option('BSE')
                page.locator('#oauth-start').click()
                expect(page.locator('#banner')).to_contain_text('BSE',timeout=30000)
                expect(page.locator('#connection-button')).to_have_text('Data connection')
                page.wait_for_function('!location.search')
                assert app.provider.token=='browser-private-token'
                assert exchange.call_count==1
                assert app.exchange=='BSE'
                assert not app.terminal.enabled
                assert not context.cookies()
                assert 'browser-private-token' not in page.content()
                assert 'browser-app-secret' not in page.content()
                page.locator('#connection-button').click()
                expect(page.locator('#oauth-status')).to_contain_text('Test Account')
                deny=True
                page.locator('#oauth-start').click()
                expect(page.locator('#toast')).to_contain_text('cancelled',timeout=30000)
                expect(page.locator('#connection')).to_be_visible()
                assert app.provider.token=='browser-private-token';assert exchange.call_count==1
                expect(page.locator('#oauth-status')).to_contain_text('cancelled')
                page.reload()
                page.locator('#connection-button').click()
                expect(page.locator('#oauth-status')).to_contain_text('cancelled')
                # A new attempt clears the prior diagnostic, including after a reload.
                with app.oauth.lock:app.oauth.last_error=None
                app.oauth_user['expires_at']=(now()-timedelta(seconds=1)).isoformat();app.poll()
                expect(page.locator('#oauth-status')).to_contain_text('expired',timeout=10000)
                expect(page.locator('#oauth-start')).to_contain_text('Reconnect')
                page.set_viewport_size({'width':390,'height':844})
                page.screenshot(path=str(output/'reconnect-mobile.png'))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert not errors,'\n'.join(errors)
                browser.close()
                print('OAuth browser passed: setup, provider redirect, callback cookie, automatic connection, clean URL, cancellation, expiry and mobile layout. Provider responses were mocked.')
        finally:
            server.shutdown();server.server_close()


if __name__=='__main__':main()
