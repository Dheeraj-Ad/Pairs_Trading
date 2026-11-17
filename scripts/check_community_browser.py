"""Community UI QA uses a labeled fixture; demo trades use real simulator models."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from app.server import Application, handler_for
from app.market import PAIRS


def main():
    output=Path('outputs/community-preview');output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        app=Application(directory)
        original=app.book.read(app.account_key);quotes=deepcopy(app.quotes)
        rows={p[leg]:dict(symbol=p[leg],price=100.+i,change=1.,percent_change=1.,volume=1000.,exchange_trade_time=None)
              for i,p in enumerate(PAIRS) for leg in ('y','x')}
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with patch('app.community.fetch_snapshots',return_value=(rows,[])),sync_playwright() as pw:
                browser=pw.chromium.launch(channel='chrome',headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1000})
                errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.locator('[data-page="community"]').click()
                page.locator('#community-url').fill('http://127.0.0.1:8787')
                page.locator('#community-refresh').click()
                expect(page.locator('#community-status')).to_contain_text('20/20 stocks',timeout=15000)
                expect(page.locator('#community-table tbody tr')).to_have_count(10)
                assert app.book.read(app.account_key)==original
                assert app.quotes==quotes
                page.locator('#demo-run-community').click()
                expect(page.locator('#demo-session-status')).to_contain_text('Completed:',timeout=180000)
                assert not app.demo_session.job['error'],app.demo_session.job
                assert app.demo_session.job['new_trades']>0,app.demo_session.job
                assert not app.book.read(app.account_key)['positions']
                assert all(t['origin']=='auto' for t in app.book.read(app.account_key)['trades'])
                with app.lock:app.community.next_poll=0
                expect(page.locator('#community-refresh')).to_be_enabled(timeout=10000)
                page.locator('#community-refresh').click()
                expect(page.locator('#community-plot svg')).to_be_visible(timeout=15000)
                page.screenshot(path=str(output/'community-fixture-desktop.png'),full_page=True)
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Mobile overflow'
                page.screenshot(path=str(output/'community-fixture-mobile.png'),full_page=True)
                page.locator('#page-community [data-page="history"]').click()
                expect(page.locator('#history-table')).to_contain_text('auto ·')
                page.screenshot(path=str(output/'automated-demo-history.png'),full_page=True)
                assert not errors,errors
                print('Passed: snapshot comparison stays separate from execution, auto demo uses actual model gates, histories, observed-price plot and mobile layout.')
                print(app.demo_session.job)
                browser.close()
        finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
