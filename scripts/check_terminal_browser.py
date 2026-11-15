"""Browser QA using an isolated ledger and explicitly labeled synthetic data."""
from pathlib import Path
import sys
import tempfile
import threading
import uuid
from http.server import ThreadingHTTPServer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from app.server import Application, handler_for
from app.market import PAIRS


def main():
    output=Path(__file__).resolve().parents[1]/'outputs'/'terminal-preview'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        app=Application(directory,initial_mode='live')
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with sync_playwright() as pw:
                browser=pw.chromium.launch(channel='chrome',headless=True)
                page=browser.new_page(viewport={'width':1512,'height':1000},device_scale_factor=1)
                errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
                page.goto(f'http://127.0.0.1:{server.server_port}')
                expect(page.locator('#feed')).to_have_text('DISCONNECTED')
                expect(page.locator('#charts')).to_be_hidden()
                expect(page.locator('#start-auto')).to_be_disabled()
                page.locator('#connection-button').click()
                page.locator('#connection-options summary').click()
                page.locator('#mode').select_option('demo')
                page.locator('#connection-form button[type="submit"]').click()
                expect(page.locator('#connection')).not_to_be_visible()
                page.locator('#scan').click()
                expect(page.locator('#pair-list [data-pair="banks"]')).to_be_visible(timeout=60000)
                expect(page.locator('#charts')).to_be_visible()
                expect(page.locator('#pair-title')).to_have_text('HDFCBANK / ICICIBANK')
                expect(page.locator('#start-auto')).to_be_enabled()
                expect(page.locator('#local-controls')).to_be_visible()
                assert page.locator('#ohlc-y').inner_text().find(' O ')>=0
                before=page.evaluate('charts[0].count')
                page.locator('#candle-y').hover();page.mouse.wheel(0,-200)
                page.wait_for_function(f'charts[0].count < {before}')
                page.locator('[data-interval="15"]').click()
                expect(page.locator('#pair-subtitle')).to_contain_text('15-minute')
                page.locator('[data-interval="5"]').click()
                page.locator('#reset-charts').click()
                page.locator('#toast').evaluate('(e)=>e.hidden=true')
                page.screenshot(path=str(output/'terminal-desktop.png'),full_page=True)
                page.locator('[data-page="models"]').click()
                expect(page.locator('#model-table tbody tr')).to_have_count(len(PAIRS))
                expect(page.locator('#candidate-universe')).to_contain_text('20 companies')
                expect(page.locator('#history-pair option')).to_have_count(len(PAIRS)+1)
                expect(page.locator('#model-table')).to_contain_text('REJECTED')
                page.screenshot(path=str(output/'models-desktop.png'),full_page=True)
                page.locator('[data-page="terminal"]').click()
                page.locator('#manual-open').click()
                expect(page.locator('[data-close]')).to_have_count(1,timeout=10000)
                page.locator('#local-step').click()
                expect(page.locator('#local-clock')).to_contain_text('1 candles advanced',timeout=10000)
                page.locator('[data-close]').click()
                page.locator('[data-page="history"]').click()
                expect(page.locator('#history-table tbody tr')).to_have_count(1)
                expect(page.locator('#history-table')).to_contain_text('Manual close')
                expect(page.locator('#transaction-table tbody tr')).to_have_count(4)
                page.locator('#history-origin').select_option('auto')
                expect(page.locator('#transaction-table')).to_contain_text('No matching transactions')
                page.locator('#history-origin').select_option('all')
                page.locator('[data-page="performance"]').click()
                expect(page.locator('#performance-methods tbody tr')).to_have_count(2)
                expect(page.locator('#performance-pairs tbody tr')).to_have_count(1)
                page.screenshot(path=str(output/'performance-desktop.png'),full_page=True)
                page.locator('[data-page="models"]').click()
                expect(page.locator('#paper-training-status')).to_contain_text('1 / 40 eligible')
                expect(page.locator('#train-paper')).to_be_disabled()
                page.locator('[data-page="history"]').click()
                page.locator('#connection-button').click()
                page.locator('#exchange').select_option('BSE')
                page.locator('#connection-form button[type="submit"]').click()
                expect(page.locator('#history-table')).to_contain_text('No completed trades')
                page.locator('[data-page="terminal"]').click()
                expect(page.locator('#charts')).to_be_hidden()
                page.locator('#scan').click()
                expect(page.locator('#pair-list [data-pair="banks"]')).to_be_visible(timeout=60000)
                page.set_viewport_size({'width':390,'height':844})
                page.locator('#toast').evaluate('(e)=>e.hidden=true')
                page.screenshot(path=str(output/'terminal-mobile.png'),full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),'Mobile overflow'
                assert not errors,'\n'.join(errors)
                browser.close()
                print('Browser passed: disconnected state, candidate filtering, OHLCV, zoom, timeframes, history, account isolation, mobile layout, no JS errors.')
        finally:
            server.shutdown();server.server_close()


if __name__=='__main__':main()
