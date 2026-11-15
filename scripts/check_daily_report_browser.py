"""Exercise the report job, downloads, persistence and source isolation in Chrome."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from app.server import Application, handler_for


def main():
    output = Path(__file__).resolve().parents[1]/'outputs'/'daily-report-preview'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        app = Application(directory)
        original_account = app.book.read(app.account_key)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel='chrome', headless=True)
                page = browser.new_page(viewport={'width':1512, 'height':1000})
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('console', lambda m: errors.append(m.text) if m.type=='error' else None)
                base = f'http://127.0.0.1:{server.server_port}'
                page.goto(base)
                page.locator('[data-page="report"]').click()
                page.locator('#run-report').click()
                expect(page.locator('#report-job')).to_contain_text('Completed:', timeout=180000)
                expect(page.locator('#report-content')).to_contain_text('synthetic generated daily prices')
                expect(page.locator('#report-content')).to_contain_text('Research trade history')
                expect(page.locator('#report-pair option')).to_have_count(1)
                result = app.report_lab.view()['result']
                assert result and not app.report_lab.job['error']
                for name in result['artifacts']+['report.json']:
                    response = page.request.get(base+'/api/report/artifact', params={'run':result['run_id'], 'name':name})
                    assert response.status==200, name
                    assert response.body(), name
                for img in page.locator('#report-content img').all():
                    img.scroll_into_view_if_needed()
                    expect(img).to_be_visible()
                    img.evaluate('(e)=>e.loading="eager"')
                page.wait_for_function('Array.from(document.querySelectorAll("#report-content img")).every(e=>e.complete && e.naturalWidth>0)')
                page.screenshot(path=str(output/'daily-report-desktop.png'), full_page=True)
                page.set_viewport_size({'width':390, 'height':844})
                page.screenshot(path=str(output/'daily-report-mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile overflow'
                assert app.book.read('demo:NSE') == original_account, 'Research mutated the paper account'
                page.set_viewport_size({'width':1512, 'height':1000})
                page.locator('#connection-button').click()
                page.locator('#connection-options summary').click()
                page.locator('#exchange').select_option('BSE')
                page.locator('#connection-form button[type="submit"]').click()
                expect(page.locator('#report-content')).to_contain_text('Run the daily workflow for this account')
                expect(page.locator('#report-content img')).to_have_count(0)
                denied = page.request.get(base+'/api/report/artifact', params={'run':result['run_id'], 'name':'report.json'})
                assert denied.status==404, 'Cross-account artifact was accessible'
                restored = Application(directory)
                assert restored.report_lab.view()['result']['run_id']==result['run_id']
                assert not errors, '\n'.join(errors)
                browser.close()
                print('Passed: daily job, models, research history, all CSV/PNG/JSON downloads, image decoding, mobile layout, account isolation, restart persistence; paper account unchanged.')
                print(json.dumps(result['backtest']['metrics'], indent=2))
        finally:
            server.shutdown()
            server.server_close()


if __name__=='__main__':
    main()
