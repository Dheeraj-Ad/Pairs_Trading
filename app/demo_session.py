"""Bounded automated demonstration using ordinary model and execution rules."""
import json
import threading
import time


class DemoSession:
    def __init__(self, app):
        self.app = app
        self.cancelled = False
        self.job = dict(running=False, message='Run a bounded automated session using synthetic local prices.',
                        account=None, advanced=0, new_trades=0, pnl=0., error=None)

    def start(self):
        app = self.app
        if app.mode != 'demo': raise ValueError('Select Local simulation before running the automated demo.')
        if self.job['running'] or app.local.batch_active or app.terminal.job['running']:
            raise ValueError('Wait for the current simulation or scan to finish.')
        if app.book.read(app.account_key)['positions']:
            raise ValueError('Close existing positions before starting a bounded demonstration.')
        self.cancelled = False
        account = app.account_key
        before = {t['id'] for t in app.book.read(account)['trades']}
        self.job = dict(running=True, account=account, message='Scanning and training the actual demo models...',
                        advanced=0, new_trades=0, pnl=0., error=None)
        app.local.running = False
        try:
            app.terminal.scan()
        except Exception:
            self.job['running'] = False
            raise
        def worker():
            try:
                deadline = time.monotonic()+180
                while app.terminal.job['running']:
                    if time.monotonic()>deadline: raise ValueError('Model scan timed out. Wait for the scan before trying again.')
                    time.sleep(.1)
                with app.lock:
                    if self.cancelled: return
                    app.terminal.configure({'enabled':True})
                for _ in range(150):
                    with app.lock:
                        if self.cancelled: break
                        app.advance_local()
                        self.job['advanced'] += 1
                        trades = [t for t in app.book.read(account)['trades'] if t['id'] not in before]
                        self.job.update(message=f"Advancing synthetic prices: {self.job['advanced']} / 150 candles",
                                        new_trades=len(trades), pnl=sum(t['pnl'] for t in trades))
                    time.sleep(.01)
            except Exception as exc:
                with app.lock:
                    self.job['error'] = str(exc) if isinstance(exc,ValueError) else 'Automated demo failed. Review the paper positions.'
            finally:
                with app.lock:
                    app.terminal.enabled = False
                    app.local.running = False
                    try:
                        # Keep the normal risk manager active to close remaining auto positions.
                        for _ in range(6):
                            if not app.book.read(account)['positions']: break
                            app.advance_local()
                        trades = [t for t in app.book.read(account)['trades'] if t['id'] not in before]
                        self.job.update(new_trades=len(trades), pnl=sum(t['pnl'] for t in trades),
                                        wins=sum(t['pnl']>0 for t in trades), losses=sum(t['pnl']<0 for t in trades))
                        self.job['message'] = (self.job['error'] or
                            f"{'Stopped' if self.cancelled else 'Completed'}: {len(trades)} automatic paper trades. Open Trade history and P&L.")
                        if app.book.read(account)['positions']:
                            self.job['error'] = self.job['message'] = 'A paper position still needs attention. Review Positions.'
                    except Exception:
                        self.job['error'] = self.job['message'] = 'Could not finish the demo. Review paper positions before continuing.'
                    self.job['running'] = False
                    (app.directory/'automated-demo.json').write_text(json.dumps(self.job,allow_nan=False,indent=2),encoding='utf-8')

        threading.Thread(target=worker, daemon=True).start()
