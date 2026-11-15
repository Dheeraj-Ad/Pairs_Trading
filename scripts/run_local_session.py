"""Run a bounded, real simulator session and persist paper fills (never broker orders).

Run with the app server stopped. Existing trades are preserved. All entries pass
through PaperBook; automatic entries use the normal qualified model/risk gates.
"""
import argparse
import json
from pathlib import Path
import socket
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.server import Application


def run_session(app, bars=150):
    if app.mode != 'demo':
        raise ValueError('This session runner supports local synthetic paper trading only.')
    if app.book.read(app.account_key)['positions']:
        raise ValueError('Close existing positions in the application before running this separate local session.')
    if not 10 <= bars <= 750:
        raise ValueError('Choose 10 to 750 candles.')
    before = {t['id'] for t in app.book.read(app.account_key)['trades']}
    app.terminal.scan()
    while app.terminal.job['running']:
        time.sleep(.05)
    qualified = [r['id'] for r in app.terminal.screen if r['eligible']]
    print(json.dumps({'qualified_pairs': qualified}), flush=True)
    if not qualified:
        raise ValueError('No pairs passed the screen. No fallback trades are fabricated.')
    # Exercise the manual ticket through its normal API, then close at future prices.
    for pair_id in qualified[:3]:
        signal = app.terminal.signals[pair_id]
        app.action('/api/open', dict(account_key=app.account_key, pair=pair_id,
            side=signal['side'], budget=50000, request_id=str(uuid.uuid4())))
    app.advance_local(5)
    for position in list(app.book.read(app.account_key)['positions']):
        app.action('/api/close', dict(account_key=app.account_key, position_id=position['id'], request_id=str(uuid.uuid4())))
    app.terminal.configure({'enabled': True})
    try:
        for step in range(bars):
            app.advance_local()
            if (step+1) % 25 == 0:
                print(json.dumps({'candles_advanced': step+1, 'closed_trades': len(app.book.read(app.account_key)['trades'])}), flush=True)
    finally:
        app.terminal.configure({'enabled': False})
        # Let the normal risk rules finish any open automated position.
        for _ in range(6):
            if not app.book.read(app.account_key)['positions']: break
            app.advance_local()
    if app.book.read(app.account_key)['positions']:
        raise ValueError('A position still needs attention. Open the app and review it before continuing.')
    state = app.snapshot()
    created = [t for t in state['account']['trades'] if t['id'] not in before]
    result = dict(account=app.account_key, source='Synthetic local data; paper money only',
                  qualified_pairs=qualified, new_trades=len(created),
                  automatic=sum(t.get('origin')=='auto' for t in created),
                  manual=sum(t.get('origin')=='manual' for t in created),
                  wins=sum(t['pnl']>0 for t in created), losses=sum(t['pnl']<0 for t in created),
                  new_net_pnl=sum(t['pnl'] for t in created), performance=state['performance'],
                  transaction_fills=len(state['transactions']), market_time=state['market_time'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars',type=int,default=150)
    parser.add_argument('--port',type=int,default=8765,help='Port to check for a running app before writing its ledger')
    args=parser.parse_args()
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1',args.port)) == 0:
            parser.error('Stop the app server first so two processes cannot advance the same local market.')
    directory=ROOT/'runtime'
    database=directory/'paper.sqlite3'
    if database.exists():
        backup=directory/('before-local-session-'+uuid.uuid4().hex+'.sqlite3')
        source=sqlite3.connect(database)
        target=sqlite3.connect(backup)
        try: source.backup(target)
        finally: target.close(); source.close()
    result=run_session(Application(directory,initial_mode='demo'),args.bars)
    output=ROOT/'outputs'/'local-session.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
