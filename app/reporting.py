"""Reconciled active-account attribution and an immutable view of all leg fills."""
from app.paper import FEE


def summary(account):
    def group(positions, trades, label):
        realized = sum(t['pnl'] for t in trades)
        open_net = sum(p['unrealized'] - p['entry_fee'] for p in positions)
        return dict(label=label, realized=realized, open_net=open_net, total=realized + open_net,
                    fees=sum(t['fees'] for t in trades) + sum(p['entry_fee'] for p in positions),
                    borrow=sum(t['borrow'] for t in trades) + sum(p['borrow'] for p in positions),
                    closed=len(trades), open=len(positions), wins=sum(t['pnl'] > 0 for t in trades),
                    losses=sum(t['pnl'] < 0 for t in trades))
    positions, trades = account['positions'], account['trades']
    methods = [group([p for p in positions if p.get('origin', 'manual') == origin],
                     [t for t in trades if t.get('origin', 'manual') == origin], origin)
               for origin in ('manual', 'auto')]
    pairs = [group([p for p in positions if p['pair'] == pair],
                   [t for t in trades if t['pair'] == pair], pair)
             for pair in sorted({p['pair'] for p in positions + trades})]
    return dict(overall=group(positions, trades, 'Overall'), methods=methods, pairs=pairs)


def transactions(account):
    rows = []
    for trade in account['positions'] + account['trades']:
        for event, timestamp, legs in [('entry', trade['opened'], trade.get('entry_legs', trade.get('legs', []))),
                                       ('exit', trade.get('closed'), trade.get('exit_legs', []))]:
            for leg in legs:
                rows.append(dict(trade_id=trade['id'], time=timestamp, event=event, pair=trade['pair'],
                    origin=trade.get('origin', 'manual'), symbol=leg['symbol'],
                    side='BUY' if leg['qty'] > 0 else 'SELL', quantity=abs(leg['qty']),
                    price=leg['fill'], value=abs(leg['qty']) * leg['fill'],
                    fee=abs(leg['qty']) * leg['fill'] * FEE))
    return sorted(rows, key=lambda row: (row['time'], row['trade_id'], row['event']))
