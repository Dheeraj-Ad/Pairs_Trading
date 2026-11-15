"""Stage 5: delayed daily events, equal-dollar entries and portfolio metrics."""
import math
import numpy as np
from ml.report.features import COSTS


def metrics(equity,trades,capital):
    values=np.array([r['equity'] for r in equity],dtype=float)
    returns=np.diff(np.r_[capital,values])/np.r_[capital,values][:-1]
    std=returns.std(ddof=1) if len(returns)>1 else 0.
    downside=float(np.sqrt(np.mean(np.minimum(returns,0.)**2))) if len(returns) else 0.
    peaks=np.maximum.accumulate(np.r_[capital,values])[1:]
    drawdown=1-values/peaks
    wins=int(sum(t['pnl']>0 for t in trades));losses=int(sum(t['pnl']<0 for t in trades))
    turnover=sum(t['raw_turnover'] for t in trades)
    gross=sum(t['raw_price_pnl'] for t in trades)
    borrow=sum(t['borrow'] for t in trades)
    return dict(capital=capital,final_equity=float(values[-1]) if len(values) else capital,
        net_pnl=float(values[-1]-capital) if len(values) else 0.,
        roi=float(values[-1]/capital-1) if len(values) else 0.,
        sharpe=float(np.sqrt(252)*returns.mean()/std) if std>1e-12 else None,
        sortino=float(np.sqrt(252)*returns.mean()/downside) if downside>1e-12 else None,
        max_drawdown=float(drawdown.max()) if len(drawdown) else 0.,
        wins=wins,losses=losses,flat=len(trades)-wins-losses,
        win_loss_ratio=float(wins/losses) if losses else None,
        trades=len(trades),turnover=turnover,fees=sum(t['fees'] for t in trades),borrow=borrow,
        breakeven_all_in_bps=float((gross-borrow)/turnover*10000) if turnover else None,
        annualization=252,risk_free_rate=0.,daily_observations=len(returns))


def backtest(prices,bundles,fee=None,capital=1_000_000.):
    fee=COSTS['fee'] if fee is None else fee
    friction=COSTS['slippage']+COSTS['half_spread']
    start=int(len(prices)*.8)
    balance=capital;positions={};pending={};trades=[];ledger=[];events=[]
    def borrow(p,index):
        days=(prices.index[index]-prices.index[p['entry_index']]).total_seconds()/86400
        short=sum(abs(q)*v for q,v in zip(p['qty'],p['entry_fills']) if q<0)
        return short*COSTS['borrow']*days/365
    def mark(index):
        unrealized=0.;gross=0.
        for p in positions.values():
            current=[float(prices[s].iloc[index]) for s in p['symbols']]
            unrealized+=sum(q*(v-e) for q,v,e in zip(p['qty'],current,p['entry_fills']))-borrow(p,index)
            gross+=sum(abs(q)*v for q,v in zip(p['qty'],current))
        return balance+unrealized,gross
    def close(pair_id,index,reason):
        nonlocal balance
        p=positions.pop(pair_id)
        raw=[float(prices[s].iloc[index]) for s in p['symbols']]
        fills=[v*(1-np.sign(q)*friction) for q,v in zip(p['qty'],raw)]
        exit_fee=fee*sum(abs(q)*v for q,v in zip(p['qty'],fills))
        price_pnl=sum(q*(v-e) for q,v,e in zip(p['qty'],fills,p['entry_fills']))
        cost_borrow=borrow(p,index)
        balance+=price_pnl-exit_fee-cost_borrow
        net=price_pnl-p['entry_fee']-exit_fee-cost_borrow
        trade=dict(pair=pair_id,symbols=p['symbols'],side=p['signal']['side'],
            signal_index=p['signal_index'],entry_index=p['entry_index'],exit_index=index,
            signal_time=str(prices.index[p['signal_index']]),opened=str(prices.index[p['entry_index']]),closed=str(prices.index[index]),
            quantity=p['qty'],entry_fills=p['entry_fills'],exit_fills=fills,
            fees=p['entry_fee']+exit_fee,borrow=cost_borrow,pnl=net,exit_reason=reason,
            score=p['signal']['score'],features=p['signal']['features'],high_volatility=p['signal']['high_volatility'],
            raw_price_pnl=sum(q*(v-e) for q,v,e in zip(p['qty'],raw,p['entry_quotes'])),
            raw_turnover=sum(abs(q)*(v+e) for q,v,e in zip(p['qty'],raw,p['entry_quotes'])),
            dollar_imbalance=p['dollar_imbalance'])
        trades.append(trade)
    for index in range(start,len(prices)):
        for pair_id,order in list(pending.items()):
            if order['execute_index']!=index:continue
            pending.pop(pair_id)
            if order['kind']=='exit':
                if pair_id in positions:close(pair_id,index,order['reason'])
                continue
            if pair_id in positions or index==len(prices)-1:continue
            bundle=bundles[pair_id];signal=order['signal']
            symbols=[bundle['report']['y'],bundle['report']['x']]
            quotes=[float(prices[s].iloc[index]) for s in symbols]
            equity,gross=mark(index)
            budget=min(equity*.15,max(0,equity-gross)/(1+fee+friction))
            if signal['high_volatility']:budget*=.5
            if budget<1000:continue
            d=1 if signal['side']=='long' else -1
            qty=[d*math.floor(budget/2/quotes[0]/(1+friction)), -d*math.floor(budget/2/quotes[1]/(1+friction))]
            if min(map(abs,qty))<1:continue
            fills=[v*(1+np.sign(q)*friction) for q,v in zip(qty,quotes)]
            entry_fee=fee*sum(abs(q)*v for q,v in zip(qty,fills))
            balance-=entry_fee
            positions[pair_id]=dict(symbols=symbols,qty=qty,entry_quotes=quotes,entry_fills=fills,
                entry_fee=entry_fee,entry_index=index,signal_index=order['signal_index'],signal=signal,
                dollar_imbalance=abs(abs(qty[0])*quotes[0]-abs(qty[1])*quotes[1]))
        for pair_id,bundle in bundles.items():
            signal=bundle['signals'].get(index)
            if pair_id in positions:
                p=positions[pair_id]
                age=index-p['entry_index'];reason=None
                if signal is None:reason='Unavailable daily signal'
                elif signal['action']=='Exit / halt':reason='Breakdown proxy'
                elif abs(signal['zscore'])>=4:reason='Spread stop'
                elif (p['signal']['side']=='long' and signal['zscore']>=-.5) or (p['signal']['side']=='short' and signal['zscore']<=.5):reason='Spread reverted'
                elif age>=4:reason='Five-session holding limit'  # next-row execution makes age 5.
                if reason and index+1<len(prices):pending[pair_id]=dict(kind='exit',execute_index=index+1,reason=reason)
            elif signal and signal['action'] in ('Long','Short') and index+1<len(prices)-1:
                pending[pair_id]=dict(kind='entry',execute_index=index+1,signal_index=index,signal=signal)
                events.append(dict(pair=pair_id,time=str(prices.index[index]),action=signal['action'],score=signal['score']))
        if index==len(prices)-1:
            for pair_id in list(positions):close(pair_id,index,'Scheduled end-of-test liquidation')
        equity,gross=mark(index)
        benchmark=capital*float((prices.iloc[index]/prices.iloc[start]).mean())
        ledger.append(dict(time=str(prices.index[index]),equity=float(equity),gross=float(gross),benchmark=benchmark))
    return dict(metrics=metrics(ledger,trades,capital),equity=ledger,trades=trades,events=events,
        costs={**COSTS,'fee':fee},execution='Signal at daily close; execute at the next observed daily close',
        sizing='Equal-dollar gross targets, integer rounding; half allocation in predicted high-volatility regimes')
