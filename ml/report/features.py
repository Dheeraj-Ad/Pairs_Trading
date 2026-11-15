"""Stage 3: causal rolling Ridge hedge, price features and delayed labels."""
import numpy as np
import pandas as pd

HORIZON = 5
FEATURES = ['zscore','abs_zscore','spread_change','spread_volatility','y_momentum',
            'x_momentum','momentum_gap','market_volatility','beta_change']
COSTS = dict(fee=.0003, slippage=.0005, half_spread=.0002, borrow=.10)


def engineer(prices, pair):
    y, x = prices[pair['y']], prices[pair['x']]
    # Rolling standardized one-predictor Ridge, using only the preceding 120 rows.
    mx, my = x.shift().rolling(120).mean(), y.shift().rolling(120).mean()
    vx = x.shift().rolling(120).var(ddof=0)
    cov = x.shift().rolling(120).cov(y.shift(), ddof=0)
    beta = (cov/vx)/(1+.1/120)  # Ridge alpha .1 on standardized X; intercept unpenalized.
    alpha = my-beta*mx
    spread = y-alpha-beta*x
    mean = spread.shift().rolling(60).mean()
    std = spread.shift().rolling(60).std().replace(0, np.nan)
    z = (spread-mean)/std
    ry, rx = np.log(y).diff(), np.log(x).diff()
    market = np.log(prices).diff().mean(axis=1)
    xx = pd.DataFrame(dict(zscore=z,abs_zscore=z.abs(),spread_change=spread.diff()/std,
        spread_volatility=spread.diff().rolling(20).std()/std,
        y_momentum=ry.rolling(5).sum(),x_momentum=rx.rolling(5).sum(),
        momentum_gap=(ry-rx).rolling(5).sum(),market_volatility=market.rolling(20).std(),
        beta_change=beta.diff(5)), index=prices.index)
    # Fixed signal-time beta for regression target; no future hedge estimates.
    forecast_target=((y.shift(-1)-y)-beta*(x.shift(-1)-x))/std
    direction = np.where(z < 0, 1., -1.)
    ey, ex, ly, lx = y.shift(-1), x.shift(-1), y.shift(-6), x.shift(-6)
    friction = COSTS['slippage']+COSTS['half_spread']
    e_y,e_x = ey*(1+direction*friction),ex*(1-direction*friction)
    l_y,l_x = ly*(1-direction*friction),lx*(1+direction*friction)
    qy,qx = .5/ey,.5/ex  # Fractional equal-dollar label; backtester rounds shares.
    stamp=pd.Series(prices.index,index=prices.index)
    days=(stamp.shift(-6)-stamp.shift(-1)).dt.total_seconds()/86400
    gross=direction*(qy*(l_y-e_y)-qx*(l_x-e_x))
    fee=COSTS['fee']*(qy*(e_y+l_y)+qx*(e_x+l_x))
    borrow=np.where(direction>0,qx*e_x,qy*e_y)*COSTS['borrow']*days/365
    net=gross-fee-borrow
    future_z=(ly-alpha-beta*lx-mean)/std
    labels=pd.Series(0,index=prices.index,dtype=float)
    opportunity=z.abs()>=1.5
    labels.loc[opportunity & ((future_z.abs()>=4) | ((future_z.abs()-z.abs())>.5))]=-1
    labels.loc[opportunity & (net>0) & (future_z.abs()<z.abs())]=1
    labels.loc[net.isna() | future_z.isna()]=np.nan
    # A structural-break class is an observable future-divergence proxy, not proof.
    return xx, labels, forecast_target, net, pd.DataFrame(dict(beta=beta,alpha=alpha,spread=spread,mean=mean,std=std),index=prices.index)
