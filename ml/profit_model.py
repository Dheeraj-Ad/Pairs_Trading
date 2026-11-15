"""Five-minute classifier: positive two-leg P&L after modeled costs.

This is a research score, not a calibrated probability or profit guarantee.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from ml.research import baseline, FEATURES

HORIZON = 5
FEE, SLIPPAGE, BORROW = .0003, .0005, .10


def features(frame, info):
    spread = frame.y-info['alpha']-info['beta']*frame.x
    mean = spread.shift(1).rolling(60).mean()
    std = spread.shift(1).rolling(60).std().replace(0, np.nan)
    z = (spread-mean)/std
    return pd.DataFrame({'zscore':z,'abs_zscore':z.abs(), 'spread_change':spread.diff()/std,
                         'spread_volatility':spread.diff().rolling(20).std()/std})


def net_targets(frame, info, xx):
    """Signal at close t, fill next open, exit open t+6; no future execution at t."""
    side = np.where(xx.zscore < 0, 1., -1.)
    ey, ex = frame.y_open.shift(-1), frame.x_open.shift(-1)
    ly, lx = frame.y_open.shift(-6), frame.x_open.shift(-6)
    beta = info['beta']
    entry_y, entry_x = ey*(1+side*SLIPPAGE), ex*(1-side*SLIPPAGE)
    exit_y, exit_x = ly*(1-side*SLIPPAGE), lx*(1+side*SLIPPAGE)
    gross = ey+beta*ex
    pnl = side*((exit_y-entry_y)-beta*(exit_x-entry_x))
    costs = FEE*(entry_y+beta*entry_x+exit_y+beta*exit_x)
    short = np.where(side > 0, beta*entry_x, entry_y)
    net = (pnl-costs-short*BORROW*(25/(365*24*60)))/gross
    # Never label across overnight gaps or missing five-minute intervals.
    stamps = pd.Series(frame.index, index=frame.index)
    continuous = (stamps.shift(-6)-stamps).dt.total_seconds() == 1800
    return net.where(continuous)


def train_profit(frame):
    info = baseline(frame[['y','x']])
    xx = features(frame, info)
    net = net_targets(frame, info, xx)
    n=len(frame)
    a,b,c=int(n*.3),int(n*.6),int(n*.8)
    def part(lo,hi):
        ids=np.arange(lo,hi-6)
        keep=np.isfinite(xx.iloc[ids]).all(axis=1) & net.iloc[ids].notna()
        ids=ids[keep.to_numpy()]
        return xx.iloc[ids], (net.iloc[ids]>0).astype(int), net.iloc[ids]
    xt,yt,_=part(a,b)
    xv,yv,nv=part(b,c)
    xe,ye,ne=part(c,n)
    if min(len(xt),len(xv),len(xe))<30 or yt.nunique()<2:
        raise ValueError('Need more completed, continuous candles and both profitable/unprofitable labels.')
    model=RandomForestClassifier(n_estimators=180,max_depth=5,min_samples_leaf=10,random_state=42,n_jobs=1)
    model.fit(xt,yt)
    vp=model.predict_proba(xv)[:,list(model.classes_).index(1)]
    choices=[]
    for threshold in [.55,.60,.65,.70,.75,.80]:
        accepted=(vp>=threshold)&(xv.abs_zscore>=1.5)&(xv.abs_zscore<4)
        # Thin samples never qualify even if their average return is high.
        avg=float(nv[accepted].mean()) if accepted.any() else 0.
        choices.append((int(accepted.sum())>=8 and avg>0, avg, threshold, int(accepted.sum())))
    passing=[row for row in choices if row[0]]
    choice=max(passing,key=lambda r:r[1]) if passing else choices[2]
    threshold=choice[2]
    ep=model.predict_proba(xe)[:,list(model.classes_).index(1)]
    report=dict(algorithm='Random forest · net-profit target',target='Positive net return over five 5-minute bars',
        timeframe='5m', version=2, threshold=threshold, train_rows=len(xt),validation_rows=len(xv),test_rows=len(xe),
        purge_sessions=6, accuracy=float(accuracy_score(ye,ep>=threshold)),
        balanced_accuracy=float(balanced_accuracy_score(ye,ep>=threshold)),
        auc=float(roc_auc_score(ye,ep)) if ye.nunique()==2 else None,
        majority_baseline=float(max(ye.mean(),1-ye.mean())),
        validation_pass=bool(choice[0]), validation_signals=choice[3],validation_mean_return=choice[1],
        validation_auc=float(roc_auc_score(yv,vp)) if yv.nunique()==2 else None,
        feature_importance=dict(zip(FEATURES,map(float,model.feature_importances_))),baseline=info,
        splits={'formation_end':str(frame.index[a-1]),'train_end':str(frame.index[b-1]),
                'validation_end':str(frame.index[c-1]),'test_end':str(frame.index[-1])})
    return model,report


def predict(model, frame, info):
    row=features(frame,info).iloc[[-1]]
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError('Cannot score a candle with invalid features.')
    score=float(model.predict_proba(row)[0,list(model.classes_).index(1)])
    return dict(score=score,zscore=float(row.zscore.iloc[0]),bar_time=int(frame.index[-1].timestamp()),
                side='long' if row.zscore.iloc[0]<0 else 'short')
