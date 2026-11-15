"""Stage 1: aligned daily prices, return stationarity and formation-only PCA."""
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


def prepare(prices):
    frame = prices.copy()
    if any(not isinstance(symbol, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.&-]{0,63}', symbol)
           for symbol in frame.columns):
        raise ValueError('Stock symbols must be 1–64 letters/digits, dots, underscores, ampersands or hyphens, starting with a letter/digit.')
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.isna().any():
        raise ValueError('Daily observations must have valid dates.')
    frame = frame.sort_index()
    if frame.index.has_duplicates or frame.columns.duplicated().any():
        raise ValueError('Duplicate dates or symbols in report data.')
    frame = frame.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    frame = frame.where(frame > 0)
    removed = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()  # No invented prices or forward fills.
    if len(frame) < 600 or len(frame.columns) < 6:
        raise ValueError('Daily report needs at least 600 aligned daily observations and six stocks.')
    if frame.index.to_series().diff().dropna().dt.total_seconds().min() < 86400:
        raise ValueError('The report workflow accepts daily observations, not intraday bars.')
    formation_end = int(len(frame)*.30)
    returns = np.log(frame).diff().dropna()
    formation = returns.loc[returns.index < frame.index[formation_end]]
    stationarity = []
    for symbol in frame:
        values = formation[symbol]
        if values.std() <= 1e-10:
            stationarity.append(dict(symbol=symbol, adf_p=None, eligible=False, reason='Constant returns'))
            continue
        p = float(adfuller(values, autolag='AIC')[1])
        stationarity.append(dict(symbol=symbol, adf_p=p, eligible=p < .05,
                                 reason='' if p < .05 else 'Return ADF did not reject a unit root'))
    kept = [r['symbol'] for r in stationarity if r['eligible']]
    if len(kept) < 6:
        raise ValueError('Fewer than six stocks passed formation-period return stationarity checks.')
    pca = PCA(n_components=5, svd_solver='full').fit(formation[kept])
    # PCA of the covariance of log returns: stocks are described by factor loadings.
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    norm = np.linalg.norm(loadings, axis=1, keepdims=True)
    coordinates = loadings / np.maximum(norm, 1e-12)
    factors = pd.DataFrame(pca.transform(returns[kept]), index=returns.index,
                           columns=[f'PC{i+1}' for i in range(5)])
    return frame, returns, factors, dict(rows=len(frame), dropped_rows=removed,
        formation_rows=formation_end, formation_end=str(frame.index[formation_end-1]),
        stationarity=stationarity, symbols=kept, coordinates=coordinates.tolist(),
        explained_variance=pca.explained_variance_ratio_.tolist(),
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_).tolist())
