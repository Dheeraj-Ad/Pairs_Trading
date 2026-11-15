"""Stage 2: OPTICS candidates, cointegration/FDR, Hurst and OU half-life."""
from itertools import combinations
import numpy as np
from sklearn.cluster import OPTICS
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


def hurst_rs(spread):
    # R/S of increments estimates scaling of the spread path. Applying R/S to
    # spread levels instead estimates a different object. See REPORT_ALIGNMENT.
    increments = np.diff(np.asarray(spread, dtype=float))
    sizes = np.unique(np.geomspace(8, max(16, len(increments)//3), 6).astype(int))
    scales, ranges = [], []
    for size in sizes:
        ratios = []
        for start in range(0, len(increments)-size+1, size):
            block = increments[start:start+size]
            sd = block.std(ddof=1)
            if sd > 1e-12:
                ratio = np.ptp(np.cumsum(block-block.mean()))/sd
                if ratio > 0: ratios.append(ratio)
        if ratios:
            scales.append(size); ranges.append(float(np.mean(ratios)))
    if len(scales) < 3: raise ValueError('Insufficient variation for Hurst estimation.')
    return float(np.polyfit(np.log(scales), np.log(ranges), 1)[0])


def half_life(spread):
    values = np.asarray(spread, dtype=float)
    gamma, _ = np.polyfit(values[:-1], np.diff(values), 1)
    phi = 1+gamma
    return float(np.log(2)/-np.log(phi)) if 0 < phi < 1 else None


def discover(prices, data):
    coords = np.asarray(data['coordinates'])
    optics = OPTICS(min_samples=2, min_cluster_size=2, xi=.05, cluster_method='xi').fit(coords)
    clusters = [dict(symbol=symbol, cluster=int(label), pc1=float(point[0]), pc2=float(point[1]))
                for symbol, label, point in zip(data['symbols'], optics.labels_, coords)]
    formation = prices.iloc[:data['formation_rows']]
    rows = []
    for label in sorted(set(optics.labels_) - {-1}):
        members = [r['symbol'] for r in clusters if r['cluster'] == label]
        for y, x in combinations(sorted(members), 2):
            row = dict(id=y+'__'+x, y=y, x=x, cluster=int(label), p=1., alpha=None,
                       beta=None, hurst=None, half_life=None, reasons=[])
            try:
                beta, alpha = np.polyfit(formation[x], formation[y], 1)
                spread = formation[y]-alpha-beta*formation[x]
                row.update(alpha=float(alpha), beta=float(beta), p=float(coint(formation[y], formation[x])[1]),
                           hurst=hurst_rs(spread), half_life=half_life(spread))
                if not all(np.isfinite(row[k]) for k in ('alpha','beta','p','hurst')):
                    raise ValueError('Non-finite statistical estimate.')
                if beta <= 0: row['reasons'].append('Non-positive hedge ratio')
                if not row['hurst'] < .5: row['reasons'].append('Hurst is not below 0.5')
                if row['half_life'] is None or not 1 <= row['half_life'] <= 252:
                    row['reasons'].append('OU half-life outside 1–252 trading days')
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(p=1., reasons=['Statistics unavailable: '+str(exc)])
            rows.append(row)
    if rows:
        adjusted = multipletests([r['p'] for r in rows], method='fdr_bh')[1]
        for row, q in zip(rows, adjusted):
            row['fdr'] = float(q)
            if q >= .05: row['reasons'].append('Cointegration FDR is not below 0.05')
            row['verified'] = not row['reasons']
    return clusters, rows
