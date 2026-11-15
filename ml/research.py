"""Chronological pair research. Never sends orders or mutates an account."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from statsmodels.tsa.stattools import coint

FEATURES = ["zscore", "abs_zscore", "spread_change", "spread_volatility"]


def validate_history(frame):
    frame = frame.copy().sort_index()
    if len(frame) < 400 or list(frame.columns) != ["y", "x"]:
        raise ValueError("At least 400 aligned daily closes with y and x columns are required.")
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("History has duplicate or invalid timestamps.")
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ValueError("History must contain finite positive prices without gaps.")
    return frame


def baseline(frame):
    frame = validate_history(frame)
    n = int(len(frame) * .3)
    formation = frame.iloc[:n]
    beta, alpha = np.polyfit(formation.x, formation.y, 1)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("This pair has no positive hedge ratio in the formation period.")
    spread = frame.y - alpha - beta * frame.x
    prior = spread.shift(1)
    z = (spread - prior.rolling(60).mean()) / prior.rolling(60).std()
    p = float(coint(formation.y, formation.x)[1])
    if not np.isfinite(p) or spread.iloc[-60:].std() <= 1e-9:
        raise ValueError("The pair has insufficient spread variation.")
    return {"alpha": float(alpha), "beta": float(beta), "coint_p": p,
            "correlation": float(formation.pct_change().dropna().corr().iloc[0, 1]),
            "mean": float(spread.iloc[-60:].mean()), "std": float(spread.iloc[-60:].std()),
            "formation_rows": n, "formation_end": str(formation.index[-1]),
            "history_end": str(frame.index[-1]), "rows": len(frame),
            "chart": [{"date": str(frame.index[i]), "z": float(z.iloc[i])}
                      for i in range(max(60, len(frame)-120), len(frame)) if np.isfinite(z.iloc[i])]}


def train(frame):
    """Fit on 30–60%, select threshold on 60–80%, report untouched 80–100%.

    Label is absolute-spread contraction after five sessions, not trade profit.
    Labels crossing a partition boundary are purged before fitting/tuning.
    """
    frame = validate_history(frame)
    info = baseline(frame)
    s = frame.y - info["alpha"] - info["beta"] * frame.x
    mean, std = s.shift(1).rolling(60).mean(), s.shift(1).rolling(60).std()
    z = (s - mean) / std
    features = pd.DataFrame({"zscore": z, "abs_zscore": z.abs(),
                             "spread_change": s.diff() / std,
                             "spread_volatility": s.diff().rolling(20).std() / std})
    horizon, n = 5, len(frame)
    target = ((s.shift(-horizon) - mean).abs() < (s - mean).abs()).astype(int)
    a, b, c = int(n*.3), int(n*.6), int(n*.8)
    def part(lo, hi):
        xx, yy = features.iloc[lo:hi-horizon], target.iloc[lo:hi-horizon]
        valid = np.isfinite(xx).all(axis=1)
        return xx.loc[valid], yy.loc[valid]
    xt, yt = part(a, b)
    xv, yv = part(b, c)
    xe, ye = part(c, n)
    if min(len(xt), len(xv), len(xe)) < 30 or yt.nunique() < 2:
        raise ValueError("Insufficient training samples or only one target class.")
    model = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=8,
                                   class_weight="balanced", random_state=42, n_jobs=1)
    model.fit(xt, yt)
    vp = model.predict_proba(xv)[:, 1]
    thresholds = [.4, .5, .6, .7]
    threshold = max(thresholds, key=lambda t: accuracy_score(yv, vp >= t))
    ep = model.predict_proba(xe)[:, 1]
    report = {"algorithm": "Random forest", "target": "Spread contracts over the next 5 sessions",
              "threshold": threshold, "train_rows": len(xt), "validation_rows": len(xv),
              "test_rows": len(xe), "purge_sessions": horizon,
              "accuracy": float(accuracy_score(ye, ep >= threshold)),
              "balanced_accuracy": float(balanced_accuracy_score(ye, ep >= threshold)),
              "auc": float(roc_auc_score(ye, ep)) if ye.nunique() == 2 else None,
              "majority_baseline": float(max(ye.mean(), 1-ye.mean())),
              "feature_importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
              "splits": {"formation_end": str(frame.index[a-1]), "train_end": str(frame.index[b-1]),
                         "validation_end": str(frame.index[c-1]), "test_end": str(frame.index[-1])},
              "baseline": info}
    return model, report
