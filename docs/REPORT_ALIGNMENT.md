# Implementation of your ML-enhanced pairs trading report

The **Daily report** workspace implements the five-module daily workflow in `ML_Pairs_Trading_Project_Report_BW (1).pdf`. It runs alongside the existing five-minute paper terminal. Daily backtest trades have their own history, equity ledger and explanations; they do not credit the terminal account or become paper-model training observations.

## Run it

1. Install `requirements.txt`, then start `python -m app.server` from the repository.
2. Open http://127.0.0.1:8765 and choose **Daily report → Run all five stages**.
3. The selected source is used: local mode generates daily data; connected Upstox mode requests daily historical closes for the 20 configured stocks. Failures never fall back to synthetic prices.
4. Inspect the five stages, select a fitted pair, view model comparisons and research trade history, then download the CSVs, figures and JSON.

The job runs in the background. The terminal remains usable. Results persist under `runtime/daily-report/<source-exchange>/<run-id>/`; restarting the server reloads the last completed report for each account. A failed new run preserves the last completed result. Downloads are restricted to the active account's report.

To run independently, without creating or changing any paper account:

```powershell
.\.venv\Scripts\python.exe scripts/run_daily_report.py --output outputs/my-daily-run
# Your own daily closes: date first, then one column per symbol
.\.venv\Scripts\python.exe scripts/run_daily_report.py --csv data/closes.csv --output outputs/my-csv-run
```

Use a new output directory for each CLI run. Input needs at least 600 aligned daily observations and six stocks. Stock names must start with a letter/digit and contain at most 64 letters/digits, dots, underscores, ampersands or hyphens. Duplicate or invalid dates/symbols and intraday observations are rejected; invalid/missing prices cause entire rows to be dropped, with the count reported. No price is forward-filled. The CSV's provenance is the user's responsibility. Files in `outputs/` and `runtime/` are ignored by Git.

## Report-to-code map

| Report pages | Module | Implementation | Outputs |
|---|---|---|---|
| 8, 11, 17 | 1: preparation and PCA | `ml/report/data.py` | Daily prices, log returns, ADF checks, PCA factors, cumulative variance figure |
| 9, 11, 18 | 2: discovery | `ml/report/discovery.py` | OPTICS cluster map/CSV, all candidate statistics and rejection reasons, verified-pair CSV |
| 9–12, 18–19 | 3: regression and classification | `ml/report/features.py`, `models.py` | Rolling hedge/features, OLS/Ridge/Lasso comparisons, Lasso coefficients, Logistic/SVM/KNN comparisons, SVM score slices |
| 10, 12–13, 19 | 4: ensembles and costs | `ml/report/models.py` | Decision tree/RF/boosting comparisons, RF out-of-bag error, depth/loss figure, standard/weighted confusion matrices |
| 13–16, 19–20 | 5: execution and explanation | `ml/report/backtest.py`, `explain.py`, `pipeline.py` | Trade/equity ledgers, performance tearsheet, cost sensitivity, equity/drawdown figure, actual per-entry SHAP values and plots |

`daily_report.ipynb` provides an executable walkthrough of these modules. `app/report_lab.py` connects the pipeline to the application; `app/static/report.js` displays it. The original `pairs_trading.ipynb` and `scripts/build_notebook.py` remain the separate teaching notebook and its generator.

## Chronology and discovery

The split is chronological: first 30% formation, next 30% training, next 20% validation, final 20% test. Six observations are purged from the end of each supervised partition because a label reaches t+6. Test outcomes never select a pair, regressor, ensemble, feature subset or threshold.

For positive daily closes P:

```text
r[t] = log(P[t] / P[t-1])
```

Formation-period return ADF tests must reject a unit root at p < .05 to include a stock in PCA. Log returns are not assumed stationary merely because they are returns. PCA fits the covariance structure of formation log returns, keeps five components, and describes each stock by its loadings `component * sqrt(eigenvalue)`. Loadings are normalized to unit length for OPTICS. Return transformation and loadings are frozen thereafter.

OPTICS uses `min_samples=2`, `min_cluster_size=2`, `xi=.05`. Noise label −1 is excluded. All within-cluster unordered pairs are tested; this is discovery across the 20 stocks, unlike the terminal's ten predefined candidates. There are 190 possible pairs before clustering, not necessarily 190 tests. Formation OLS estimates:

```text
Y[t] = alpha + beta * X[t] + residual[t]
S[t] = Y[t] - alpha - beta * X[t]
```

Verification requires positive beta, Engle–Granger p-values adjusted by Benjamini–Hochberg FDR < .05, Hurst < .5, and OU half-life between 1 and 252 trading observations. The adjustments cover all tested cluster candidates. They do not establish error control for every aspect of the preceding data-driven selection. These tests provide evidence, not proof of future convergence. Price integration orders are not separately certified; Engle–Granger interpretation retains that econometric assumption.

For Hurst, the implementation applies rescaled-range analysis to **spread increments**, using six log-spaced block sizes from 8 to approximately one third of the formation sample. In each block, divide the range of cumulative demeaned increments by their sample standard deviation. H is the slope of log(mean R/S) against log(block size). Applying R/S directly to spread levels estimates a different object; short-sample H is noisy. This convention is explicit because the report does not fix it.

OU approximation fits `ΔS[t] = intercept + gamma*S[t-1] + error`, sets `phi = 1+gamma`, and reports `half_life = log(2)/(-log(phi))` only for `0 < phi < 1`. Units are trading observations, not calendar days. At most eight verified pairs are modeled per run, ranked by formation FDR only. Rejected candidates remain visible; an empty verified set stays empty.

## Daily features and targets

For each signal time t, rolling Ridge uses the preceding 120 prices, excluding t. With centered moments and standardized X, Ridge alpha=.1 gives:

```text
beta[t] = (cov120(X,Y) / var120(X)) / (1 + .1/120)
alpha[t] = mean120(Y) - beta[t]*mean120(X)
spread[t] = Y[t] - alpha[t] - beta[t]*X[t]
z[t] = (spread[t] - mean(previous 60 spreads)) / std(previous 60 spreads)
forecast_target[t] = (ΔY[t+1] - beta[t]*ΔX[t+1]) / spread_std[t]
```

Standard deviations use sample denominators except the explicit rolling Ridge moments. Other features are absolute Z, normalized spread change, 20-observation spread-change volatility, each stock's five-observation log momentum, their momentum difference, equal-weight universe-return volatility, and five-observation beta change.

The three labels describe future outcomes, not buy/sell directions:

- **+1:** absolute Z at least 1.5, positive modeled net return and a smaller future absolute spread Z.
- **−1:** absolute Z at least 1.5 and future absolute Z at least 4, or an increase in absolute Z greater than .5; the profitable-reversion condition takes precedence.
- **0:** remaining observations.

Labels enter at close t+1 and exit at close t+6. Future Z uses the signal-time hedge and normalization. Labels use fractional equal-dollar legs with the same modeled fees, spread, slippage and calendar-time borrowing as the backtest. Executed positions use integers and may exit earlier, so a label is not an exact forecast of realized execution P&L. −1 is a divergence proxy, not proof of a permanent structural break. Long/Short comes from Z's sign.

## Models and selection

Every scaler is fitted on training data only. MinMax values outside [0,1] are possible on later observations; future extrema are not used to rescale training.

| Family | Compared settings |
|---|---|
| OLS | LinearRegression |
| Ridge | alpha .1, 1 |
| Lasso | alpha .01 |
| Logistic | C .1, 1 |
| Linear SVM | C .1, 1 |
| RBF SVM | C .1, 1; gamma=scale |
| KNN | 5, 11 neighbors |
| Decision tree | log-loss splitting, depths 2/4/6, min leaf 8 |
| Random forest | **120 trees**, log-loss splitting, depths 2/4/6, min leaf 8, OOB enabled |
| Gradient boosting | **80 stages**, depths 2/3, min leaf 8 |

Regressor selection uses a chronological 70/30 split inside training, with a two-row gap. Three chronological folds with six-row gaps produce out-of-fold forecasts for classifier training. Early rows without such forecasts are excluded. The chosen regressor fits all usable training rows for validation/test prediction. Training-only Lasso selects nonzero original features; if all vanish, the report explicitly records retention of all original features. Forecast change is added as a classifier feature. Lasso pruning and regressor selection use the overall training partition, so the internal forecasts are not a fully nested feature-selection experiment.

Non-entry labels −1 and 0 receive weight 3; +1 receives weight 1. KNN implements these integer weights by repeating training examples. Weighted diagnostic log-loss is `mean(-weight[y] * log(p[y]))`. Class weighting is an approximation to asymmetric decision cost, not a custom economic loss function for every estimator. Threshold selection explicitly minimizes **3×false positives + false negatives**.

SVM probability estimates use chronological sigmoid calibration on the last 30% of its training input, with a six-row gap from the fitting prefix. No shuffled `SVC(probability=True)` calibration is used. Other classifier scores are uncalibrated estimates. A class with insufficient chronological samples is reported as unavailable.

Separate long/short validation thresholds are searched from .50 to .95 in .05 steps, plus 1.01 to disable entries. A side with fewer than five validation opportunities, or no positive labels, is disabled. Entry Z magnitude must be in [2,4). Among RF and boosting candidates, validation 3FP+FN chooses the ensemble, with weighted log-loss breaking ties. The test metrics are displayed only. Standard-versus-weighted comparisons use the same chosen architecture. OOB error and depth curves are diagnostics; temporal holdout remains the evaluation boundary.

A separate Logistic model predicts whether universe volatility five observations ahead exceeds the training median. This reduces the allocation by half when predicted high. It does not remove market risk. No order-flow feature is invented because the available input is daily closes.

The daily RF's 120 trees are separate from the **180-tree five-minute terminal model**, **120-tree completed-paper-trade model**, and **160-tree legacy daily model**. Tree counts mean ensemble members, not companies or trades. See [the terminal calculations](CALCULATIONS.md).

## Execution, P&L and evaluation

Daily research starts with INR 10,00,000. A signal at t executes at the next available daily close t+1; it never fills at the signal close. Each pair targets 15% of current equity gross, halved in the predicted high-volatility regime, with aggregate entry capacity limited to current equity. Half the target is allocated to each leg, rounded down to integer shares. Price movement can subsequently increase exposure. Dollar imbalance from rounding is recorded.

This resolves a conflict in the report: **dynamic beta describes spread/features; equal-dollar targets determine shares**. Equal dollars generally do not produce a beta-neutral hedge. Use a separate strategy experiment if you want hedge-ratio share sizing.

Illustrative per-fill costs are commission 3 bps, half-spread 2 bps and adverse slippage 5 bps. Borrow is 10% annually on short entry notional using calendar elapsed days/365. For signed quantity q:

```text
entry_fill = entry_close * (1 + sign(q)*.0007)
exit_fill  = exit_close  * (1 - sign(q)*.0007)
fees       = .0003 * sum(abs(q)*(entry_fill + exit_fill))
net_P&L    = sum(q*(exit_fill - entry_fill)) - fees - borrow
equity     = balance + marked unrealized_P&L - accrued borrow
```

Entry fees debit balance immediately; exit P&L credits it net of exit fees and borrowing. Realized trade net P&L still includes entry fees. Open positions are marked at raw closes; hypothetical exit costs are recognized on execution. Exits are scheduled for next close after reversion, breakdown proxy, spread stop, missing signal or the five-session holding limit. The known final test date liquidates remaining positions with costs. No fictitious next-open OHLC price is created from close-only data.

Daily returns use successive marked equity values, including a capital baseline. Sharpe is `sqrt(252)*mean(return)/sample_std(return)` with zero risk-free rate. Sortino uses `sqrt(mean(min(return,0)^2))` in the denominator. Maximum drawdown includes the initial-capital peak. ROI is final equity/capital−1. Undefined ratios are null, shown as a dash. Break-even all-in transaction bps is `(raw_quote_P&L - borrow)/raw_quote_turnover*10000`; a negative result means the sampled trades lose even before transaction costs. Sensitivity reruns the same frozen signals at commissions 0/1/3/5/10 bps; sizing can change with equity. The benchmark is an equal-weight buy-and-hold basket of the input universe with no benchmark trading costs.

## Explanations and limitations

SHAP uses the actual selected model pipeline and the +1 output score. Permutation SHAP uses up to 30 evenly sampled training-background rows and a three-permutation evaluation budget with seed 42. Every executed entry has exported feature values, base score and contributions, checked against the actual prediction: `base + sum(contributions) = score`. Plots use those values. Approximate SHAP explains model output, not causes of returns or guaranteed profit. The SVM figure is a two-feature score slice with other features fixed at training medians; it is not a complete multidimensional decision boundary.

Local daily prices and five-minute local candles are synthetic. Upstox historical closes are provider data, currently unadjusted; corporate actions can invalidate raw-price relationships. No survivorship-free exchange universe, real borrow inventory, order book, partial fills, taxes, broker-specific margin or official settlement simulation is claimed. Research with these costs is not evidence of executable real-money profit. Live Upstox access requires the user's valid authorization; all application executions remain paper money.

## Make your own daily model

1. Add causal features in `features.engineer`; do not place forward returns or labels in `xx`.
2. Add an estimator to `models.train_pair`'s candidate list. It must support class probabilities for labels −1/0/+1. `probabilities()` maps available classes explicitly.
3. Extend `fit_classifier` if your estimator lacks sample weights or needs special chronological calibration. Keep training-only preprocessing inside its pipeline.
4. To allow that estimator to become the execution model, deliberately add its name to the `ensembles` selection list. Merely adding a comparison model does not activate it.
5. Keep held-out data out of selection and run the tests below. If labels, horizon or execution rules change, update the purge, cost model, history definitions and documentation together.
6. Rerun Daily report to produce a new result. Daily report models are not automatically promoted to five-minute automation. Follow [CUSTOM_MODELS.md](CUSTOM_MODELS.md) to modify the terminal classifier and paper-outcome model.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_daily_report_browser.py
```

Tests verify future-price isolation, held-out model-selection invariance, delayed fills, independently calculated accounting, source-scoped artifacts, strict JSON export and actual SHAP additivity. Browser checks exercise background execution, figures/downloads, mobile layout and source switching without changing the user's paper account.

Implementation references: [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html), [chronological splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [classifier calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), [permutation SHAP](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html).
