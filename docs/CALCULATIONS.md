# Implemented calculations and data provenance

For PCA, OPTICS, Hurst, half-life, daily regression/ensembles, daily backtesting and SHAP, see [the daily report calculations](REPORT_ALIGNMENT.md). This document describes the separate five-minute terminal.

This documents the current browser terminal, with source links for auditing. It describes calculations the code performs, rather than promising future profitability. Return to the [README](../README.md) for usage and the [custom-model guide](CUSTOM_MODELS.md) for changes.

## 1. Inputs and notation

- `Y_t`, `X_t`: aligned prices of the first and second stocks at completed five-minute bar `t`.
- `O_Y,t`, `O_X,t`: their opening prices. Bar timestamps refer to the start of the bar.
- `alpha`, `beta`: fitted intercept and hedge ratio, using price levels, not log prices.
- `s_t`: residual spread; `z_t`: standardized spread.
- `d=+1`: long spread (buy Y, sell X); `d=-1`: short spread (sell Y, buy X).
- `f=0.0003`: fee rate per fill, or 3 basis points.
- `l=0.0005`: adverse slippage per fill, or 5 basis points.
- `b=0.10`: annualized illustrative short borrowing rate.

Inputs must be finite, positive and aligned. Missing legs are not forward-filled. A bar enters model calculations only if `bar_start + 300 seconds <= market_clock`. OHLCV validation requires `low <= min(open, close) <= max(open, close) <= high`, positive prices and nonnegative volume. See [market.py](../app/market.py).

In local mode the market clock is simulated. In Upstox mode it is wall-clock time and quotes/candles come from the provider. Calling a variable `price`, or displaying a real company symbol, does not make a generated price an exchange observation.

## 2. Hedge ratio and intercept

Let the formation set be the first `floor(0.30*N)` observations. The code uses `numpy.polyfit(X, Y, 1)`, equivalent to ordinary least squares with an intercept:

```text
beta  = sum((X_i - mean(X)) * (Y_i - mean(Y))) / sum((X_i - mean(X))^2)
alpha = mean(Y) - beta * mean(X)
s_t   = Y_t - alpha - beta * X_t
```

The fitted beta must be finite and positive. Alpha and beta are fixed for that fitted model; they are not re-estimated every candle. A hedge-ratio position is not necessarily dollar-neutral or market-beta-neutral. The intercept changes spread measurement but is not an additional traded instrument. Source: [research.baseline](../ml/research.py).

## 3. Correlation, cointegration and multiple testing

Formation-period simple returns are:

```text
rY_t = Y_t / Y_(t-1) - 1
rX_t = X_t / X_(t-1) - 1
rho  = sum((rY - mean(rY)) * (rX - mean(rX)))
       / sqrt(sum((rY - mean(rY))^2) * sum((rX - mean(rX))^2))
```

The pair needs `rho >= 0.40`. This checks return co-movement; correlation alone does not establish that the price spread is stationary.

`statsmodels.tsa.stattools.coint(formation.Y, formation.X)` performs the augmented Engle–Granger test with its default constant and automatic lag selection. It fits a price relationship and tests the residual for a unit root. Its null hypothesis is no cointegration; a small p-value provides evidence against that null under the test's assumptions. The code delegates the test statistic and approximate p-value to statsmodels; it does not hand-assign p-values. The I(1) price-series assumption is not separately checked by this app. See the [official coint reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).

The terminal applies Benjamini–Hochberg FDR across all configured candidate p-values, including `p=1` for candidates whose scan failed. For sorted p-values `p_(1) <= ... <= p_(m)`:

```text
q_(i) = min(1, min over j>=i of (m/j) * p_(j))
```

Adjusted values are restored to original pair order. A pair needs `q < 0.05`. The implementation is `multipletests(..., method='fdr_bh')`; its validity depends on assumptions about the tests' dependence, and it is not a guarantee of future stability. See [statsmodels multiple testing](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## 4. Rolling spread statistics and features

The signal excludes the current spread from its reference window:

```text
mu_t    = (s_(t-60) + ... + s_(t-1)) / 60
sigma_t = sqrt(sum over i=t-60..t-1 of (s_i - mu_t)^2 / 59)
z_t     = (s_t - mu_t) / sigma_t
delta_s_t = s_t - s_(t-1)
```

Pandas rolling standard deviation uses the sample denominator (`ddof=1`). Zero standard deviations become missing values rather than producing a valid signal. The four classifier inputs, in order, are:

| Feature | Formula |
| --- | --- |
| `zscore` | `z_t` |
| `abs_zscore` | `abs(z_t)` |
| `spread_change` | `(s_t - s_(t-1)) / sigma_t` |
| `spread_volatility` | Sample standard deviation of the latest 20 spread changes, including `delta_s_t`, divided by `sigma_t` |

The fourth feature includes the current completed bar's change, which is known at scoring time. It uses no future changes. Source: [profit_model.features](../ml/profit_model.py).

There are two related references in execution: the completed-bar ML Z-score above uses the previous 60 spreads; the live-quote entry recheck uses the latest 60 completed spreads, including the most recently completed one. That second mean and standard deviation are frozen in the position's decision record for spread-based exits. They need not produce identical Z-scores. The chart computes Z-scores from its returned candle window; the earliest 60 displayed input bars establish its rolling reference.

## 5. What the 180-tree model is trained to predict

For every eligible historical signal row `t`:

```text
d = +1 if z_t < 0 else -1
Entry raw prices: ey = O_Y,t+1, ex = O_X,t+1
Exit raw prices:  ly = O_Y,t+6, lx = O_X,t+6

EY = ey * (1 + d*l)       EX = ex * (1 - d*l)
LY = ly * (1 - d*l)       LX = lx * (1 + d*l)

gross_price_pnl = d * ((LY - EY) - beta * (LX - EX))
fees           = f * (EY + beta*EX + LY + beta*LX)
short_notional = beta*EX if d=+1 else EY
borrow         = short_notional * b * 25 / (365*24*60)
net_return     = (gross_price_pnl - fees - borrow) / (ey + beta*ex)
target         = 1 if net_return > 0 else 0
```

The label uses one unit of Y and a fractional beta units of X; actual paper orders round to integer shares. Entry at open `t+1` and exit at open `t+6` are 25 minutes apart. The code requires the start timestamps of `t` and `t+6` to be exactly 30 minutes apart, excluding overnight or missing-bar labels. Future prices are used only to create historical training outcomes, never as current input features.

Label costs are explicit assumptions. Label exits are fixed-horizon; actual automatic positions can close earlier under risk rules. Consequently this label is an approximation to the actual exit policy. Source: [profit_model.net_targets](../ml/profit_model.py).

## 6. Training, score and qualification

The price model splits rows chronologically:

```text
0%-30%: formation (hedge ratio and statistical screen)
30%-60%: train classifier
60%-80%: select score threshold on validation data
80%-100%: report test metrics
```

Six final label rows are excluded from each train/validation/test block. Non-finite feature rows and unavailable labels are discarded. Each fitted partition must retain at least 30 examples, and training must contain both classes.

The forest has 180 trees, depth at most 5, leaf size at least 10, `random_state=42`, and one training job. Default bootstrap sampling and randomized split features apply. For positive-class tree estimates `p_k(x)`:

```text
forest_score(x) = sum(k=1..180, p_k(x)) / 180
```

This averages tree probability estimates; it is not simply the percentage of trees voting yes. The estimates are not calibrated against future exchange outcomes. See [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Validation tries thresholds `[0.55, 0.60, 0.65, 0.70, 0.75, 0.80]`. For each threshold it selects rows with `score >= threshold` and `1.5 <= abs(z) < 4`. Passing requires at least eight such rows and their mean `net_return > 0`. Among passing choices it selects the highest mean net return. With no passing choice it records the 0.65 result but `validation_pass=False`; that pair is rejected.

Full qualification requires: positive beta, correlation at least 0.40, FDR below 0.05, passing validation signals, and validation AUC above 0.50. The test set does not select the threshold or qualify pairs. A new scan can change qualification because its input window and splits have changed.

Metrics:

```text
accuracy          = correct class predictions / total predictions
balanced_accuracy = (true_positive_rate + true_negative_rate) / 2  [binary two-class case]
majority_baseline = max(fraction positive, fraction non-positive)
AUC               = P(score of a positive > score of a negative)
                    + 0.5 * P(tied scores), over positive-negative pairs
```

Test class predictions use the validation-selected threshold. AUC is absent when an evaluation set has only one class. Feature importances in the current forests measure normalized training impurity reduction, not causal effects or guarantees of economic usefulness. Overlapping 25-minute labels are dependent; the validation mean is not a portfolio backtest, and tuning after looking at test results invalidates its holdout interpretation.

## 7. Actual entry sizing and paper fills

Automatic entry first needs the user's configured Z band (`2 <= abs(z) < 4` by default), `score >= max(user_score, validation_threshold)`, a qualified pair, sufficient available equity, fresh synchronized quotes, and the correct entry session. If enabled, the paper-outcome filter also requires its score to be at least 0.50. A rejected or unavailable score cannot force an entry.

Let `B` be the gross budget, and `PY`, `PX` the current quote prices:

```text
u   = B / ((PY + beta*PX) * (1+l))
nY  = floor(u)
nX  = floor(beta*u)
qY  = d*nY
qX  = -d*nX
```

Both `nY` and `nX` must be at least one. Each signed quantity `q_i` is positive for a long and negative for a short:

```text
entry_fill_i = quote_i * (1 + sign(q_i)*l)
entry_fee    = f * sum(abs(q_i) * entry_fill_i)
```

The account deducts the entry fee immediately. Both legs are stored in one SQLite transaction, not sent to an exchange. A permanent request ID makes retries idempotent. Automatic IDs are derived from account, pair and signal-bar time, preventing repeated entry on the same signal even after closing.

Entry budget must be no more than 50% of marked equity, and:

```text
B * (1 + f + 2*l) <= available_capacity
available_capacity = max(0, equity - current_gross_exposure)
```

There is one open position per pair. The gross cap is checked at entry; prices can subsequently move exposure beyond the initial cap. Source: [paper.py](../app/paper.py).

## 8. Mark-to-market, exits, borrowing and net P&L

Let elapsed holding time in days be `D = max(0, (now - opened)/86400 seconds)`:

```text
short_entry_notional = sum over q_i<0 of abs(q_i)*entry_fill_i
borrow              = short_entry_notional * 0.10 * D / 365
unrealized          = sum(q_i * (mark_i - entry_fill_i)) - borrow
current_gross       = sum(abs(q_i) * mark_i)
open_net            = unrealized - entry_fee
```

Missing quotes are marked at entry fill and flagged non-fresh; stale quotes may remain visible as stale estimates. Non-fresh marks prevent relevant fills. Future exit costs are not yet deducted from the displayed open P&L.

On closing:

```text
exit_fill_i = quote_i * (1 - sign(q_i)*l)
exit_fee    = f * sum(abs(q_i)*exit_fill_i)
price_pnl   = sum(q_i * (exit_fill_i - entry_fill_i))
net_pnl     = price_pnl - entry_fee - exit_fee - borrow
```

Fees are charged on both legs at entry and exit. Slippage is already inside fill prices; subtracting it again from `net_pnl` would double-count it. Borrow accrues for elapsed calendar time, including overnight/weekends, using the simulation clock in local mode. Manual positions are not automatically squared off by the automatic-trader risk loop and can accrue overnight borrow.

## 9. Risk exit calculations

The automatic trader estimates a position's after-exit return before deciding whether to close:

```text
estimated_net = unrealized - entry_fee - current_gross*(0.0003+0.0005)
entry_gross   = sum(abs(q_i)*entry_fill_i)
risk_return   = estimated_net / entry_gross
```

Default stop loss: `risk_return <= -0.005`. Default profit target: `risk_return >= 0.01`. The closing-cost estimate is approximate; actual close P&L uses the fill formula above.

Frozen-reference spread checks use:

```text
z_live = (quoteY - alpha - beta*quoteX - frozen_mean) / frozen_std
```

Exit on `abs(z_live) >= 4`; or on reversion (`z_live >= -0.5` for a long spread, `z_live <= +0.5` for a short spread). Other exits are holding time at least 25 minutes by default, session square-off at/after 15:20 IST or a changed session date, and daily loss halt. The code checks daily halt, session, stop loss, profit target, holding limit, then spread exits in that order.

```text
daily_pnl = current_account_equity - equity_at_first_automation_check_that_day
halt if daily_pnl <= -configured_daily_loss  [default INR 10,000]
```

The daily calculation covers the active account, including manual P&L, but automatic risk exits manage automatic positions. The anchor/halt is persisted. Entries are allowed on weekdays from 09:20 inclusive to 14:55 exclusive. Both quote ages must be between -5 and +60 seconds; leg timestamps must differ by at most 15 seconds. A completed-candle entry signal must be no more than 90 seconds old. Upstox price models expire for entry after 24 hours; local models stay frozen until rescan. Stops cannot guarantee a fill during an outage. Source: [terminal.manage](../app/terminal.py).

## 10. Account and section totals

The default capital is INR 1,000,000. The balance is a simulator capital/P&L balance, not a broker cash settlement ledger; opening does not exchange full long/short notionals in it.

```text
balance     = initial_capital + sum(closed net_pnl) - sum(open entry_fees)
equity      = balance + sum(open unrealized)
overall_pnl = equity - initial_capital
            = sum(closed net_pnl) + sum(open_net)
```

The method/pair breakdown applies the same sums to each group. Thus manual total plus automatic total equals overall P&L, and the pair totals also reconcile. Overall means the selected account, not all local and Upstox accounts combined. Win counts use `net_pnl > 0`, losses use `< 0`, flat results use `= 0`. Trade exports contain positions; transaction exports contain individual leg fills, so a closed two-leg trade normally contributes four fill rows. Source: [reporting.py](../app/reporting.py).

## 11. Worked execution example

Illustrative quotes, not real stock observations: `PY=100`, `PX=50`, `beta=2`, budget `B=10,000`, long spread. Close one day later at `Y=110`, `X=48`:

```text
u = 10000 / ((100 + 2*50)*1.0005) = 49.975012...
qY = +49; qX = -99
Entry fills: Y=100.05; X=49.975
Entry fee = 0.0003 * (49*100.05 + 99*49.975) = 2.9549925

Exit fills: Y=109.945; X=48.024
Price P&L = 49*(109.945-100.05) - 99*(48.024-49.975) = 678.004
Exit fee = 0.0003 * (49*109.945 + 99*48.024) = 3.0425043
Borrow = (99*49.975)*0.10/365 = 1.3554863...
Net P&L = 678.004 - 2.9549925 - 3.0425043 - 1.3554863... = INR 670.6510169...
```

The quantity of X is `floor(beta*u)`, not `beta*floor(u)`, explaining the 49/-99 rounding.

## 12. Learning from closed paper trades

The second forest uses the same four features saved at entry and the actual simulator outcome:

```text
target = 1 if closed_trade.net_pnl > 0 else 0
```

Profit, exit price and exit reason are not input features. Training pools manual and automatic trades in the active account only; local and Upstox examples are not pooled. Missing entry features exclude legacy trades. A flat P&L is a non-profitable ML label, even though the P&L table distinguishes flat from a loss.

At least 40 eligible examples are required. Sort by entry time; use 60% training / 20% validation / 20% test. Remove training rows whose closing time is at or after the first validation entry, and validation rows closing at or after the first test entry. Require at least 12 remaining training rows, six validation rows, six test rows, and both training classes.

The forest uses 120 trees, maximum depth 4, minimum leaf size 4 and seed 19. Evaluation uses a fixed score threshold 0.50. Validation AUC and balanced accuracy must both exceed 0.50 to permit its optional automatic-entry filter. Forty trades is only a software minimum, not statistical proof of reliability. Training is explicit; saved outcomes do not automatically train or activate a model. Source: [paper_model.py](../ml/paper_model.py).

## 13. How local charts are generated

The initial history uses seeded random numbers and configured synthetic scale `base` and synthetic hedge coefficient `beta_demo`:

```text
X_t = base + cumulative_sum(epsilon_t),  epsilon_t ~ Normal(0, (0.004*base)^2)
u_0 = 0
u_t = 0.87*u_(t-1) + eta_t,             eta_t ~ Normal(0, (0.006*base)^2)
Y_t = 80 + beta_demo*X_t + u_t
```

New local bars continue a similar process, with prices clamped to at least 100 during advancement. The initial candle history has 720 observations mapped onto ten weekday sessions of 72 bars each; continuation uses its own weekday clock up to 15:30. This is a demonstration calendar, not a historical exchange calendar.

For each generated candle:

```text
open   = previous synthetic close
wiggle = close * Uniform(0.0001, 0.002)
high   = max(open, close) + wiggle
low    = min(open, close) - wiggle
volume = random integer from 1000 through 19999
```

This deliberately builds mean reversion into synthetic pairs. It tests the software pipeline, not whether HDFCBANK, POWERGRID or any other real company behaves this way. Source: [market.demo_history/demo_candles](../app/market.py) and [local_market.py](../app/local_market.py).

In both modes, the UI aggregates 15-minute bars from five-minute bars: first open, maximum high, minimum low, last close and summed volume. Upstox-mode 5-minute bars come from the historical/intraday candle endpoints; they are not reconstructed from sparse LTP polls. The custom canvas chart is a local renderer, not the TradingView service. The spread chart is a derived calculation rather than a directly traded exchange instrument.

## 14. Separate research workflows and limits

`ml/research.py` has a 160-tree forest for a different daily target:

```text
target_t = 1 if abs(s_(t+5)-mu_t) < abs(s_t-mu_t) else 0
```

It uses a five-session purge, balanced class weights, and chooses among thresholds 0.4/0.5/0.6/0.7 by validation accuracy. That target is spread contraction, not profit after costs. Editing this daily estimator does not replace the terminal's 180-tree model.

`scripts/nse_paper.py` and the notebook are separate daily workflows, with their own configuration, execution timing, outputs and account assumptions. The formulas above document the browser terminal; do not apply its 3/5 bps defaults to every notebook/CLI result. Read their saved configuration and source when auditing those results. A `BLOCKED` daily status means no valid daily update was produced.

Neither local nor Upstox paper mode reproduces actual borrow availability, taxes, broker margin/settlement, liquidity, order-book spreads, partial fills, corporate actions, exchange holidays or market impact. Real price data does not remove these execution assumptions. Chronological splits reduce one source of leakage but do not establish a deployable strategy; this implementation does not perform a complete walk-forward portfolio evaluation.
