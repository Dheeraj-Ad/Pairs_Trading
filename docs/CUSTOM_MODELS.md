# Changing the application and supplying your own model

For the new five-stage **daily report** pipeline, see [REPORT_ALIGNMENT.md](REPORT_ALIGNMENT.md#make-your-own-daily-model). This guide covers the separate five-minute terminal and completed-paper-trade learner.

Return to the [README](../README.md) for functionality and [CALCULATIONS.md](CALCULATIONS.md) for the implemented mathematics. This is a guide to source-code changes; the application does not currently have a model-upload button, automatic plugin discovery, or a model selector in `.env`.

## 1. Choose the component you actually want to change

| Goal | Start here | Effect |
| --- | --- | --- |
| Change the terminal's entry classifier | [ml/profit_model.py](../ml/profit_model.py), `train_profit()` | Replaces the classifier trained for each pair by Scan & train |
| Change price-derived features | Same file, `features()`; also [ml/research.py](../ml/research.py), `FEATURES` | Changes classifier inputs and the features saved for later paper learning |
| Change the learned filter from closed trades | [ml/paper_model.py](../ml/paper_model.py), `train_trades()` | Changes the optional second classifier; does not replace the primary price model |
| Change legacy daily research | [ml/research.py](../ml/research.py), `train()` | Changes the separate daily research model only |
| Change screening or automatic entry/exit rules | [app/terminal.py](../app/terminal.py), `scan()`, `configure()`, `manage()` | Changes qualification, accepted signals or risk exits |
| Change fees, sizing, fills or paper accounting | [app/paper.py](../app/paper.py) | Changes simulator execution; training costs and displays must be kept consistent |
| Add companies/pairs | [app/market.py](../app/market.py), `PAIRS`, `DEMO_PARAMETERS` | Extends the configured candidate universe and local generator parameters |
| Change local price behavior | [app/local_market.py](../app/local_market.py), [app/market.py](../app/market.py) | Changes synthetic data, not real exchange prices |
| Change screens/charts/history | [app/static/terminal.html](../app/static/terminal.html), [terminal.js](../app/static/terminal.js), [local.js](../app/static/local.js), [terminal.css](../app/static/terminal.css) | Changes the browser interface |

The notebook and `scripts/nse_paper.py` have separate workflows. Changing them does not automatically change browser-terminal behavior.

## 2. How a model reaches an automatic trade

```text
Local candles OR Upstox candles
  -> completed, aligned five-minute DataFrame
  -> train_profit(frame): formation + training + validation + test
  -> terminal.scan(): statistical and ML qualification
  -> predict(model, current_frame, frozen_baseline)
  -> terminal.manage(): score, spread, freshness, time and capital checks
  -> optional paper-outcome score filter
  -> PaperBook.trade(): simulated two-leg fills
  -> persistent trade history and saved entry features
  -> explicit paper-outcome training after sufficient completed trades
```

The model supplies an entry score. It does not bypass the pair screen, exposure limits or risk exits. Supplying a different estimator does not itself implement a different execution strategy.

## 3. Small change: tune the existing Random Forest

Find this constructor in `ml/profit_model.py`:

```python
model = RandomForestClassifier(
    n_estimators=180,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=1,
)
```

For example, setting `n_estimators=300` trains 300 trees. Increasing depth allows more complex rules; increasing minimum leaf size constrains them. More trees or depth do not guarantee better performance. Compare using the same data, cost assumptions and chronological validation; reserve a final holdout for reporting. If you repeatedly inspect and optimize against that holdout, it is no longer untouched.

Tree counts are implemented in source code, not `.env`. Restart the Python server and run **Scan & train pairs** after editing. Editing the source while a server is running does not hot-reload its already fitted models. Close existing automated positions before replacing their models. Update the UI's model descriptions if you change the settings.

## 4. Worked example: a custom Extra Trees classifier

This example uses an estimator already supplied by scikit-learn. It preserves the existing target, features, chronological split and report structure. It is an alternative to evaluate, not a claim of superior returns.

Create `ml/custom_model.py` with:

```python
from sklearn.ensemble import ExtraTreesClassifier


def build_model():
    return ExtraTreesClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,
    )
```

In `ml/profit_model.py`, add:

```python
from ml.custom_model import build_model
```

Replace only the existing `RandomForestClassifier(...)` construction inside `train_profit()` with:

```python
model = build_model()
model.fit(xt, yt)  # Keep the existing fit; do not add a second fit below it.
```

The function already calls `model.fit(xt, yt)` immediately after construction; the result should contain exactly one such fit. Leave the existing feature generation, targets, partitions, purge, validation threshold selection and test evaluation in place.

Update the report's algorithm label to identify Extra Trees, and assign a new model version. `app/terminal.py` currently also writes a hardcoded `model_version: 2` into automatic decisions; update that metadata to match your version. Update the model descriptions in `app/static/local.js` and the README. The ledger should identify the model that actually made each decision.

ExtraTreesClassifier supplies `classes_`, `predict_proba` and `feature_importances_`, so it matches the attributes currently consumed by the training and prediction code. Its randomization differs from Random Forest, including the default bootstrap setting; keep its actual settings in your experiment record. See the [official Extra Trees reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

To experiment with a different estimator in the paper-outcome learner instead, change the constructor inside `ml/paper_model.py:train_trades()` and retain that function's account-specific dataset, chronological splits and overlap purge.

## 5. Model and module interfaces

### Estimator requirements

For a replacement that leaves the surrounding code intact:

```text
fit(X_train, y_train)
classes_                         includes labels 0 and 1 after fitting
predict_proba(X)                  returns [number_of_rows, number_of_classes]
feature_importances_              one numeric value per input feature
```

`X` is a pandas DataFrame with these columns in this order:

```python
["zscore", "abs_zscore", "spread_change", "spread_volatility"]
```

The score is extracted using:

```python
positive_column = list(model.classes_).index(1)
score = float(model.predict_proba(X)[0, positive_column])
```

Do not assume class 1 is always the second output column in your own adapter. Return finite scores in `[0, 1]` for the positive-net-profit class. If your neural model emits logits, convert them appropriately and evaluate their calibration before interpreting them as probabilities. A regressor's predicted rupee P&L is not interchangeable with the current probability-score threshold.

An estimator without `feature_importances_`, such as logistic regression, also needs a report change in both training modules where applicable. For example:

```python
importance = getattr(model, "feature_importances_", None)
importance_report = (
    dict(zip(FEATURES, map(float, importance)))
    if importance is not None else {}
)
# Use feature_importance=importance_report in the returned report.
```

An empty report is preferable to inventing importances. Coefficients and tree impurity importances measure different things. Scaling/preprocessing should be fitted on training rows only, ideally inside a sklearn Pipeline; never fit a scaler on the full train/validation/test dataset.

### Whole-module requirements

`app/terminal.py` imports these functions from `ml/profit_model.py`:

```python
train_profit(frame) -> (fitted_model, report_dict)
predict(model, frame, baseline_info) -> signal_dict
features(frame, baseline_info) -> feature_dataframe
```

The input frame is indexed by timezone-aware candle-start timestamps and provides `y`, `x`, `y_open`, `x_open`. Values must be aligned and only completed candles may be scored. The statistical baseline currently requires at least 400 observations; after labels and purging, partition-size checks still apply.

The prediction result contains:

```python
{
    "score": 0.72,           # illustrative value only
    "zscore": -2.3,         # calculated from the actual current input row
    "bar_time": 0,          # replace with that row's UTC epoch seconds
    "side": "long",         # long for negative Z, short for non-negative Z
}
```

The example dictionary demonstrates the schema, not valid values to return verbatim. The terminal adds freshness, training time and threshold fields. The score and Z-score must come from the same latest available completed row. Returning a constant score to force trading invalidates the model workflow.

Preserve the report keys consumed by the terminal and UI:

| Group | Fields |
| --- | --- |
| Identity | `algorithm`, `target`, `timeframe`, `version` |
| Qualification | `threshold`, `validation_pass`, `validation_signals`, `validation_mean_return`, `validation_auc` |
| Evaluation | `train_rows`, `validation_rows`, `test_rows`, `purge_sessions`, `accuracy`, `balanced_accuracy`, `auc`, `majority_baseline`, `feature_importance`, `splits` |
| Statistical baseline | Preserve the full dictionary returned by `ml.research.baseline`, including `alpha`, `beta`, `correlation`, `coint_p` and `history_end` |

The terminal adds `source`, `exchange` and `trained_at`. Report values must be JSON-serializable, with `None` for unavailable metrics rather than NaN. A changed model with the wrong report structure can fail the scan even if its estimator fits successfully.

If you move all three functions to a new module, change the import in `app/terminal.py` consistently. Its `features()` import is also used to save entry features for later paper learning. Changing only `predict()` while leaving the feature recorder pointed at a different schema creates inconsistent training records.

## 6. Adding features or changing the prediction horizon

When adding a feature such as a rolling return:

1. Calculate it in `features()` using information available by the completed signal bar. Never use negative shifts or future rows for inputs.
2. Update the feature names/order and every consumer. `FEATURES` currently lives in `ml/research.py` and is shared with the paper learner and daily research. Either update all producers consistently or create separate explicit schemas for different model families.
3. Ensure `Terminal.entry_decision()` saves the same features used for prediction. New paper-model features must exist in stored entry snapshots. Old rows missing them will be excluded; do not silently reconstruct them using future data.
4. Update reports, tests and the model/schema version. Rescan and retrain; existing fitted models do not gain a new input column automatically.

To change the forecast horizon, editing `HORIZON = 5` alone is **not enough**: the current implementation explicitly uses `shift(-6)`, 25-minute borrowing, a 1,800-second continuity check, six-row partition purges and corresponding descriptions. For `H` holding bars, entry remains `t+1`, exit becomes `t+1+H`, holding time is `5*H` minutes, and the continuity/purge rules must change consistently. The automatic holding-limit configuration is another separate setting. The ML target and actual exit policy need to be compared deliberately.

## 7. Changing risk, cost or universe settings

Use the existing UI for supported score, allocation, stop-loss, profit-target and daily-loss settings. `DEFAULTS` in `app/terminal.py` supplies defaults only: already saved account settings override them. Change saved settings through the app/API rather than deleting the ledger.

For cost changes, review all of:

- Actual execution constants in `app/paper.py`.
- Label cost constants and fixed holding assumptions in `ml/profit_model.py`.
- The closing-cost estimate (`0.0008`) in `Terminal.manage()`.
- The cost metadata in `Application.snapshot()` and UI/documentation text.

Keep historical fills and recorded costs unchanged. Version the new assumptions rather than rewriting old profits. The current learner pools eligible outcomes across its account, so mixed model/cost policies can change what it learns; use a separate experimental account/database when comparison matters.

For a new candidate pair, add its ID, display name, symbols and verified ISINs to `PAIRS`, plus corresponding synthetic parameters in `DEMO_PARAMETERS`. Current local generation assumes the pairs use distinct symbols: repeating one symbol in multiple independently generated pairs can create inconsistent quote paths. Supporting overlapping pairs requires a shared per-symbol price process first. Existing universe migration adds missing synthetic pair histories at the current simulation clock without replacing prior candles; it is not an importer of exchange history.

The terminal currently accepts local generated candles or the Upstox adapter. The notebook/daily CLI's CSV support is not automatically a custom CSV feed for the terminal; a new terminal feed needs a provider producing the same aligned OHLCV/quote format and a clearly identified data source.

## 8. Validate before using the changed model

Use a Git branch and an isolated data directory for experiments. Keep `.env`, real credentials, runtime databases and generated artifacts out of source control. Preserve existing accounts before changing persistence or feature formats.

An isolated price-model scan can run without modifying the main paper ledger:

```python
import tempfile
import time
from app.server import Application

with tempfile.TemporaryDirectory() as directory:
    app = Application(directory, initial_mode="demo")
    app.terminal.scan()
    deadline = time.monotonic() + 120
    while app.terminal.job["running"] and time.monotonic() < deadline:
        time.sleep(0.1)
    if app.terminal.job["running"]:
        raise TimeoutError("Scan did not finish in time")
    for row in app.terminal.screen:
        print(row["id"], row["eligible"], row["reasons"])
```

Local training verifies integration, not real-market performance. Evaluate on appropriately sourced historical market data with realistic costs before drawing trading conclusions. Any walk-forward evaluation, probability calibration or different execution policy must be implemented and measured; the current UI does not automatically supply those experiments.

Run existing checks after meaningful model changes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check app/static/terminal.js
node --check app/static/local.js
# Browser checks require Playwright and local Chrome:
.\.venv\Scripts\python.exe scripts/check_terminal_browser.py
```

Add checks appropriate to the changed behavior: feature order, finite positive-class scores, no future-data leakage, unchanged training/threshold selection when only test data changes, realistic label costs, and no entry on rejected/stale signals. Do not merely assert that your own function returns the constant you supplied.

Finally, close automated positions, restart the server, run **Scan & train**, inspect validation and test results, and explicitly enable paper automation. For the closed-trade model, use **Train from completed trades** and review its eligibility before enabling its filter.

## 9. Saved models and important boundaries

Training writes `.joblib` artifacts and JSON reports under `runtime/`, but the app does not automatically unpickle those estimators on restart. Report metadata may still be shown; a loaded model and an old report are different states. Copying a `.pkl`/`.joblib` file into the directory does not activate a custom model. Implement and audit explicit loading if you need it; pickle/joblib files can execute code during loading and should only come from a trusted source.

Neural sequence models require additional work beyond swapping the estimator: causal window construction, masks/gap handling, feature scaling fitted only on training data, model serialization, compatible probability output and appropriate chronological purging. Add dependencies deliberately and document the resulting interface.

Whatever model you add, Upstox remains a read-only data connection in this application. A new classifier does not turn simulated execution into real orders.
