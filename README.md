# Pairwise - live-data pairs trading terminal

A local NSE/BSE pairs trading terminal with synthetic local simulation, actual Upstox OHLCV when connected, ML-qualified pairs, manual and automatic **paper** execution, and detailed profit/loss history. All execution uses virtual money. No broker order endpoint is implemented.

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.server
```

Open **http://127.0.0.1:8765**. Use `--port 8766` if needed. `start_app.ps1` is an alternative launcher. The app starts in **local simulation**, scans the candidate pairs, and needs no API credentials. Use `--live` to start with the Upstox source instead. It binds to localhost and is not intended for public hosting.

## Use locally now

1. Wait for the initial scan, then select a qualified pair. Local OHLCV is **generated synthetic data**, not downloaded NSE/BSE history.
2. Open a manual pair trade using the ticket, or click **Enable automatic paper trades**. Automatic entries retain the model score, qualification and risk gates; trades are not forced and profits are not guaranteed.
3. Click **Run local market** (one five-minute candle approximately every two seconds), **Step 5 minutes**, or **Advance 75 candles**. The simulated clock drives quotes, candles, fills, borrowing, holding limits and session exits. Pausing the local market freezes time; pausing automatic entries keeps exits active when data advances.
4. Review **P&L** for overall, manual, automatic and pair/sector results. **Trade history** includes closed positions and every individual entry/exit fill, with method/pair filters and CSV exports.
5. In **ML models → Learn from paper trading**, train once at least 40 eligible trades have closed. Both manual and automatic trades save entry features. Legacy trades without those features are counted but excluded from training. The model uses chronological 60/20/20 splits, removes overlapping outcomes at boundaries, and reports validation plus untouched test metrics. You may enable an additional automatic-entry filter only after validation passes; retraining pauses that filter for review.

Local candles, progress, account balances and transactions persist in SQLite. Restart pauses playback and entries; models are rebuilt by scanning. Existing automated positions retain their original risk settings and must close before replacing models. Local models remain frozen across simulated sessions until rescanned; Upstox models retain the 24-hour expiry. The local weekday calendar does not simulate exchange holidays or corporate actions. Local and Upstox training data and accounts are isolated.

## Connect Upstox later

1. **Connect with Upstox**: choose NSE or BSE, click the sign-in button, and log in on Upstox. The app receives the authorization code and obtains the access token automatically. Complete the one-time app setup below first. Tokens remain in server memory. Manual token entry is available under **Connection options & manual-token fallback**.
2. **Scan & train pairs**: fetch approximately 27 calendar days of five-minute candles and today's intraday candles, align both legs, screen pairs, and train a model for each candidate. The background scan reports progress in **ML models**.
3. Only qualifying pairs appear in the terminal. Select a pair to see both stocks' actual candlesticks, volume, and completed-candle spread Z-scores. Charts support hover OHLCV, wheel zoom, drag pan, and 5m/15m intervals. Fifteen-minute candles are aggregates of actual five-minute OHLCV. LTP is shown separately; sparse quote samples never manufacture provider candles.
4. Choose score, allocation, stop-loss, profit target, and daily loss limit, then **Enable automatic paper trades**. In Upstox mode, entries require fresh exchange data; all orders still use virtual money. Connecting does not transfer local profits or models into the Upstox paper account.
5. Review positions, completed trades, and decision logs in the terminal or **Trade history**. CSV export includes both leg fills, model decision, origin, exit reason, fees, borrowing, and net P&L.

## One-time Upstox sign-in setup

The app owner must register an app in [Upstox Developer Apps](https://account.upstox.com/developer/apps). An existing Upstox trading account by itself does not supply the application's API key and secret. Once configured, the person using this local app signs in with their existing Upstox account, without copying access tokens.

1. Register this exact callback URL in your Upstox developer app:

   ```text
   http://127.0.0.1:8765/auth/upstox/callback
   ```

2. Copy `.env.example` to `.env` **if `.env` does not already exist**. Enter your app's API key as `UPSTOX_CLIENT_ID` and its API secret as `UPSTOX_CLIENT_SECRET`. Keep `UPSTOX_REDIRECT_URI` equal to the registered callback URL. Do not enter your Upstox password or PIN here. `.env` is ignored by Git and is never served by the application; keep this local credentials file private.
3. Restart the server and open **http://127.0.0.1:8765**. Click **Connect Upstox**, choose the exchange, then **Connect with Upstox**. Complete any login, OTP/PIN, or authorization required on Upstox's own page. The browser returns to the terminal with polling connected and automatic trade entries paused.

Environment variables override `.env` values. If you change the port, update both the registered callback and `UPSTOX_REDIRECT_URI`. Use the same hostname in the browser and callback: `localhost` and `127.0.0.1` are different origins. This local implementation supports HTTP loopback callbacks only; it is not a hosted, multi-user login service.

Login attempts expire after ten minutes and are tied to an HttpOnly, SameSite=Lax browser cookie and a single-use random state. Tokens and the API secret never enter browser storage. The server exchanges the code directly with Upstox; the callback then redirects to a clean application URL. A denied or failed authorization preserves the prior connection. Reconnecting to the same exchange remains possible with existing automated positions; changing exchanges still requires closing them first.

Upstox access tokens expire at the next 03:30 IST. The app shows the expiry, pauses new entries after expiry, and offers **Reconnect with Upstox**. A server restart also requires sign-in again because tokens are kept only in memory. This automates the token exchange, not the broker's login/OTP requirements; there is no promise of permanent unattended authorization. See the [Upstox authentication](https://upstox.com/developer/api-documentation/authentication/) and [token](https://upstox.com/developer/api-documentation/get-token/) documentation.

## Pair classification and model

The configured universe contains three candidate pairs: HDFCBANK/ICICIBANK, INFY/TCS, and MARUTI/M&M. `app/market.py` defines their ISINs. This is not a full-exchange scan. NSE/BSE instrument keys come from Upstox's current instrument masters.

Qualification requires a positive formation hedge ratio, return correlation of at least 0.40, cointegration FDR below 0.05 across all candidates, validation AUC above 0.50, and at least eight threshold-selected validation signals with positive average modeled net return. Failed candidates remain visible with reasons in ML models, but have no terminal chart. An empty qualified set is a valid result; no fallback pair is inserted.

`ml/profit_model.py` fits a random forest to predict positive two-leg net return. Features are Z-score, absolute Z-score, normalized spread change, and spread volatility. Labels observe signal close t, enter at open t+1, and exit at open t+6. Both legs include 3 bps fees per fill, 5 bps adverse slippage per fill, and illustrative 10% annual short borrow. Labels use fractional hedge quantities; actual paper fills use integer shares. Labels spanning missing intervals or overnight are excluded.

Data splits are chronological: 30% formation, 30% train, 20% validation, 20% test. The hedge ratio uses formation only; six bars are purged at partition boundaries. Validation selects the threshold. Test metrics never qualify pairs or tune thresholds. Overlapping labels are dependent, and validation mean signal returns are not an independent portfolio backtest. Scores are **uncalibrated model estimates**, not guaranteed win probabilities. Earlier exits in paper execution differ from the fixed-horizon ML label policy.

## Polling and execution

- Quote worker: target every five seconds. Candle worker: target every fifteen seconds. Exit/entry checks: every second in a separate worker, so candle HTTP latency cannot block risk checks. Actual refresh times are visible; provider latency can delay target cadences.
- Both legs need positive last prices, last-trade timestamps under 60 seconds old, and timestamps no more than 15 seconds apart. Failed quote requests block fills. Failed candle requests block new automatic entries while quote-driven exit management continues. Missing candle legs are never forward-filled.
- Model inputs use completed five-minute candles only. Automatic entries require a signal no more than 90 seconds after its candle closes, a model trained within 24 hours, a score at least the larger of the user threshold and validation threshold, and spread magnitude between the entry threshold (default 2) and 4. The current quote spread must still satisfy the entry band and direction.
- Default entry window: weekdays 09:20-14:55 IST. Exits: stop-loss, profit target, spread reversion, spread stop at absolute Z of 4, holding limit (maximum 25 minutes), daily loss halt, and session square-off from 15:20 IST. These are simulator settings, not an authoritative exchange calendar. Exchange holidays naturally block fills through stale quotes.
- One open position per pair, integer quantities, 50% equity maximum allocation per pair, 1x aggregate entry gross exposure. Both legs commit atomically to SQLite. A permanent account/pair/candle request ID prevents repeat entries on the same signal, including after an exit.
- Fees and slippage are charged on both legs at entry and exit. Borrow accrues by elapsed time on short entry notional. Exit triggers estimate closing costs and include entry fees. A stop is not a guaranteed fill price; stale quotes can delay exits until quotes recover.
- Daily loss uses marked equity relative to the first automation activation/check that day. The daily anchor and halt persist in SQLite. A halt cannot be cleared by toggling automation on the same day.
- **Pause new entries** keeps automatic exits active. Close automated positions before switching exchanges or sources. Reconnecting a token for the same account remains possible with open positions. Settings and trade metadata persist; active models are rebuilt by scanning, and new entries always start disabled after a server restart. Existing automatic positions resume exit checks once their original account has fresh quotes. Keep the local server running for automation.

## Code and saved state

```text
app/server.py          Local HTTP API, quote/candle/execution workers
app/market.py          Upstox GET adapter, OHLCV validation, explicit demo data
app/oauth.py           Browser-bound Upstox authorization and server-side token exchange
app/terminal.py        Screening, signal inference, automation and decision audit
app/paper.py           Persistent atomic two-leg simulator
app/local_market.py    Persistent synthetic candles and simulation clock
app/reporting.py       Reconciled method/pair P&L and all leg transactions
app/static/local.js    Local controls, manual ticket, P&L and learning workspace
app/static/terminal.*  Trading terminal, candlestick renderer, models and history
ml/profit_model.py     Five-minute profit classifier and causal target construction
ml/paper_model.py      Closed-paper-trade classifier with chronological evaluation
ml/research.py         Preserved daily spread research
runtime/              Ignored SQLite account state, models, and reports
pairs_trading.ipynb    Preserved transaction-level research notebook
```

Demo/live and NSE/BSE accounts are separate, each initially funded with INR 10,00,000. Previous paper accounts are preserved. Legacy daily research APIs remain available, but are not the automatic five-minute model.

The simulator does not model real short-borrow availability, broker-specific margins, taxes, dividends, splits, order-book liquidity, partial fills, or cash-equity settlement constraints. Corporate actions can break raw-price models. No model can guarantee profitable trades. Use the notebook below for further backtesting.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check app/static/terminal.js
# Optional UI QA with locally installed Chrome:
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe scripts/check_terminal_browser.py
.\.venv\Scripts\python.exe scripts/check_oauth_browser.py
```

Tests cover target chronology, qualification data, atomic fills, persistent daily limits, duplicate signals, stale feeds, automatic loss/profit exits, account isolation, token reconnection, OAuth state/cookie binding, replay rejection, expiry, charts, and mobile layout. Browser tests use temporary accounts, synthetic candles, and mocked Upstox authorization/token responses; screenshots go to `outputs/terminal-preview/` and `outputs/oauth-preview/`. Real authorization requires your registered app credentials and Upstox account; no live connection is claimed by mocked tests.

Upstox references: [quotes](https://upstox.com/developer/api-documentation/get-full-market-quote-v3/), [historical candles](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/), [intraday candles](https://upstox.com/developer/api-documentation/v3/get-intra-day-candle-data/), [authentication](https://upstox.com/developer/api-documentation/authentication/).

---

# Original pairs trading research notebook

Open **pairs_trading.ipynb** and run its cells from top to bottom. The default is a reproducible, offline synthetic-data demonstration, explicitly labeled in every result. It includes data checks, correlation and clustering, cointegration screening with multiple-testing correction, a fixed hedge ratio, rolling Z-scores, a transaction-level backtest, and a random-forest entry filter.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab pairs_trading.ipynb
```

An existing Jupyter/Anaconda installation also works if the dependencies are available. Select that Python kernel. The notebook does not install packages automatically.

## Use your own data

In the configuration cell set `DATA_SOURCE = "csv"`. Supply `data/prices.csv` with a `Date` column and one adjusted-close column per asset, all in the same currency and on a common trading calendar. Optional `data/volumes.csv` has the same layout. Missing price rows are dropped, never forward-filled; the notebook reports the count. At least 900 common observations are required.

Alternatively, set `DATA_SOURCE = "yfinance"` and edit `TICKERS`, `START`, and `END`. This requires internet access and the optional yfinance package. Example symbols are a data-loading example, not a claim that they form suitable pairs. A download failure never silently switches to synthetic data. The notebook stops if no pair passes the formation-period screen.

## Design and limitations

- Formation / ML training / validation / test use chronological 30% / 30% / 20% / 20% blocks.
- Pair selection and the hedge ratio use only formation data. ML labels crossing the validation boundary are excluded. Only the probability threshold is selected on validation data.
- A signal observed at close t executes at close t+1. Fixed share quantities earn P&L from subsequent price changes. Costs cover both legs, entry, exit, and final liquidation; borrow cost accrues on short holdings by calendar days.
- Gross exposure is capped at 1x equity at entry; exposure can drift while shares remain fixed. A hedge-ratio spread is not necessarily dollar-neutral or market-beta-neutral.
- Synthetic assets intentionally contain mean reversion. Their performance is not evidence of a real trading opportunity. The illustrative costs omit cash interest, margin funding, taxes, borrow availability, corporate-action execution, market impact, and intraday stop fills. Adjusted-price shares are a research approximation.
- The final test is a single held-out block, not a full walk-forward study. Repeated tuning after looking at test results invalidates that holdout.

The notebook cites the relevant statsmodels, scikit-learn, and yfinance documentation. It exports charts, metrics, positions, trades, and a run configuration to `outputs/`.

## Verify or regenerate

```powershell
python scripts/build_notebook.py
python scripts/verify_notebook.py
```

The builder regenerates the notebook and clears its saved outputs, so do not run it after making manual notebook edits. The verifier executes all cells, saves the executed notebook, and checks signal timing, transaction costs, borrowing, liquidation, and invariance to future-price changes.
