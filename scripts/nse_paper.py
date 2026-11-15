"""Daily NSE research and a local simulated account. No broker/order API."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]


def today_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()


def validate_config(c):
    symbols = [s for group in c["groups"].values() for s in group]
    if len(symbols) != len(set(symbols)) or not symbols or not all(s.endswith(".NS") for s in symbols):
        raise ValueError("Use unique Yahoo NSE symbols ending in .NS.")
    if not any(len(group) >= 2 for group in c["groups"].values()):
        raise ValueError("At least one group needs two symbols.")
    numeric = ["formation_observations", "lookback", "minimum_correlation", "fdr_alpha",
               "entry_z", "exit_z", "stop_z", "max_hold", "max_model_age", "capital_inr",
               "gross_fraction", "cost_bps_per_leg", "annual_borrow_rate", "max_data_age_days"]
    if not all(np.isfinite(c[k]) for k in numeric):
        raise ValueError("Configuration numbers must be finite.")
    if not (0 <= c["exit_z"] < c["entry_z"] < c["stop_z"]):
        raise ValueError("Require 0 <= exit_z < entry_z < stop_z.")
    if not (0 < c["gross_fraction"] <= 1 and c["capital_inr"] > 0):
        raise ValueError("Require positive capital and gross_fraction in (0, 1].")
    if not (0 < c["fdr_alpha"] < 1 and -1 <= c["minimum_correlation"] <= 1):
        raise ValueError("Invalid statistical screening parameters.")
    if min(c["cost_bps_per_leg"], c["annual_borrow_rate"]) < 0:
        raise ValueError("Costs cannot be negative.")
    for key in ["formation_observations", "lookback", "max_hold", "max_model_age", "max_data_age_days"]:
        if c[key] < 1 or int(c[key]) != c[key]:
            raise ValueError(f"{key} must be a positive integer.")
    if c["lookback"] < 20 or c["formation_observations"] < max(120, c["lookback"] + 2):
        raise ValueError("Formation must exceed lookback and contain at least 120 observations.")


def load_prices(c, as_of, csv_dir=None):
    """Keep raw closes for fills and adjusted closes for statistical signals."""
    if csv_dir is not None:
        folder = Path(csv_dir)
        close = pd.read_csv(folder / "close.csv", index_col="Date", parse_dates=True)
        adjusted = pd.read_csv(folder / "adjusted_close.csv", index_col="Date", parse_dates=True)
        source = f"User CSV: {folder.resolve()}"
    else:
        import yfinance as yf
        cache = ROOT / ".market-cache"
        cache.mkdir(exist_ok=True)
        yf.set_tz_cache_location(str(cache))
        symbols = sorted(s for group in c["groups"].values() for s in group)
        raw = yf.download(symbols, start=c["history_start"], end=as_of, interval="1d",
                          auto_adjust=False, actions=False, progress=False, threads=False,
                          group_by="column", timeout=20)
        if raw is None or raw.empty:
            raise ValueError("Yahoo returned no prices. No current signal or paper update was produced.")
        close, adjusted = raw["Close"], raw["Adj Close"]
        source = "Yahoo Finance via yfinance; daily research data"
    return close, adjusted, source


def validate_prices(close, adjusted, c, as_of):
    symbols = sorted(s for group in c["groups"].values() for s in group)
    cleaned = []
    for name, frame in [("Close", close), ("Adjusted Close", adjusted)]:
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
        if frame.index.has_duplicates or frame.index.hasnans or frame.columns.has_duplicates:
            raise ValueError(f"{name}: duplicate dates/symbols or invalid dates.")
        missing = set(symbols) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing symbols {sorted(missing)}; no silent substitution.")
        frame = frame.loc[frame.index < pd.Timestamp(as_of), symbols].sort_index().astype(float)
        if frame.empty:
            raise ValueError(f"{name}: no completed daily observations before {as_of}.")
        latest = frame.apply(lambda s: s.last_valid_index())
        if latest.isna().any() or latest.nunique() != 1:
            raise ValueError(f"{name}: assets have inconsistent latest dates: {latest.to_dict()}")
        # Dropping gaps could make a next-session fill use a later day's price.
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError(f"{name}: gaps or invalid prices; repair the input rather than forward-fill.")
        cleaned.append(frame)
    close, adjusted = cleaned
    if not close.index.equals(adjusted.index):
        raise ValueError("Raw and adjusted price dates must match exactly.")
    age = (pd.Timestamp(as_of) - close.index[-1]).days
    if not 1 <= age <= c["max_data_age_days"]:
        raise ValueError(f"STALE DATA: latest close {close.index[-1].date()}, {age} days before {as_of}. No paper update.")
    if len(close) < c["formation_observations"] + 1:
        raise ValueError("Insufficient common history for formation plus a fresh signal observation.")
    return close, adjusted


def signal_series(adjusted, model, lookback):
    y, x = model["pair"]
    s = adjusted[y] - model["alpha"] - model["beta"] * adjusted[x]
    prior = s.shift(1)
    return (s - prior.rolling(lookback).mean()) / prior.rolling(lookback).std().replace(0, np.nan)


def screen_pairs(adjusted, c):
    # Latest observation is reserved for the first forward signal.
    formation = adjusted.iloc[-c["formation_observations"] - 1:-1]
    returns = formation.pct_change(fill_method=None).dropna()
    rows = []
    for group, symbols in c["groups"].items():
        for y, x in combinations(sorted(symbols), 2):
            if returns[[y, x]].std().min() <= 0:
                raise ValueError(f"Constant series in {y}/{x}.")
            fit = sm.OLS(formation[y], sm.add_constant(formation[x])).fit()
            _, p, _ = coint(formation[y], formation[x], trend="c", autolag="aic")
            rows.append({"group": group, "y": y, "x": x, "alpha": float(fit.params.iloc[0]),
                         "beta": float(fit.params.iloc[1]), "correlation": returns[y].corr(returns[x]),
                         "coint_p": float(p)})
    table = pd.DataFrame(rows)
    table["fdr_p"] = multipletests(table.coint_p.fillna(1), method="fdr_bh")[1]
    table["eligible"] = ((table.fdr_p < c["fdr_alpha"]) & (table.beta > 0) &
                         (table.correlation >= c["minimum_correlation"]))
    return table.sort_values(["fdr_p", "coint_p", "y", "x"]).reset_index(drop=True)


def next_action(z, side, age, c, model_age):
    if side:
        if not np.isfinite(z):
            return "exit_missing_signal"
        if abs(z) >= c["stop_z"]:
            return "exit_z_stop"
        # A next-close exit will add one more holding interval.
        if age + 1 >= c["max_hold"]:
            return "exit_time"
        if (side == 1 and z >= -c["exit_z"]) or (side == -1 and z <= c["exit_z"]):
            return "exit_mean"
        return "hold"
    if model_age >= c["max_model_age"]:
        return "wait_model_expired"
    if not np.isfinite(z) or not c["entry_z"] <= abs(z) < c["stop_z"]:
        return "wait"
    return "enter_long" if z < 0 else "enter_short"


def new_account(close, adjusted, c, row, as_of):
    pair = [row["y"], row["x"]]
    model = {"pair": pair, "alpha": float(row["alpha"]), "beta": float(row["beta"]),
             "created_on": str(close.index[-1].date()),
             "formation_end": str(close.index[-2].date())}
    date = close.index[-1]
    z = float(signal_series(adjusted, model, c["lookback"]).iloc[-1])
    if not np.isfinite(z):
        raise ValueError("Latest Z-score is not finite.")
    # Preserve the adjustment scale on which the model was fitted.
    anchor = {s: float(adjusted.loc[date, s]) for s in pair}
    state = {"version": 1, "config": deepcopy(c), "model": model,
             "anchor_adjusted": anchor, "last_date": str(date.date()),
             "last_close": close.loc[date, pair].to_list(),
             "last_adjustment": (adjusted.loc[date, pair] / close.loc[date, pair]).to_list(),
             "equity": c["capital_inr"], "shares": [0, 0], "side": 0, "age": 0,
             "model_age": 0, "entry_equity": None, "entry_date": None, "trades": [],
             "pending": next_action(z, 0, 0, c, 0), "ledger": []}
    state["ledger"].append({"date": state["last_date"], "observed_on": as_of,
                            "equity_inr": state["equity"], "gross_pnl_inr": 0., "cost_inr": 0.,
                            "borrow_inr": 0., "shares_y": 0, "shares_x": 0, "zscore": z,
                            "executed": "account_started", "pending": state["pending"]})
    return state


def advance_account(state, close, adjusted, as_of):
    state = deepcopy(state)  # Failed checks must not partially modify persisted state.
    c, model = state["config"], state["model"]
    pair = model["pair"]
    anchor_date, last_date = pd.Timestamp(model["created_on"]), pd.Timestamp(state["last_date"])
    if anchor_date not in adjusted.index or last_date not in close.index:
        raise ValueError("Input history is missing the account anchor or last processed date.")
    # Future cash distributions may rebase all earlier adjusted prices. Restore
    # the scale at the original anchor before using the frozen coefficients.
    adjusted = adjusted[pair] * (pd.Series(state["anchor_adjusted"]) / adjusted.loc[anchor_date, pair])
    if not np.allclose(close.loc[last_date, pair], state["last_close"], rtol=1e-6, atol=.005):
        raise ValueError("Previously processed raw prices changed. Reconcile data/corporate actions before updating.")
    if not np.allclose(adjusted.loc[last_date, pair] / close.loc[last_date, pair],
                       state["last_adjustment"], rtol=1e-5, atol=1e-8):
        raise ValueError("Historical adjustment factors changed; reconcile the paper account before updating.")
    zscores = signal_series(adjusted, model, c["lookback"])
    new_dates = close.index[close.index > last_date]
    for date in new_dates:
        price = close.loc[date, pair].to_numpy(float)
        factors = (adjusted.loc[date, pair] / close.loc[date, pair]).to_numpy(float)
        if not np.allclose(factors, state["last_adjustment"], rtol=1e-5, atol=1e-8):
            raise ValueError(f"Corporate action/adjustment change on {date.date()}. Paper update paused; dividends/splits require reconciliation.")
        old_price = np.array(state["last_close"])
        shares = np.array(state["shares"], dtype=int)
        days = (date - pd.Timestamp(state["last_date"])).days
        gross = float(shares @ (price - old_price))
        borrow = float(np.maximum(-shares, 0) @ old_price) * c["annual_borrow_rate"] * days / 365
        state["equity"] += gross - borrow
        if state["side"]:
            state["age"] += 1
        action, cost = state["pending"], 0.
        if action.startswith("exit_") and state["side"]:
            cost = float(np.abs(shares) @ price) * c["cost_bps_per_leg"] / 10_000
            state["equity"] -= cost
            state["trades"].append({"entry_date": state["entry_date"], "exit_date": str(date.date()),
                                     "side": state["side"], "net_pnl_inr": state["equity"] - state["entry_equity"],
                                     "holding_intervals": state["age"], "reason": action})
            shares = np.zeros(2, dtype=int)
            state.update(side=0, age=0, entry_date=None, entry_equity=None)
        elif action.startswith("enter_") and not state["side"]:
            # Convert adjusted-price beta to raw share quantities, then floor each leg.
            raw_beta = model["beta"] * factors[1] / factors[0]
            units = c["gross_fraction"] * state["equity"] / (price[0] + raw_beta * price[1])
            quantities = np.floor([units, raw_beta * units]).astype(int)
            if quantities.min() < 1:
                action = "skip_insufficient_capital"
            else:
                side = 1 if action == "enter_long" else -1
                shares = side * quantities * np.array([1, -1])
                state.update(side=side, age=0, entry_equity=state["equity"], entry_date=str(date.date()))
                cost = float(np.abs(shares) @ price) * c["cost_bps_per_leg"] / 10_000
                state["equity"] -= cost
        if state["equity"] <= 0:
            raise ValueError("Paper equity exhausted; review the account before continuing.")
        state["model_age"] += 1
        z = float(zscores.loc[date])
        pending = next_action(z, state["side"], state["age"], c, state["model_age"])
        # Maintain at least one flat observation after any exit.
        if action.startswith("exit_") and pending.startswith("enter_"):
            pending = "wait"
        state.update(last_date=str(date.date()), last_close=price.tolist(), last_adjustment=factors.tolist(),
                     shares=shares.tolist(), pending=pending)
        state["ledger"].append({"date": state["last_date"], "observed_on": as_of,
                                "equity_inr": state["equity"], "gross_pnl_inr": gross,
                                "cost_inr": cost, "borrow_inr": borrow,
                                "shares_y": int(shares[0]), "shares_x": int(shares[1]),
                                "zscore": z if np.isfinite(z) else None,
                                "executed": action, "pending": pending})
    return state


@contextmanager
def session_lock(folder):
    lock = folder / "run.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise ValueError(f"Another run holds {lock}. If it crashed, verify no run is active before removing this lock.") from exc
    try:
        handle.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(config_path=ROOT / "nse_config.json", output=ROOT / "outputs/nse_paper", as_of=None, csv_dir=None):
    as_of = as_of or today_ist()
    if pd.Timestamp(as_of) != pd.Timestamp(as_of).normalize():
        raise ValueError("as_of must be a date, without a time.")
    if pd.Timestamp(as_of).date() > pd.Timestamp(today_ist()).date():
        raise ValueError("Cannot request a future as-of date.")
    as_of = pd.Timestamp(as_of).date().isoformat()
    c = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config(c)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with session_lock(output):
        state_path = output / "account.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        if state and state["config"] != c:
            raise ValueError("Existing paper account has frozen settings. Use a separate output directory for a new experiment.")
        try:
            close, adjusted, source = load_prices(c, as_of, csv_dir)
            close, adjusted = validate_prices(close, adjusted, c, as_of)
            close.to_csv(output / "close.csv", index_label="Date")
            adjusted.to_csv(output / "adjusted_close.csv", index_label="Date")
            if state:
                if close.index[-1] < pd.Timestamp(state["last_date"]):
                    raise ValueError("Input ends before the paper account's last date; updates cannot move backward.")
                state = advance_account(state, close, adjusted, as_of)
            else:
                table = screen_pairs(adjusted, c)
                table.to_csv(output / "pair_screen.csv", index=False)
                eligible = table.loc[table.eligible]
                if eligible.empty:
                    result = {"status": "NO_QUALIFYING_PAIR", "as_of_exclusive": as_of,
                              "latest_close": str(close.index[-1].date()), "source": source,
                              "message": "No pair passed the fixed screen. No account/trade created."}
                    save_json(output / "status.json", result)
                    return result
                state = new_account(close, adjusted, c, eligible.iloc[0], as_of)
            state["source"] = source
            save_json(state_path, state)  # Account JSON is the authoritative, atomic commit.
            pd.DataFrame(state["ledger"]).to_csv(output / "paper_ledger.csv", index=False)
            pd.DataFrame(state["trades"], columns=["entry_date", "exit_date", "side", "net_pnl_inr",
                                                   "holding_intervals", "reason"]).to_csv(output / "paper_trades.csv", index=False)
            result = {"status": "PAPER_ONLY", "as_of_exclusive": as_of,
                      "latest_close": state["last_date"], "source": source,
                      "pair": state["model"]["pair"], "zscore": state["ledger"][-1]["zscore"],
                      "pending_next_close": state["pending"], "equity_inr": state["equity"],
                      "shares": state["shares"], "closed_trades": len(state["trades"]),
                      "model_age_observations": state["model_age"],
                      "short_borrow": "Assumed available for paper simulation; NOT verified",
                      "costs": "Illustrative all-in per-leg bps and annual borrow; NOT a broker tax/fee schedule"}
            save_json(output / "status.json", result)
            return result
        except Exception as exc:
            save_json(output / "status.json", {"status": "BLOCKED", "as_of_exclusive": as_of,
                                               "message": str(exc), "account_updated": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "nse_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/nse_paper")
    parser.add_argument("--as-of", help="Exclusive date YYYY-MM-DD; defaults to today in Asia/Kolkata")
    parser.add_argument("--csv-dir", type=Path, help="Use close.csv and adjusted_close.csv from your own data provider")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, args.output, args.as_of, args.csv_dir), indent=2))
    except Exception as exc:
        parser.exit(1, f"Paper run stopped: {exc}\n")


if __name__ == "__main__":
    main()
