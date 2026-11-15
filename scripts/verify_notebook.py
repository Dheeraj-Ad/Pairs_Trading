"""Execute the notebook and run independent accounting/timing regression checks."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter-runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".jupyter-runtime" / "ipython")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
path = ROOT / "pairs_trading.ipynb"
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
original_count = len(nb.cells)
checks = r'''
# Deterministic hand-calculated trade: signal on day 0, entry day 1, exit day 2.
check_dates = pd.date_range("2024-01-01", periods=6)
check_px = pd.DataFrame({"Y": [100., 100., 110., 120., 130., 140.], "X": [100.] * 6}, index=check_dates)
check_z = pd.Series([-3., 0., 0., 0., 0., 0.], index=check_dates)
check_ledger, check_trades = backtest(check_px, check_z, check_dates, 1.,
                                     fee_bps=10., slippage_bps=0., borrow_rate=.02)
assert check_ledger.iloc[0].shares_y == 0
assert check_ledger.iloc[1].shares_y == 500
assert check_ledger.iloc[1].gross_pnl == 0
assert check_ledger.iloc[2].gross_pnl == 5000
np.testing.assert_allclose(check_ledger.trading_cost.sum(), 205.)
np.testing.assert_allclose(check_ledger.borrow_cost.sum(), 50_000 * .02 / 365)
np.testing.assert_allclose(check_ledger.equity.iloc[-1], 100_000 + 5000 - 205 - 50_000 * .02 / 365)
assert check_trades.iloc[0].entry_date == check_dates[1]
assert check_trades.iloc[0].exit_date == check_dates[2]

# Constant signals force final liquidation; a zero signal never trades.
final_ledger, final_trades = backtest(check_px, pd.Series(-3., index=check_dates), check_dates, 1., borrow_rate=0.)
assert final_trades.iloc[-1].exit_reason == "final liquidation"
assert final_ledger.iloc[-1].trading_cost > 0
assert final_ledger.iloc[-1].shares_y == final_ledger.iloc[-1].shares_x == 0
flat_ledger, flat_trades = backtest(check_px, pd.Series(0., index=check_dates), check_dates, 1.)
assert flat_trades.empty and (flat_ledger.equity == INITIAL_CAPITAL).all()
blocked_ledger, blocked_trades = backtest(check_px, check_z, check_dates, 1.,
                                          pd.Series(.1, index=check_dates), threshold=.5)
assert blocked_trades.empty

# Reversing the spread direction must reverse the leg signs and the price P&L.
short_ledger, short_trades = backtest(check_px, -check_z, check_dates, 1.,
                                     fee_bps=0., slippage_bps=0., borrow_rate=0.)
assert short_ledger.iloc[1].shares_y == -500
assert short_ledger.iloc[1].shares_x == 500
assert short_ledger.iloc[2].gross_pnl == -5000

# Perturb future prices: earlier features, predictions, and orders must be identical.
cut = len(pair_prices) - 30
changed_px = pair_prices.copy()
changed_px.iloc[cut:, 0] *= np.linspace(1.1, 1.5, len(changed_px) - cut)
changed_spread = changed_px.iloc[:, 0] - ALPHA - BETA * changed_px.iloc[:, 1]
changed_z = (changed_spread - changed_spread.shift(1).rolling(LOOKBACK).mean()) / changed_spread.shift(1).rolling(LOOKBACK).std().replace(0, np.nan)
changed_features = make_features(changed_px, volumes[[Y, X]], changed_spread, changed_z)
pd.testing.assert_frame_equal(features.iloc[:cut], changed_features.iloc[:cut])
changed_prob = pd.Series(np.nan, index=prices.index)
changed_complete = changed_features.notna().all(axis=1)
changed_prob.loc[changed_complete] = model.predict_proba(changed_features.loc[changed_complete])[:, positive_class]
np.testing.assert_allclose(probabilities.iloc[:cut], changed_prob.iloc[:cut], equal_nan=True)
changed_ledger, _ = backtest(changed_px, changed_z, test_index, BETA,
                            None if CHOSEN_THRESHOLD == 0 else changed_prob, CHOSEN_THRESHOLD)
pd.testing.assert_frame_equal(filtered.loc[filtered.index < prices.index[cut]],
                              changed_ledger.loc[changed_ledger.index < prices.index[cut]])
assert label_end.loc[train_mask].max() < validation_index[0]
print("PASS: manual P&L, signal delay, both-leg costs, borrow, final liquidation, flat/no-entry cases, short direction, and future-data invariance.")
'''
nb.cells.append(nbformat.v4.new_code_cell(checks))


def report(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Executing code cell {cell_index + 1}/{len(nb.cells)}", flush=True)


client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=report)
client.execute()
for output in nb.cells[-1].outputs:
    if output.output_type == "stream":
        print(output.text)
nb.cells = nb.cells[:original_count]
nbformat.validate(nb)
nbformat.write(nb, path)
print(f"Saved executed notebook: {path}")
