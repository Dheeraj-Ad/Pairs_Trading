"""Build the executable five-stage walkthrough without duplicating model code."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md('''
# ML-enhanced pairs trading: daily report workflow

This notebook implements the five modules in the project report using the same
Python modules as the application's **Daily report** workspace. It uses synthetic
daily data by default, records losses as well as profits, and never changes a
paper account or submits broker orders. For equations, assumptions, report page
mapping and custom-model changes, read [REPORT_ALIGNMENT.md](docs/REPORT_ALIGNMENT.md).

Run all cells from this repository's Python environment. To use your own daily
close CSV, set `CSV_PATH` below. No network download is required for the default run.
''')
code('''
from pathlib import Path
import sys, json, uuid
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
assert (ROOT / 'ml' / 'report').is_dir(), 'Open this notebook from the repository root.'
sys.path.insert(0, str(ROOT))
from scripts.run_daily_report import synthetic_prices
from ml.report.data import prepare
from ml.report.discovery import discover
from ml.report.models import train_pair
from ml.report.backtest import backtest
from ml.report.explain import explain_trades
from ml.report.pipeline import artifacts

CSV_PATH = None  # e.g. Path('data/closes.csv'); date first, one column per stock
source = 'Synthetic generated daily prices' if CSV_PATH is None else 'User-supplied CSV; provenance not verified'
raw = synthetic_prices() if CSV_PATH is None else pd.read_csv(CSV_PATH, index_col=0)
output = ROOT / 'outputs' / ('daily-notebook-' + str(uuid.uuid4()))
print(source, '|', len(raw), 'daily rows |', len(raw.columns), 'stocks')
''')
md('''
## 1. Clean daily data and fit PCA

Invalid/missing rows are dropped without forward filling. Formation-only ADF
checks screen log returns. Five-component covariance PCA fits formation data only;
future returns are transformed using that frozen fit.
''')
code('''
prices, returns, factors, data = prepare(raw)
display(pd.DataFrame(data['stationarity']))
display(pd.DataFrame({'component': range(1, 6), 'variance': data['explained_variance'],
                      'cumulative': data['cumulative_variance']}))
''')
md('''
## 2. OPTICS discovery and statistical verification

Stock loadings define clusters. Noise remains unassigned. Within-cluster pairs
must pass formation-only cointegration/FDR, Hurst and half-life checks. No pair
is forced through the screen. At most eight are modeled, ranked by formation FDR.
''')
code('''
clusters, candidates = discover(prices, data)
verified = sorted([p for p in candidates if p['verified']], key=lambda p: (p['fdr'], p['id']))
display(pd.DataFrame(clusters))
display(pd.DataFrame(candidates))
print(len(verified), 'verified pairs')
''')
md('''
## 3–4. Regression, classification and asymmetric costs

Use rolling Ridge hedge features, OLS/Ridge/Lasso forecasts, Lasso selection,
Logistic/linear-SVM/RBF-SVM/KNN baselines and decision-tree/RF/boosting comparisons.
Six-session purges separate supervised partitions. Ensemble and direction-specific
threshold selection use validation outcomes; test outcomes remain held out.
''')
code('''
bundles, failures = {}, []
for pair in verified[:8]:
    try:
        bundles[pair['id']] = train_pair(prices, pair)
    except ValueError as exc:
        failures.append({'pair': pair['id'], 'error': str(exc)})
for key, bundle in bundles.items():
    print(key, bundle['report']['selected']['model'], bundle['report']['selected']['parameters'])
    display(pd.DataFrame(bundle['report']['regression']))
    display(pd.json_normalize(bundle['report']['models']))
display(pd.DataFrame(failures))
''')
md('''
## 5. Delayed execution, reconciliation and SHAP

Signal at close t, execute at next daily close t+1. Equal-dollar targets are rounded
to integers. Both legs pay fees, half-spread and slippage; shorts accrue calendar-time
borrow. The final test date liquidates remaining positions. SHAP explains each actual
entry score using only training background observations. A losing result remains a loss.
''')
code('''
result = backtest(prices, bundles)
result['sensitivity'] = [dict(commission_bps=bps, **backtest(prices, bundles, fee=bps/10000)['metrics'])
                         for bps in [0, 1, 3, 5, 10]]
explanations = explain_trades(bundles, result['trades'])
assert abs(sum(t['pnl'] for t in result['trades']) - result['metrics']['net_pnl']) < 1e-6
for e in explanations:
    assert abs(e['base_value'] + sum(e['contributions'].values()) - e['score']) < 1e-6
display(pd.DataFrame([result['metrics']]))
display(pd.DataFrame(result['trades']))
display(pd.DataFrame(explanations))
''')
md('''
## Export and inspect the module figures

These are plots of the calculations above, not invented illustrations. Empty
results stay empty. The CSVs contain the full candidate, model, trade and explanation
records. The application presents the same artifacts with source-scoped persistence.
''')
code('''
files = artifacts(output, prices, returns, factors, data, clusters, candidates, bundles, result, explanations)
(output / 'notebook_summary.json').write_text(json.dumps(dict(source=source, data=data,
    verified=len(verified), modeled=len(bundles), failures=failures, backtest=result,
    explanations=explanations), allow_nan=False, indent=2), encoding='utf-8')
for name in files:
    if name.endswith('.png'):
        display(Image(filename=str(output / name), width=900))
print('Saved', len(files), 'module artifacts to', output)
''')
md('''
## Continue your research

Change model candidates in `ml/report/models.py`, causal features in `features.py`,
or a documented execution assumption in `backtest.py`. Re-run the chronological
tests before comparing experiments. Reusing this holdout repeatedly makes it part
of model development; reserve fresh out-of-sample data for an independent evaluation.
The five-minute terminal and paper-outcome learner remain separate systems.
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}})
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'daily_report.ipynb')
print('Built daily_report.ipynb')
