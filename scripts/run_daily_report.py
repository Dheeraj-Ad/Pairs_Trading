"""Run the five-stage daily research workflow independently of the paper account."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.market import PAIRS, demo_history
from ml.report.pipeline import run


def synthetic_prices():
    columns = {}
    for i, pair in enumerate(PAIRS):
        frame = demo_history(i)
        for leg in ('y', 'x'):
            columns[pair[leg]] = frame[leg]
    prices = pd.DataFrame(columns)
    prices.index = pd.to_datetime(prices.index, utc=True)
    return prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Daily closes: first column date, remaining columns stock symbols.')
    parser.add_argument('--output', type=Path, default=Path('outputs/daily-report'))
    args = parser.parse_args()
    prices = pd.read_csv(args.csv, index_col=0) if args.csv else synthetic_prices()
    source = f'User-supplied daily CSV: {args.csv.name}; provenance not independently verified' if args.csv else 'Synthetic generated daily prices; not exchange history'
    # Avoid stale artifacts from a previous universe being presented as this run.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output directory.')
    result = run(prices, args.output, source, lambda message: print(message, flush=True))
    print(json.dumps(dict(source=source, verified=result['verified'], modeled=result['modeled'],
                         metrics=result['backtest']['metrics'], failures=result['failures']), indent=2))
    print(f'Report: {args.output.resolve() / "report.json"}')


if __name__ == '__main__':
    main()
