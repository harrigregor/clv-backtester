"""
Main entry point. Runs football + tennis CLV backtests on whatever data is
in data/, writes results/results.json for the dashboard, and prints a verdict.

Usage:
  python src/run.py                 # uses sample_*.csv if present
  python src/run.py --football path1.csv path2.csv
  python src/run.py --tennis path.csv
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

import load_football
import load_tennis
import clv_engine
import clv_engine_tennis

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def run_football(paths):
    df = load_football.build_dataset(paths)
    res, bets = clv_engine.run_backtest(df)
    calib = clv_engine.calibration_report(df)
    res["calibration"] = calib.to_dict(orient="records")
    res["sport"] = "football"
    res["matches_loaded"] = len(df)
    if not bets.empty:
        res["sample_bets"] = bets.head(15).assign(
            date=bets["date"].astype(str)).to_dict(orient="records")
        res["equity_curve"] = bets.sort_values("date")["pnl"].cumsum().round(2).tolist()
    return res


def run_tennis(paths):
    df = load_tennis.build_dataset(paths)
    res, bets = clv_engine_tennis.run_backtest(df)
    res["sport"] = "tennis"
    res["matches_loaded"] = len(df)
    if not bets.empty:
        res["sample_bets"] = bets.head(15).assign(
            date=bets["date"].astype(str)).to_dict(orient="records")
        res["equity_curve"] = bets.sort_values("date")["pnl"].cumsum().round(2).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--football", nargs="*", default=None)
    ap.add_argument("--tennis", nargs="*", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "results": []}

    fb = args.football or ([str(DATA / "sample_football.csv")]
                           if (DATA / "sample_football.csv").exists() else None)
    tn = args.tennis or ([str(DATA / "sample_tennis.csv")]
                         if (DATA / "sample_tennis.csv").exists() else None)

    if fb:
        try:
            print(f"Running football on {fb} ...")
            out["results"].append(run_football(fb))
        except Exception as e:
            out["results"].append({"sport": "football", "error": str(e)})
            print(f"  football error: {e}")
    if tn:
        try:
            print(f"Running tennis on {tn} ...")
            out["results"].append(run_tennis(tn))
        except Exception as e:
            out["results"].append({"sport": "tennis", "error": str(e)})
            print(f"  tennis error: {e}")

    (RESULTS / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'results.json'}")
    for r in out["results"]:
        if "error" in r:
            print(f"  [{r['sport']}] ERROR: {r['error']}")
        else:
            print(f"  [{r['sport']}] bets={r['n_bets']} "
                  f"avg_CLV={r.get('avg_clv_pct')}% ROI={r.get('roi_pct')}%")
            print(f"     {r.get('verdict','')}")


if __name__ == "__main__":
    main()
