# CLV Backtester

A closing-line-value backtester for football and tennis. It answers one question honestly: **does a predictive model beat the closing line?** If it does, there may be a real, durable edge worth pursuing. If it doesn't, no stake-sizing strategy rescues it — and that's the expected, honest result for most attempts.

This exists because "value betting" only works if your probabilities are genuinely better than the market's *and* that edge isn't already priced in. The single best test of that, before risking any money, is closing-line value: did your bets get a better price than the eventual closing line? Beat the close consistently and you'll beat the bookmaker over time, essentially by definition.

## What it does

- Loads free historical odds data (Football-Data.co.uk / tennis-data.co.uk formats), which include bookmaker odds **and** Pinnacle closing odds — the sharp benchmark.
- Builds **point-in-time-safe** features (rolling form / rankings using only pre-match information — no leakage).
- Trains a **calibrated** model (calibration matters more than accuracy for honest staking) with strict walk-forward, time-ordered splits — never trains on the future.
- Measures **average CLV**, beat-the-close rate, calibration, and a backtest equity curve.
- Renders it all in a self-contained HTML dashboard (`docs/index.html`), regenerated weekly by GitHub Actions.

## Quick start

```bash
pip install -r requirements.txt
python src/make_synthetic.py     # creates sample data so it runs out of the box
python src/run.py                # runs both backtests -> results/results.json
```

Then open the dashboard. Because browsers can't read local JSON via `file://`, serve the folder:

```bash
python -m http.server 8000
# visit http://localhost:8000/docs/
```

The sample data is a synthetic, near-efficient market, so the verdict is correctly **NO EDGE**. That's the engine working: if it claimed a fat edge on data designed to have none, *that* would be the bug.

## Using real data

The synthetic generator is only a stand-in. For a real test:

**Football** — download CSVs from football-data.co.uk (e.g. the Premier League file `E0.csv` for a season). They already contain `B365*` retail odds and `PSC*` Pinnacle closing odds.

```bash
python src/run.py --football data/E0.csv data/E0_prev.csv
```

**Tennis** — download the yearly archives from tennis-data.co.uk (they include `PSW/PSL` Pinnacle closing odds and `B365W/L`).

```bash
python src/run.py --tennis data/2025.csv
```

More seasons = more trustworthy. Fewer than ~50 qualifying bets is inconclusive by design.

## Auto-update (GitHub Actions)

`.github/workflows/weekly-backtest.yml` reruns the backtest every Monday and commits the refreshed `results/results.json`, so the dashboard stays current hands-off. By default it regenerates synthetic data; to point it at real data, edit the **DATA STEP** in that workflow to `curl` the CSVs you want (and add any paid API keys as repo **Secrets**).

To publish the dashboard for free: repo **Settings → Pages → Source: deploy from branch → `main` / `docs`**.

## Honest expectations

- Most models **do not** beat the closing line. That is the normal outcome and learning it cheaply is the point.
- A positive CLV result is a *starting point* for more validation (more seasons, more leagues), **never** a green light to bet large.
- If you ever act on a real edge, size with **fractional Kelly** (¼ or ½) and hard loss limits. A backtested edge always looks more certain than it is.
- This measures edge; it doesn't place bets and guarantees nothing. Gamble only what you can afford to lose.

## Layout

```
src/
  load_football.py        # Football-Data.co.uk loader + point-in-time features
  load_tennis.py          # tennis-data.co.uk loader (2-outcome)
  clv_engine.py           # core: no-vig probs, walk-forward, CLV (football, 3-way)
  clv_engine_tennis.py    # 2-outcome variant
  make_synthetic.py       # sample data in the real formats
  run.py                  # entry point -> results/results.json
docs/index.html           # dashboard (GitHub Pages)
results/results.json      # generated output
.github/workflows/        # weekly auto-rerun
```
