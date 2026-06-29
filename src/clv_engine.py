"""
CLV backtester — the part that answers the only question that matters:
does the model beat the closing line?

The logic, plainly:
  1. Convert odds to probabilities and REMOVE THE VIG (bookmaker margin).
     Raw 1/odds sums to >1 because of the margin; we normalise to a true
     probability distribution. The Pinnacle no-vig close is our "truth proxy".
  2. Train a model on PAST matches only, predict CALIBRATED probabilities
     for future matches (strict time-ordered split — never train on the future).
  3. For every match, the model would "bet" the outcome where its probability
     exceeds the retail implied probability by some edge threshold.
  4. CLV = did we get a better price than the closing line?
     Specifically: for bets we'd place, compare the retail odds we took vs the
     Pinnacle closing odds. Positive average CLV is the single best evidence
     of a real, durable edge. If CLV <= 0, no staking trick saves you.

This file is deliberately model-light. Calibration and honest evaluation
matter far more than a fancy model, so we use logistic regression with
isotonic calibration by default — interpretable and hard to overfit.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


FEATURES = ["home_form", "away_form", "home_gd", "away_gd", "form_diff", "gd_diff"]
OUTCOMES = ["H", "D", "A"]  # indices 0,1,2


def devig(odds_triple):
    """Convert a triple of decimal odds to a no-vig probability distribution."""
    inv = 1.0 / np.asarray(odds_triple, dtype=float)
    return inv / inv.sum()


def implied_probs(df, prefix):
    """Per-row no-vig probabilities from odds columns prefix_H/D/A."""
    o = df[[f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]].values.astype(float)
    inv = 1.0 / o
    return inv / inv.sum(axis=1, keepdims=True)


def time_split_indices(df, train_frac=0.6, n_folds=5):
    """
    Walk-forward expanding-window splits. Each fold trains on everything
    before a cutoff and tests on the next chunk — never peeks forward.
    """
    n = len(df)
    start = int(n * train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        tr_end = bounds[i]
        te_end = bounds[i + 1]
        if te_end - tr_end < 5 or tr_end < 20:
            continue
        yield np.arange(0, tr_end), np.arange(tr_end, te_end)


def make_model():
    """Calibrated logistic regression. Calibration is the point, not accuracy."""
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    # isotonic calibration so a predicted 0.30 actually means 30%
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def run_backtest(df, edge_threshold=0.02, stake=1.0, train_frac=0.6, n_folds=5):
    """
    Returns a dict of results + a per-bet dataframe.

    edge_threshold: only bet when model prob - retail implied prob > this.
                    (a real edge must clear noise; 2% is a sane default)
    """
    df = df.sort_values("date").reset_index(drop=True)
    bets = []

    for tr_idx, te_idx in time_split_indices(df, train_frac, n_folds):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        model = make_model()
        model.fit(train[FEATURES].values, train["y"].values)
        proba = model.predict_proba(test[FEATURES].values)  # calibrated

        retail_imp = implied_probs(test, "odds")  # what the market charges

        for j, (_, row) in enumerate(test.iterrows()):
            for k in range(3):  # H, D, A
                model_p = proba[j, k]
                market_p = retail_imp[j, k]
                edge = model_p - market_p
                if edge > edge_threshold:
                    took_odds = row[f"odds_{OUTCOMES[k]}"]
                    close_odds = row[f"close_{OUTCOMES[k]}"]
                    won = int(row["y"] == k)
                    pnl = stake * (took_odds - 1) if won else -stake
                    # CLV: positive if we beat the closing price.
                    # (took a higher price than it closed at = value)
                    clv_pct = (took_odds / close_odds - 1.0) * 100
                    bets.append({
                        "date": row["date"], "match": f"{row['home']} v {row['away']}",
                        "pick": OUTCOMES[k], "model_p": model_p, "market_p": market_p,
                        "edge": edge, "took_odds": took_odds, "close_odds": close_odds,
                        "clv_pct": clv_pct, "won": won, "pnl": pnl,
                    })

    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        return {"n_bets": 0, "message": "No bets cleared the edge threshold. "
                "Lower edge_threshold or check data."}, bets_df

    n = len(bets_df)
    roi = bets_df["pnl"].sum() / (n * stake) * 100
    avg_clv = bets_df["clv_pct"].mean()
    clv_positive_rate = (bets_df["clv_pct"] > 0).mean() * 100
    beat_close = avg_clv > 0

    results = {
        "n_bets": n,
        "hit_rate_pct": round(bets_df["won"].mean() * 100, 2),
        "roi_pct": round(roi, 2),
        "avg_clv_pct": round(avg_clv, 3),
        "clv_positive_rate_pct": round(clv_positive_rate, 2),
        "total_pnl_units": round(bets_df["pnl"].sum(), 2),
        "beat_closing_line": bool(beat_close),
        "verdict": _verdict(avg_clv, n),
    }
    return results, bets_df


def _verdict(avg_clv, n):
    if n < 50:
        return ("INCONCLUSIVE — too few bets to trust. Need more data/seasons "
                "before believing any number here.")
    if avg_clv > 0.5:
        return ("POSITIVE CLV — the model beat the closing line on average. "
                "This is the rare 'there might be a real edge' result. Validate "
                "on more seasons and other leagues before risking money, and "
                "size with FRACTIONAL Kelly only.")
    if avg_clv > 0:
        return ("MARGINAL — slightly positive CLV but small. Likely noise. "
                "Treat as 'no proven edge' until it holds across much more data.")
    return ("NO EDGE — the model did not beat the closing line. This is the "
            "expected, honest result for most attempts. No staking strategy "
            "rescues negative CLV. Do not bet this model.")


def calibration_report(df, train_frac=0.6, n_folds=5, bins=10):
    """
    Reliability check: when the model says X%, does it happen X% of the time?
    Returns a dataframe of (predicted_bucket, actual_freq, count).
    If this is way off, Kelly sizing will lie to you.
    """
    df = df.sort_values("date").reset_index(drop=True)
    preds, actuals = [], []
    for tr_idx, te_idx in time_split_indices(df, train_frac, n_folds):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        model = make_model()
        model.fit(train[FEATURES].values, train["y"].values)
        proba = model.predict_proba(test[FEATURES].values)
        for j in range(len(test)):
            for k in range(3):
                preds.append(proba[j, k])
                actuals.append(int(test.iloc[j]["y"] == k))
    preds, actuals = np.array(preds), np.array(actuals)
    edges = np.linspace(0, 1, bins + 1)
    rep = []
    for i in range(bins):
        m = (preds >= edges[i]) & (preds < edges[i + 1])
        if m.sum() > 0:
            rep.append({"bucket": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                        "predicted": round(preds[m].mean(), 3),
                        "actual": round(actuals[m].mean(), 3),
                        "count": int(m.sum())})
    return pd.DataFrame(rep)
