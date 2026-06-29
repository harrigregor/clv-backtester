"""
Tennis CLV engine — same philosophy as clv_engine.py but for 2 outcomes (A/B).
Kept separate for clarity; shares the no-vig + walk-forward + CLV approach.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["rank_diff", "log_rank_ratio", "pts_diff"]


def implied_probs_2(df, prefix):
    o = df[[f"{prefix}_A", f"{prefix}_B"]].values.astype(float)
    inv = 1.0 / o
    return inv / inv.sum(axis=1, keepdims=True)


def time_split_indices(df, train_frac=0.6, n_folds=5):
    n = len(df)
    start = int(n * train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        tr_end, te_end = bounds[i], bounds[i + 1]
        if te_end - tr_end < 5 or tr_end < 20:
            continue
        yield np.arange(0, tr_end), np.arange(tr_end, te_end)


def make_model():
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def run_backtest(df, edge_threshold=0.02, stake=1.0, train_frac=0.6, n_folds=5):
    df = df.sort_values("date").reset_index(drop=True)
    bets = []
    sides = ["A", "B"]
    for tr_idx, te_idx in time_split_indices(df, train_frac, n_folds):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        model = make_model()
        model.fit(train[FEATURES].values, train["y"].values)
        proba = model.predict_proba(test[FEATURES].values)  # [:,1] = P(A wins)
        pA = proba[:, 1]
        pB = proba[:, 0]
        model_p = np.column_stack([pA, pB])
        market = implied_probs_2(test, "odds")
        for j, (_, row) in enumerate(test.iterrows()):
            for k in range(2):
                edge = model_p[j, k] - market[j, k]
                if edge > edge_threshold:
                    took = row[f"odds_{sides[k]}"]
                    close = row[f"close_{sides[k]}"]
                    won = int((row["y"] == 1 and k == 0) or (row["y"] == 0 and k == 1))
                    pnl = stake * (took - 1) if won else -stake
                    clv_pct = (took / close - 1.0) * 100
                    bets.append({
                        "date": row["date"],
                        "match": f"{row['playerA']} v {row['playerB']}",
                        "pick": sides[k], "model_p": model_p[j, k],
                        "market_p": market[j, k], "edge": edge,
                        "took_odds": took, "close_odds": close,
                        "clv_pct": clv_pct, "won": won, "pnl": pnl,
                    })
    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        return {"n_bets": 0, "message": "No bets cleared threshold."}, bets_df
    n = len(bets_df)
    results = {
        "n_bets": n,
        "hit_rate_pct": round(bets_df["won"].mean() * 100, 2),
        "roi_pct": round(bets_df["pnl"].sum() / (n * stake) * 100, 2),
        "avg_clv_pct": round(bets_df["clv_pct"].mean(), 3),
        "clv_positive_rate_pct": round((bets_df["clv_pct"] > 0).mean() * 100, 2),
        "total_pnl_units": round(bets_df["pnl"].sum(), 2),
        "beat_closing_line": bool(bets_df["clv_pct"].mean() > 0),
        "verdict": _verdict(bets_df["clv_pct"].mean(), n),
    }
    return results, bets_df


def _verdict(avg_clv, n):
    if n < 50:
        return "INCONCLUSIVE — too few bets to trust."
    if avg_clv > 0.5:
        return ("POSITIVE CLV — beat the closing line. Rare. Validate on more "
                "data, size with fractional Kelly only.")
    if avg_clv > 0:
        return "MARGINAL — likely noise. Not a proven edge."
    return ("NO EDGE — did not beat the closing line. Expected, honest result. "
            "Do not bet this model.")
