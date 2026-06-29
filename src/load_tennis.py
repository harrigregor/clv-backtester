"""
Tennis data loader — tennis-data.co.uk format (the standard free source).

Tennis is a cleaner modelling problem than football: two outcomes, no draw.
Each row is one match. Useful columns:
  Date, Winner, Loser           -> result (Winner always won)
  WRank, LRank                   -> ATP/WTA rankings (pre-match feature)
  WPts, LPts                     -> ranking points
  PSW, PSL                       -> Pinnacle CLOSING odds (winner/loser) — CLV benchmark
  B365W, B365L  / AvgW, AvgL     -> retail odds to bet against
  Surface, Series, Round         -> context

Point-in-time note: rankings are as-at the tournament, which is pre-match —
safe. We DON'T peek at the result; we randomly assign which player is
"player A" so the model can't trivially learn "row winner = winner".
"""
import pandas as pd
import numpy as np
from pathlib import Path


PIN = [("PSW", "PSL"), ("AvgW", "AvgL")]
RET = [("B365W", "B365L"), ("AvgW", "AvgL")]


def _first(df, cands):
    for c in cands:
        if all(x in df.columns for x in c):
            return c
    return None


def load_raw(paths):
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="latin-1")
        except Exception:
            df = pd.read_excel(p)
        df["_source_file"] = Path(p).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Winner", "Loser", "WRank", "LRank"])
    return df.sort_values("Date").reset_index(drop=True)


def build_dataset(paths, seed=42):
    """
    Returns modelling df. To avoid leakage we randomly relabel each match as
    (playerA vs playerB); y=1 if playerA won. Features are rank/points diffs
    oriented A-minus-B. Odds are oriented to A and B accordingly.
    """
    df = load_raw(paths)
    pin = _first(df, PIN)
    ret = _first(df, RET)
    if pin is None:
        raise ValueError("No Pinnacle closing odds (PSW/PSL) — CLV needs the close.")
    if ret is None:
        raise ValueError("No retail odds (B365W/L or AvgW/L) to bet against.")

    df = df.dropna(subset=[pin[0], pin[1], ret[0], ret[1], "WRank", "LRank"]).copy()
    rng = np.random.default_rng(seed)
    a_is_winner = rng.random(len(df)) < 0.5

    # orient everything to A / B
    rankA = np.where(a_is_winner, df["WRank"], df["LRank"]).astype(float)
    rankB = np.where(a_is_winner, df["LRank"], df["WRank"]).astype(float)
    ptsA = np.where(a_is_winner, df.get("WPts", np.nan), df.get("LPts", np.nan))
    ptsB = np.where(a_is_winner, df.get("LPts", np.nan), df.get("WPts", np.nan))

    odds_ret_A = np.where(a_is_winner, df[ret[0]], df[ret[1]]).astype(float)
    odds_ret_B = np.where(a_is_winner, df[ret[1]], df[ret[0]]).astype(float)
    odds_cls_A = np.where(a_is_winner, df[pin[0]], df[pin[1]]).astype(float)
    odds_cls_B = np.where(a_is_winner, df[pin[1]], df[pin[0]]).astype(float)

    out = pd.DataFrame({
        "date": df["Date"].values,
        "playerA": np.where(a_is_winner, df["Winner"], df["Loser"]),
        "playerB": np.where(a_is_winner, df["Loser"], df["Winner"]),
        "y": a_is_winner.astype(int),  # 1 if A won
        "rank_diff": rankB - rankA,    # positive => A higher ranked (lower number)
        "log_rank_ratio": np.log(rankB) - np.log(rankA),
        "pts_diff": (pd.Series(ptsA, dtype="float") - pd.Series(ptsB, dtype="float")).values,
        "odds_A": odds_ret_A, "odds_B": odds_ret_B,
        "close_A": odds_cls_A, "close_B": odds_cls_B,
    })
    out["pts_diff"] = out["pts_diff"].fillna(0.0)
    out = out.dropna(subset=["rank_diff", "odds_A", "odds_B", "close_A", "close_B"])
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    p = sys.argv[1:]
    if not p:
        print("Usage: python load_tennis.py file1.csv [file2.csv ...]")
        sys.exit(0)
    d = build_dataset(p)
    print(f"Loaded {len(d)} tennis matches with odds + closing line.")
    print(d.head())
