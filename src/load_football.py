"""
Football data loader — Football-Data.co.uk CSV format.

Football-Data.co.uk gives free CSVs with bookmaker odds AND closing odds AND
Pinnacle (the sharp benchmark for CLV). Each row is one match.

Key columns we use (their naming):
  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR  -> result (FTR = H/D/A)
  B365H/D/A   -> Bet365 opening-ish odds (a retail book)
  PSH/D/A     -> Pinnacle CLOSING odds  (sharp; our CLV benchmark)
  PSCH/D/A    -> Pinnacle closing (newer files use PSC*)
  AvgH/D/A    -> market average closing odds

CRITICAL on point-in-time correctness:
  The features we build for a match must use ONLY information available
  BEFORE kickoff. We compute rolling team form from PAST matches only,
  shifting so the current match is never included in its own features.
  This is the same discipline as not leaking the test set.
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Pinnacle closing columns differ across file eras; try newest first.
PINNACLE_CLOSE = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")]
RETAIL = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]


def _first_present(df, candidates):
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return None


def load_raw(csv_paths):
    """Load and concatenate one or more Football-Data.co.uk CSVs."""
    frames = []
    for p in csv_paths:
        # encoding varies; latin-1 is the safe bet for these files
        df = pd.read_csv(p, encoding="latin-1")
        df["_source_file"] = Path(p).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Date parsing — files use dd/mm/yy or dd/mm/yyyy
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _rolling_form(df, n=5):
    """
    Build pre-match form features using ONLY past matches.
    Returns per-row: home_form, away_form (avg points over last n games),
    home_gd, away_gd (avg goal difference over last n games).
    Uses .shift(1) inside each team so the current match is excluded.
    """
    df = df.reset_index(drop=True).copy()
    # long format: two rows per match (home perspective, away perspective)
    recs = []
    for idx in range(len(df)):
        r = df.iloc[idx]
        hp = 3 if r.FTR == "H" else (1 if r.FTR == "D" else 0)
        ap = 3 if r.FTR == "A" else (1 if r.FTR == "D" else 0)
        gd = r.FTHG - r.FTAG
        recs.append({"midx": idx, "side": "home", "Date": r.Date,
                     "team": r.HomeTeam, "pts": hp, "gd": gd})
        recs.append({"midx": idx, "side": "away", "Date": r.Date,
                     "team": r.AwayTeam, "pts": ap, "gd": -gd})
    long = pd.DataFrame(recs).sort_values(["team", "Date", "midx"])
    long["form"] = (long.groupby("team")["pts"]
                    .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))
    long["form_gd"] = (long.groupby("team")["gd"]
                       .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

    home = long[long["side"] == "home"].set_index("midx")
    away = long[long["side"] == "away"].set_index("midx")
    df["home_form"] = home["form"].reindex(df.index).values
    df["home_gd"] = home["form_gd"].reindex(df.index).values
    df["away_form"] = away["form"].reindex(df.index).values
    df["away_gd"] = away["form_gd"].reindex(df.index).values
    return df


def build_dataset(csv_paths, form_window=5):
    """
    Returns a clean modelling dataframe with:
      - point-in-time features (form, goal diff)
      - target y in {0:H, 1:D, 2:A}
      - retail odds (what you'd bet at) and Pinnacle closing odds (CLV benchmark)
    """
    df = load_raw(csv_paths)

    pin = _first_present(df, PINNACLE_CLOSE)
    ret = _first_present(df, RETAIL)
    if pin is None:
        raise ValueError("No Pinnacle closing odds (PSC*/PS*) in these files — "
                         "CLV needs the closing line. Download files that include them.")
    if ret is None:
        raise ValueError("No retail odds (B365*/Avg*) found to bet against.")

    df = _rolling_form(df, n=form_window)

    out = pd.DataFrame({
        "date": df["Date"],
        "home": df["HomeTeam"],
        "away": df["AwayTeam"],
        "y": df["FTR"].map({"H": 0, "D": 1, "A": 2}),
        # features (all pre-match)
        "home_form": df["home_form"],
        "away_form": df["away_form"],
        "home_gd": df["home_gd"],
        "away_gd": df["away_gd"],
        "form_diff": df["home_form"] - df["away_form"],
        "gd_diff": df["home_gd"] - df["away_gd"],
        # odds you would actually bet at (retail)
        "odds_H": df[ret[0]], "odds_D": df[ret[1]], "odds_A": df[ret[2]],
        # Pinnacle CLOSING odds — the CLV benchmark
        "close_H": df[pin[0]], "close_D": df[pin[1]], "close_A": df[pin[2]],
    })
    out = out.dropna(subset=["home_form", "away_form",
                             "odds_H", "odds_D", "odds_A",
                             "close_H", "close_D", "close_A", "y"])
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python load_football.py file1.csv [file2.csv ...]")
        sys.exit(0)
    d = build_dataset(paths)
    print(f"Loaded {len(d)} matches with full odds + closing line.")
    print(d.head())
