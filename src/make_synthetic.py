"""
Generate synthetic data in the EXACT column format of the real free sources,
so the pipeline can be tested before you download real files. The synthetic
data has a known weak signal + realistic bookmaker margins, so we can confirm
the engine behaves (and importantly, that it does NOT manufacture a fake edge
out of an efficient market).
"""
import numpy as np
import pandas as pd


def make_football(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(20)]
    strength = {t: rng.normal(0, 1) for t in teams}
    dates = pd.date_range("2021-08-01", periods=n, freq="6h")
    rows = []
    for d in dates:
        h, a = rng.choice(teams, 2, replace=False)
        # true outcome probs from strengths + home advantage
        sh = strength[h] + 0.3
        sa = strength[a]
        eh, ea = np.exp(sh), np.exp(sa)
        pH = eh / (eh + ea) * 0.8
        pA = ea / (eh + ea) * 0.8
        pD = 1 - pH - pA
        probs = np.array([pH, pD, pA])
        probs = probs / probs.sum()
        outcome = rng.choice(["H", "D", "A"], p=probs)
        # market odds = true probs + noise, with ~5% margin (efficient-ish market)
        noise = rng.normal(0, 0.03, 3)
        mp = np.clip(probs + noise, 0.02, 0.96)
        mp = mp / mp.sum()
        margin = 1.05
        odds = 1.0 / (mp * margin)
        # closing (Pinnacle) slightly sharper / lower margin
        cmargin = 1.02
        codds = 1.0 / (mp * cmargin)
        hg = rng.poisson(1.5 if outcome != "A" else 0.8)
        ag = rng.poisson(1.5 if outcome != "H" else 0.8)
        rows.append({
            "Date": d.strftime("%d/%m/%Y"), "HomeTeam": h, "AwayTeam": a,
            "FTHG": hg, "FTAG": ag, "FTR": outcome,
            "B365H": round(odds[0], 2), "B365D": round(odds[1], 2), "B365A": round(odds[2], 2),
            "PSCH": round(codds[0], 2), "PSCD": round(codds[1], 2), "PSCA": round(codds[2], 2),
        })
    return pd.DataFrame(rows)


def make_tennis(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    players = [f"Player{i}" for i in range(60)]
    rank = {p: i + 1 for i, p in enumerate(rng.permutation(players))}
    pts = {p: max(100, 5000 - rank[p] * 70 + rng.normal(0, 200)) for p in players}
    dates = pd.date_range("2021-01-01", periods=n, freq="8h")
    rows = []
    for d in dates:
        p1, p2 = rng.choice(players, 2, replace=False)
        r1, r2 = rank[p1], rank[p2]
        # better rank (lower number) more likely to win
        s = (r2 - r1) / 50.0
        pWin1 = 1 / (1 + np.exp(-s))
        winner, loser = (p1, p2) if rng.random() < pWin1 else (p2, p1)
        pw = pWin1 if winner == p1 else 1 - pWin1
        mp_w = np.clip(pw + rng.normal(0, 0.03), 0.05, 0.95)
        margin = 1.05
        oddsW = 1 / (mp_w * margin / (mp_w + (1 - mp_w)))
        oddsL = 1 / ((1 - mp_w) * margin / (mp_w + (1 - mp_w)))
        codraw = 1.02
        coW = 1 / (mp_w * codraw / (mp_w + (1 - mp_w)))
        coL = 1 / ((1 - mp_w) * codraw / (mp_w + (1 - mp_w)))
        rows.append({
            "Date": d.strftime("%d/%m/%Y"), "Winner": winner, "Loser": loser,
            "WRank": rank[winner], "LRank": rank[loser],
            "WPts": round(pts[winner]), "LPts": round(pts[loser]),
            "B365W": round(oddsW, 2), "B365L": round(oddsL, 2),
            "PSW": round(coW, 2), "PSL": round(coL, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    data = Path(__file__).parent.parent / "data"
    data.mkdir(exist_ok=True)
    make_football().to_csv(data / "sample_football.csv", index=False)
    make_tennis().to_csv(data / "sample_tennis.csv", index=False)
    print(f"Wrote {data}/sample_football.csv and sample_tennis.csv")
