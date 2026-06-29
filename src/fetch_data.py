"""
Download real historical data into data/.

Football: football-data.co.uk — one CSV per league per season.
  URL pattern: https://www.football-data.co.uk/mmz4281/{SEASON}/{LEAGUE}.csv
  SEASON is 4 digits, e.g. 2526 = 2025/26. LEAGUE e.g. E0 = Premier League.
  Already contains B365* retail odds and PSC* Pinnacle CLOSING odds.

Tennis: tennis-data.co.uk — one Excel/CSV per tour per year.
  URL pattern: http://www.tennis-data.co.uk/{YEAR}/{YEAR}.xlsx (ATP)
  Contains PSW/PSL Pinnacle closing odds and B365W/L retail.

This runs inside GitHub Actions (which has open network access), so the
weekly job pulls fresh data automatically — the current-season files keep
growing as new results land.

Run:  python src/fetch_data.py
"""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)

# --- choose what to pull -------------------------------------------------
# A few recent seasons of a few big leagues = a solid, honest sample.
FOOTBALL_SEASONS = ["2324", "2425", "2526"]          # last 3 seasons
FOOTBALL_LEAGUES = ["E0", "E1", "D1", "I1", "SP1"]   # EPL, Championship, Bundesliga, Serie A, La Liga
FB_BASE = "https://www.football-data.co.uk/mmz4281"

# Tennis ATP yearly files.
TENNIS_YEARS = ["2024", "2025", "2026"]
TN_BASE = "http://www.tennis-data.co.uk"

HEADERS = {"User-Agent": "Mozilla/5.0 (clv-backtester data fetch)"}


def _get(url, dest):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 200:  # too small to be a real file
            print(f"  skip (empty/missing): {url}")
            return False
        dest.write_bytes(data)
        print(f"  ok: {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  skip ({e.__class__.__name__}): {url}")
        return False


def fetch_football():
    print("Football (football-data.co.uk):")
    got = []
    for season in FOOTBALL_SEASONS:
        for league in FOOTBALL_LEAGUES:
            url = f"{FB_BASE}/{season}/{league}.csv"
            dest = DATA / f"football_{league}_{season}.csv"
            if _get(url, dest):
                got.append(dest)
    return got


def fetch_tennis():
    print("Tennis (tennis-data.co.uk):")
    got = []
    for year in TENNIS_YEARS:
        # try xlsx first, then csv fallback
        for ext in ("xlsx", "csv"):
            url = f"{TN_BASE}/{year}/{year}.{ext}"
            dest = DATA / f"tennis_{year}.{ext}"
            if _get(url, dest):
                got.append(dest)
                break
    return got


if __name__ == "__main__":
    fb = fetch_football()
    tn = fetch_tennis()
    print(f"\nDownloaded {len(fb)} football files, {len(tn)} tennis files into {DATA}")
    if not fb and not tn:
        print("WARNING: nothing downloaded. Network blocked? Falling back to "
              "synthetic data is recommended (python src/make_synthetic.py).")
        sys.exit(0)
