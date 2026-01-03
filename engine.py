import os
import re
import statistics
import time

import pandas as pd
import requests
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

CACHE_DIR = ".cache"
RATINGS_CACHE_TTL_SEC = 6 * 60 * 60


def _ensure_cache_dir() -> None:
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, name)


def _is_cache_fresh(path: str, ttl_sec: int) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age <= ttl_sec


def normalize_team_name(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace(" NCAA", "")
    s = s.replace("&", "and")
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[\.,/()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def build_alias_map(team_names: list[str]) -> dict[str, str]:
    alias_to_canonical: dict[str, str] = {}
    for team in team_names:
        canon = team
        base = normalize_team_name(team)

        variants = {base}

        if " state" in base:
            variants.add(base.replace(" state", " st"))

        if base.startswith("saint "):
            variants.add(base.replace("saint ", "st "))

        variants.add(base.replace("north ", "n "))
        variants.add(base.replace("south ", "s "))
        variants.add(base.replace("east ", "e "))
        variants.add(base.replace("west ", "w "))

        variants.add(base.replace(" university", ""))

        for v in variants:
            alias_to_canonical[v] = canon

    return alias_to_canonical


MANUAL_FIXES = {
    "nc state": "North Carolina State",
    "ole miss": "Mississippi",
    "uconn": "Connecticut",
    "umass": "Massachusetts",
    "miami": "Miami (FL)",
}


def get_ncaa_ratings_2026() -> pd.DataFrame:
    url = "https://www.sports-reference.com/cbb/seasons/men/2026-ratings.html"
    fp = _cache_path("cbb_2026_ratings.csv")

    if _is_cache_fresh(fp, RATINGS_CACHE_TTL_SEC):
        return pd.read_csv(fp)

    dfs = pd.read_html(url, attrs={"id": "ratings"})
    df = dfs[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df[df["School"] != "School"].copy()
    df["SRS"] = pd.to_numeric(df["SRS"], errors="coerce")

    out = df[["School", "SRS"]].dropna().copy()
    out.columns = ["Team", "Rating"]
    out["Team"] = out["Team"].astype(str).str.replace(" NCAA", "", regex=False)

    out.to_csv(fp, index=False)
    return out


class TeamLookup:
    def __init__(self, ratings_df: pd.DataFrame, fuzzy_min_score: int = 93):
        self.fuzzy_min_score = fuzzy_min_score
        self.team_to_rating = dict(zip(ratings_df["Team"], ratings_df["Rating"]))
        self.team_names = list(self.team_to_rating.keys())
        self.alias_map = build_alias_map(self.team_names)
        self.alias_keys = list(self.alias_map.keys())

    def get(self, vegas_name: str):
        n = normalize_team_name(vegas_name)

        if n in MANUAL_FIXES:
            canon = MANUAL_FIXES[n]
            if canon in self.team_to_rating:
                return float(self.team_to_rating[canon]), canon, "manual"

        if n in self.alias_map:
            canon = self.alias_map[n]
            return float(self.team_to_rating[canon]), canon, "alias"

        match = rf_process.extractOne(n, self.alias_keys, scorer=rf_fuzz.ratio)
        if match:
            alias, score, _ = match
            if score >= self.fuzzy_min_score:
                canon = self.alias_map[alias]
                return float(self.team_to_rating[canon]), canon, f"fuzzy{int(score)}"

        return None, None, "miss"


def fetch_spreads(api_key: str, sport: str = "basketball_ncaab", regions: str = "us"):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets=spreads&oddsFormat=american"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "message" in data:
        raise RuntimeError(data["message"])
    return data


def consensus_home_spread(game: dict, home_team: str):
    points = []
    for bm in game.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") != "spreads":
                continue
            for o in m.get("outcomes", []) or []:
                if o.get("name") == home_team and "point" in o:
                    try:
                        points.append(float(o["point"]))
                    except Exception:
                        pass
    if not points:
        return None, 0
    return statistics.median(points), len(points)


def model_home_spread_from_srs(home_rating: float, away_rating: float, home_court_adv: float) -> float:
    expected_margin = (home_rating - away_rating) + home_court_adv
    return -expected_margin


def compute_edges(
    api_key: str,
    home_court_adv: float = 3.25,
    edge_threshold: float = 3.0,
    fuzzy_min_score: int = 93,
    min_books: int = 3,
) -> pd.DataFrame:
    ratings = get_ncaa_ratings_2026()
    lookup = TeamLookup(ratings, fuzzy_min_score=fuzzy_min_score)
    games = fetch_spreads(api_key)

    rows = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")

        hr, _, _ = lookup.get(home)
        ar, _, _ = lookup.get(away)
        if hr is None or ar is None:
            continue

        vegas_home, books = consensus_home_spread(g, home)
        if vegas_home is None or books < min_books:
            continue

        model_home = model_home_spread_from_srs(hr, ar, home_court_adv)
        diff = model_home - vegas_home

        pick = ""
        if abs(diff) >= edge_threshold:
            pick = home if model_home < vegas_home else away

        rows.append({
            "Home": home,
            "Away": away,
            "ModelHome": round(model_home, 2),
            "VegasHome": round(vegas_home, 2),
            "Diff": round(diff, 2),
            "AbsDiff": round(abs(diff), 2),
            "Books": books,
            "Pick": pick,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["AbsDiff", "Books"], ascending=[False, False]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Set ODDS_API_KEY to your The Odds API key.")

    edges = compute_edges(api_key)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_columns", None)
    print(edges)
