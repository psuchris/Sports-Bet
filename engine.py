import argparse
import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd
import requests
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

CACHE_DIR = ".cache"
RATINGS_CACHE_TTL_SEC = 6 * 60 * 60
REQUEST_TIMEOUT_SEC = 25

logger = logging.getLogger("sports_bet")


@dataclass
class ConsensusLine:
    point: float | None
    price: float | None
    books: int


@dataclass
class EdgeRow:
    Home: str
    Away: str
    ModelHome: float
    VegasHome: float
    VegasPrice: float | None
    Diff: float
    AbsDiff: float
    Books: int
    Pick: str


@dataclass
class MoneylineConsensus:
    home_price: float | None
    away_price: float | None
    books: int


@dataclass
class TotalConsensus:
    total: float | None
    over_price: float | None
    under_price: float | None
    books: int


@dataclass
class TopBet:
    Game: str
    Market: str
    Bet: str
    Line: float | None
    Price: float | None
    Confidence: int
    Edge: float
    Books: int


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
    "georgia state": "Georgia State",
    "nc state": "North Carolina State",
    "ole miss": "Mississippi",
    "uconn": "Connecticut",
    "umass": "Massachusetts",
    "miami": "Miami (FL)",
    "penn state": "Penn State",
    "gardner webb": "Gardner-Webb",
}


def _sports_reference_url(season_year: int) -> str:
    return f"https://www.sports-reference.com/cbb/seasons/men/{season_year}-ratings.html"


def get_ncaa_ratings(season_year: int) -> pd.DataFrame:
    url = _sports_reference_url(season_year)
    fp = _cache_path(f"cbb_{season_year}_ratings.csv")

    if _is_cache_fresh(fp, RATINGS_CACHE_TTL_SEC):
        return pd.read_csv(fp)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SportsBet/1.0; +https://example.com)"
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()

    dfs = pd.read_html(resp.text, attrs={"id": "ratings"})
    if not dfs:
        raise RuntimeError("No ratings table found on Sports-Reference.")
    df = dfs[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df[df["School"] != "School"].copy()
    df["SRS"] = pd.to_numeric(df["SRS"], errors="coerce")

    out = df[["School", "SRS"]].dropna().copy()
    out.columns = ["Team", "Rating"]
    out["Team"] = out["Team"].astype(str).str.replace(" NCAA", "", regex=False)

    if out.empty:
        raise RuntimeError("Ratings table parsed but no data rows were available.")

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


def fetch_spreads(
    api_key: str,
    sport: str = "basketball_ncaab",
    regions: str = "us",
    odds_format: str = "american",
) -> list[dict]:
    return fetch_odds(
        api_key=api_key,
        sport=sport,
        regions=regions,
        odds_format=odds_format,
        markets=["spreads"],
    )


def fetch_odds(
    api_key: str,
    sport: str = "basketball_ncaab",
    regions: str = "us",
    odds_format: str = "american",
    markets: list[str] | None = None,
) -> list[dict]:
    market_param = ",".join(markets or ["spreads"])
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={market_param}&oddsFormat={odds_format}"
    )
    r = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "message" in data:
        raise RuntimeError(data["message"])
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from odds API.")
    return data


def _trimmed(points: Iterable[float], trim_percent: float) -> list[float]:
    if not points:
        return []
    points = sorted(points)
    trim = int(len(points) * trim_percent)
    if trim == 0:
        return points
    return points[trim:-trim]


def consensus_home_spread(
    game: dict,
    home_team: str,
    method: str = "median",
    trim_percent: float = 0.1,
) -> ConsensusLine:
    points: list[float] = []
    prices: list[float] = []

    for bm in game.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") != "spreads":
                continue
            for o in m.get("outcomes", []) or []:
                if o.get("name") == home_team and "point" in o:
                    try:
                        points.append(float(o["point"]))
                    except (TypeError, ValueError):
                        pass
                    if "price" in o:
                        try:
                            prices.append(float(o["price"]))
                        except (TypeError, ValueError):
                            pass

    if not points:
        return ConsensusLine(point=None, price=None, books=0)

    if method == "trimmed_mean":
        trimmed = _trimmed(points, trim_percent)
        point = statistics.mean(trimmed) if trimmed else statistics.mean(points)
    else:
        point = statistics.median(points)

    price = statistics.median(prices) if prices else None
    return ConsensusLine(point=point, price=price, books=len(points))


def consensus_moneyline(game: dict, home_team: str, away_team: str) -> MoneylineConsensus:
    home_prices: list[float] = []
    away_prices: list[float] = []

    for bm in game.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []) or []:
                if o.get("name") == home_team and "price" in o:
                    try:
                        home_prices.append(float(o["price"]))
                    except (TypeError, ValueError):
                        pass
                if o.get("name") == away_team and "price" in o:
                    try:
                        away_prices.append(float(o["price"]))
                    except (TypeError, ValueError):
                        pass

    books = min(len(home_prices), len(away_prices))
    if not home_prices or not away_prices:
        return MoneylineConsensus(home_price=None, away_price=None, books=books)

    return MoneylineConsensus(
        home_price=statistics.median(home_prices),
        away_price=statistics.median(away_prices),
        books=books,
    )


def consensus_total(game: dict, method: str = "median", trim_percent: float = 0.1) -> TotalConsensus:
    totals: list[float] = []
    over_prices: list[float] = []
    under_prices: list[float] = []

    for bm in game.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") != "totals":
                continue
            for o in m.get("outcomes", []) or []:
                if o.get("name") == "Over" and "point" in o:
                    try:
                        totals.append(float(o["point"]))
                    except (TypeError, ValueError):
                        pass
                    if "price" in o:
                        try:
                            over_prices.append(float(o["price"]))
                        except (TypeError, ValueError):
                            pass
                if o.get("name") == "Under" and "point" in o:
                    try:
                        totals.append(float(o["point"]))
                    except (TypeError, ValueError):
                        pass
                    if "price" in o:
                        try:
                            under_prices.append(float(o["price"]))
                        except (TypeError, ValueError):
                            pass

    if not totals:
        return TotalConsensus(total=None, over_price=None, under_price=None, books=0)

    if method == "trimmed_mean":
        trimmed = _trimmed(totals, trim_percent)
        total = statistics.mean(trimmed) if trimmed else statistics.mean(totals)
    else:
        total = statistics.median(totals)

    return TotalConsensus(
        total=total,
        over_price=statistics.median(over_prices) if over_prices else None,
        under_price=statistics.median(under_prices) if under_prices else None,
        books=min(len(over_prices), len(under_prices)),
    )


def model_home_spread_from_srs(home_rating: float, away_rating: float, home_court_adv: float) -> float:
    expected_margin = (home_rating - away_rating) + home_court_adv
    return -expected_margin


def win_prob_from_rating(diff: float, scale: float = 7.5) -> float:
    return 1 / (1 + math.exp(-diff / scale))


def implied_prob_from_moneyline(price: float) -> float:
    if price > 0:
        return 100 / (price + 100)
    return -price / (-price + 100)


def confidence_from_edge(edge_value: float, books: int, scale: float) -> int:
    raw = abs(edge_value) * scale + books * 2
    return max(1, min(100, int(round(raw))))


def compute_edges(
    api_key: str,
    season_year: int,
    home_court_adv: float = 3.25,
    edge_threshold: float = 3.0,
    fuzzy_min_score: int = 93,
    min_books: int = 3,
    consensus_method: str = "median",
    trim_percent: float = 0.1,
    sport: str = "basketball_ncaab",
    regions: str = "us",
    odds_format: str = "american",
) -> pd.DataFrame:
    ratings = get_ncaa_ratings(season_year)
    lookup = TeamLookup(ratings, fuzzy_min_score=fuzzy_min_score)
    games = fetch_spreads(api_key, sport=sport, regions=regions, odds_format=odds_format)

    rows: list[EdgeRow] = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")

        hr, _, _ = lookup.get(home)
        ar, _, _ = lookup.get(away)
        if hr is None or ar is None:
            continue

        consensus = consensus_home_spread(
            g,
            home,
            method=consensus_method,
            trim_percent=trim_percent,
        )
        if consensus.point is None or consensus.books < min_books:
            continue

        model_home = model_home_spread_from_srs(hr, ar, home_court_adv)
        diff = model_home - consensus.point

        pick = ""
        if abs(diff) >= edge_threshold:
            pick = home if model_home < consensus.point else away

        rows.append(
            EdgeRow(
                Home=home,
                Away=away,
                ModelHome=round(model_home, 2),
                VegasHome=round(consensus.point, 2),
                VegasPrice=consensus.price,
                Diff=round(diff, 2),
                AbsDiff=round(abs(diff), 2),
                Books=consensus.books,
                Pick=pick,
            )
        )

    df = pd.DataFrame([asdict(row) for row in rows])
    if df.empty:
        return df
    df = df.sort_values(["AbsDiff", "Books"], ascending=[False, False]).reset_index(drop=True)
    return df


def _league_total_anchor(
    games: list[dict],
    method: str,
    trim_percent: float,
    min_books: int,
) -> float | None:
    totals: list[float] = []
    for g in games:
        consensus = consensus_total(g, method=method, trim_percent=trim_percent)
        if consensus.total is None or consensus.books < min_books:
            continue
        totals.append(consensus.total)
    if not totals:
        return None
    return statistics.median(totals)


def compute_top_bets(
    api_key: str,
    season_year: int,
    home_court_adv: float = 3.25,
    spread_edge_threshold: float = 3.0,
    moneyline_edge_threshold: float = 0.04,
    total_edge_threshold: float = 4.0,
    fuzzy_min_score: int = 93,
    min_books: int = 3,
    consensus_method: str = "median",
    trim_percent: float = 0.1,
    sport: str = "basketball_ncaab",
    regions: str = "us",
    odds_format: str = "american",
    top_n: int = 10,
) -> pd.DataFrame:
    ratings = get_ncaa_ratings(season_year)
    lookup = TeamLookup(ratings, fuzzy_min_score=fuzzy_min_score)
    games = fetch_odds(
        api_key=api_key,
        sport=sport,
        regions=regions,
        odds_format=odds_format,
        markets=["spreads", "h2h", "totals"],
    )

    league_total = _league_total_anchor(
        games,
        method=consensus_method,
        trim_percent=trim_percent,
        min_books=min_books,
    )

    rows: list[TopBet] = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue

        hr, _, _ = lookup.get(home)
        ar, _, _ = lookup.get(away)
        if hr is None or ar is None:
            continue

        game_label = f"{away} @ {home}"

        spread = consensus_home_spread(
            g,
            home,
            method=consensus_method,
            trim_percent=trim_percent,
        )
        if spread.point is not None and spread.books >= min_books:
            model_home = model_home_spread_from_srs(hr, ar, home_court_adv)
            diff = model_home - spread.point
            if abs(diff) >= spread_edge_threshold:
                pick_team = home if model_home < spread.point else away
                line = spread.point if pick_team == home else -spread.point
                confidence = confidence_from_edge(diff, spread.books, scale=12)
                rows.append(
                    TopBet(
                        Game=game_label,
                        Market="spread",
                        Bet=f"{pick_team} {line:+.1f}",
                        Line=round(line, 1),
                        Price=spread.price,
                        Confidence=confidence,
                        Edge=round(diff, 2),
                        Books=spread.books,
                    )
                )

        moneyline = consensus_moneyline(g, home_team=home, away_team=away)
        if (
            moneyline.home_price is not None
            and moneyline.away_price is not None
            and moneyline.books >= min_books
        ):
            rating_diff = (hr - ar) + home_court_adv
            home_win_prob = win_prob_from_rating(rating_diff)
            away_win_prob = 1 - home_win_prob

            home_implied = implied_prob_from_moneyline(moneyline.home_price)
            away_implied = implied_prob_from_moneyline(moneyline.away_price)

            home_edge = home_win_prob - home_implied
            away_edge = away_win_prob - away_implied

            if home_edge >= moneyline_edge_threshold or away_edge >= moneyline_edge_threshold:
                if home_edge >= away_edge:
                    confidence = confidence_from_edge(home_edge * 100, moneyline.books, scale=1.4)
                    rows.append(
                        TopBet(
                            Game=game_label,
                            Market="moneyline",
                            Bet=f"{home} ML",
                            Line=None,
                            Price=moneyline.home_price,
                            Confidence=confidence,
                            Edge=round(home_edge, 4),
                            Books=moneyline.books,
                        )
                    )
                else:
                    confidence = confidence_from_edge(away_edge * 100, moneyline.books, scale=1.4)
                    rows.append(
                        TopBet(
                            Game=game_label,
                            Market="moneyline",
                            Bet=f"{away} ML",
                            Line=None,
                            Price=moneyline.away_price,
                            Confidence=confidence,
                            Edge=round(away_edge, 4),
                            Books=moneyline.books,
                        )
                    )

        if league_total is not None:
            total = consensus_total(g, method=consensus_method, trim_percent=trim_percent)
            if total.total is not None and total.books >= min_books:
                model_total = league_total + hr + ar
                total_diff = model_total - total.total
                if abs(total_diff) >= total_edge_threshold:
                    if total_diff > 0:
                        pick = "Over"
                        price = total.over_price
                    else:
                        pick = "Under"
                        price = total.under_price
                    confidence = confidence_from_edge(total_diff, total.books, scale=10)
                    rows.append(
                        TopBet(
                            Game=game_label,
                            Market="total",
                            Bet=f"{pick} {total.total:.1f}",
                            Line=round(total.total, 1),
                            Price=price,
                            Confidence=confidence,
                            Edge=round(total_diff, 2),
                            Books=total.books,
                        )
                    )

    df = pd.DataFrame([asdict(row) for row in rows])
    if df.empty:
        return df
    df = df.sort_values(["Confidence", "Books"], ascending=[False, False]).head(top_n)
    return df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute model vs. market edges using SRS ratings."
    )
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEY"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--sport", default="basketball_ncaab")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--odds-format", default="american")
    parser.add_argument("--home-court-adv", type=float, default=3.25)
    parser.add_argument("--edge-threshold", type=float, default=3.0)
    parser.add_argument("--moneyline-edge-threshold", type=float, default=0.04)
    parser.add_argument("--total-edge-threshold", type=float, default=4.0)
    parser.add_argument("--fuzzy-min-score", type=int, default=93)
    parser.add_argument("--min-books", type=int, default=3)
    parser.add_argument("--consensus-method", choices=["median", "trimmed_mean"], default="median")
    parser.add_argument("--trim-percent", type=float, default=0.1)
    parser.add_argument("--top-bets", type=int, default=10)
    parser.add_argument("--mode", choices=["edges", "top", "both"], default="top")
    parser.add_argument("--output", help="Write CSV output to this path.")
    parser.add_argument("--output-json", help="Write JSON output to this path.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.api_key:
        raise SystemExit("Set ODDS_API_KEY or pass --api-key with your The Odds API key.")

    logger.info("Fetching ratings for %s season.", args.season)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)

    if args.mode in {"edges", "both"}:
        edges = compute_edges(
            api_key=args.api_key,
            season_year=args.season,
            home_court_adv=args.home_court_adv,
            edge_threshold=args.edge_threshold,
            fuzzy_min_score=args.fuzzy_min_score,
            min_books=args.min_books,
            consensus_method=args.consensus_method,
            trim_percent=args.trim_percent,
            sport=args.sport,
            regions=args.regions,
            odds_format=args.odds_format,
        )

        if edges.empty:
            logger.warning("No qualifying spread edges found. Consider lowering edge threshold.")
        else:
            print(edges)

        if args.output:
            edges.to_csv(args.output, index=False)
            logger.info("Wrote CSV output to %s.", args.output)

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(edges.to_dict(orient="records"), handle, indent=2)
            logger.info("Wrote JSON output to %s.", args.output_json)

    if args.mode in {"top", "both"}:
        top_bets = compute_top_bets(
            api_key=args.api_key,
            season_year=args.season,
            home_court_adv=args.home_court_adv,
            spread_edge_threshold=args.edge_threshold,
            moneyline_edge_threshold=args.moneyline_edge_threshold,
            total_edge_threshold=args.total_edge_threshold,
            fuzzy_min_score=args.fuzzy_min_score,
            min_books=args.min_books,
            consensus_method=args.consensus_method,
            trim_percent=args.trim_percent,
            sport=args.sport,
            regions=args.regions,
            odds_format=args.odds_format,
            top_n=args.top_bets,
        )

        if top_bets.empty:
            logger.warning("No qualifying top bets found. Try lowering thresholds.")
        else:
            print(top_bets)


if __name__ == "__main__":
    main()
