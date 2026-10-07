"""Read-only Gamma identity + CLOB books; never estimates a spread from trades.

python -m data_sources.polymarket_books --event-slug nba-orl-bos-2026-... \
    --game-id <id> --source frozen --output runs/market-book.json
Automatic match discovery recognizes the standard NBA settlement template; other rules remain unverified.
Explicit CLI rule verification requires reviewing the exact market description.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.static import teams

from data_sources import FROZEN, ROOT, read_table
from data_sources.inplay_types import utc
from data_sources.polymarket import GAMMA, CLOB


def array(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list):
        raise ValueError("market outcomes/tokens are not arrays")
    return value


def team_code(label):
    value = str(label).strip().casefold()
    for team in teams.get_teams():
        if value in {team["abbreviation"].casefold(), team["nickname"].casefold(), team["full_name"].casefold()}:
            return team["abbreviation"]
    return None


def match_market(event, game):
    candidates = []
    for market in event.get("markets", []):
        if market.get("sportsMarketType") != "moneyline":
            continue
        try:
            labels, tokens = array(market["outcomes"]), array(market["clobTokenIds"])
            codes = [team_code(label) for label in labels]
            if len(labels) != 2 or len(tokens) != 2 or len(set(tokens)) != 2:
                continue
            if set(codes) != {game.home_team, game.away_team}:
                continue
            if abs((utc(market["gameStartTime"])-utc(game.tip_time)).total_seconds()) > 60:
                continue
            if not market.get("conditionId"):
                continue
            candidates.append((market, {"home": str(tokens[codes.index(game.home_team)]), "away": str(tokens[codes.index(game.away_team)])}))
        except (KeyError, TypeError, ValueError):
            continue
    if len(candidates) != 1:
        raise ValueError("need exactly one full-game moneyline with matching teams and tip time; no fuzzy match")
    return candidates[0]


def fee_parameters(market, info):
    if market.get("feesEnabled") is False:
        return {"fee_rate": 0., "fee_exponent": 1., "fee_verified": True}
    schedule = market.get("feeSchedule")
    if schedule:
        rate, exponent = schedule.get("rate"), schedule.get("exponent")
    else:
        schedule = info.get("fd", {})
        rate, exponent = schedule.get("r"), schedule.get("e")
    try:
        rate, exponent = float(rate), float(exponent)
        # Only the documented linear sports fee curve is implemented for live books.
        if not 0 <= rate <= 1 or exponent != 1:
            raise ValueError("unsupported fee schedule")
        return {"fee_rate": rate, "fee_exponent": exponent, "fee_verified": True}
    except (TypeError, ValueError):
        return {"fee_rate": None, "fee_exponent": None, "fee_verified": False}


def standard_rules(market):
    """Recognize the verified standard NBA rule template, not a broad title match."""
    text = str(market.get("description", "")).lower()
    return (bool(re.search(r"final score including (?:any )?overtime", text))
            and "postponed" in text and "completed" in text and "cancel" in text
            and bool(re.search(r"resolve (?:to )?50[-/]50", text)))


def bet_contract_spec(market, game):
    """Recognize observed standard full-game NBA half-point contracts only."""
    try:
        kind = {'moneyline': 'moneyline', 'spreads': 'spread', 'totals': 'total'}.get(market.get('sportsMarketType'))
        labels, tokens = array(market['outcomes']), array(market['clobTokenIds'])
        if kind is None or len(labels) != 2 or len(tokens) != 2 or len(set(tokens)) != 2 or not market.get('conditionId'):
            return None
        if abs((utc(market['gameStartTime'])-utc(game.tip_time)).total_seconds()) > 60:
            return None
        if kind == 'moneyline':
            if not standard_rules(market): return None
            _, mapping = match_market({'markets': [market]}, game)
            return {'kind': kind, 'line': None, 'tokens': mapping}
        description = str(market.get('description', '')).lower()
        if ('postponed' not in description or 'completed' not in description or 'cancel' not in description
                or not re.search(r'resolve (?:to )?50[-/]50', description)
                or re.search(r'half|quarter|regulation only', description)):
            return None
        line = float(market['line'])
        if not (abs(line) < 400 and abs(line % 1-.5) < 1e-8): return None
        if kind == 'total':
            if {str(x).lower() for x in labels} != {'over', 'under'} or line <= 0: return None
            if not re.search(r'combine to score '+str(math.floor(line)+1)+r' or more points in this game', description): return None
            mapping = {str(label).lower(): str(token) for label,token in zip(labels,tokens)}
        else:
            codes = [team_code(x) for x in labels]
            if set(codes) != {game.home_team, game.away_team}: return None
            match = re.fullmatch(r'Spread: (.+?) \(([+-]?\d+(?:\.\d+)?)\)', str(market.get('question', '')))
            if not match or team_code(match[1]) != codes[0] or float(match[2]) != line or line >= 0: return None
            if not re.search(r'win the game by '+str(math.floor(-line)+1)+r' or more points', description): return None
            if codes[0] != game.home_team: line = -line
            mapping = {'home': str(tokens[codes.index(game.home_team)]), 'away': str(tokens[codes.index(game.away_team)])}
        return {'kind': kind, 'line': line, 'tokens': mapping}
    except (ValueError, TypeError, KeyError):
        return None


def normalize_book(book, token, condition, observed_at):
    if str(book.get("asset_id")) != str(token) or book.get("market") != condition:
        raise ValueError("CLOB token or condition does not match Gamma")
    value = float(book["timestamp"])
    updated = pd.to_datetime(value, unit="ms" if value > 100_000_000_000 else "s", utc=True)
    if updated > utc(observed_at)+pd.Timedelta(seconds=5):
        raise ValueError("order book source time is in the future")
    levels = {}
    for side in ("asks", "bids"):
        rows = [{"price": float(r["price"]), "size": float(r["size"])} for r in book.get(side, [])]
        if any(not 0 < r["price"] < 1 or not 0 <= r["size"] < float("inf") for r in rows):
            raise ValueError("invalid book price or quantity")
        levels[side] = sorted([r for r in rows if r["size"] > 0], key=lambda r: r["price"], reverse=side == "bids")
    ask = levels["asks"][0]["price"] if levels["asks"] else None
    bid = levels["bids"][0]["price"] if levels["bids"] else None
    return {**levels, "bid": bid, "ask": ask, "mid": (ask+bid)/2 if ask is not None and bid is not None else None,
            "updated_at": updated.isoformat(), "observed_at": utc(observed_at).isoformat(), "hash": book.get("hash"),
            "minimum_shares": float(book.get("min_order_size") or 0)}


class PolymarketReader:
    def __init__(self, session=None, clock=lambda: pd.Timestamp.now(tz="UTC")):
        self.session = session or requests.Session()
        self.clock = clock
        self.discovery = {}

    def _get(self, url, params=None):
        response = self.session.get(url, params=params, timeout=10, headers={"User-Agent": "nba-agent/0.1 read-only"})
        response.raise_for_status()
        return response.json()

    def fetch(self, slug, game, rules_verified=False):
        if not re.fullmatch(r"[a-z0-9-]+", slug):
            raise ValueError("use the event slug, not a custom request URL")
        event = self._get(GAMMA+"/events/slug/"+slug)
        return self._fetch_event(event, game, slug, rules_verified)

    def _discover_game_event(self, game):
        """Automatically find the exact scheduled matchup; never reuse another game."""
        key = (str(game.game_id), str(game.tip_time), game.home_team, game.away_team)
        cached = self.discovery.get(key)
        if cached and time.monotonic()-cached[0] < 20:
            event = cached[1]
        else:
            day = utc(game.tip_time).tz_convert("America/New_York").strftime("%Y-%m-%d")
            slug = f"nba-{game.away_team.lower()}-{game.home_team.lower()}-{day}"
            event = None
            try:
                found = self._get(GAMMA+"/events/slug/"+slug)
                match_market(found, game)
                event = found
            except (requests.RequestException, ValueError, TypeError, KeyError):
                pass
            if event is None:
                matched = []
                for offset in (0, 100):
                    page = self._get(GAMMA+"/events", {"tag_slug": "nba", "active": "true", "closed": "false", "limit": 100, "offset": offset})
                    for candidate in page:
                        try:
                            match_market(candidate, game)
                            matched.append(candidate)
                        except (ValueError, TypeError, KeyError):
                            continue
                    if len(page) < 100:
                        break
                if len(matched) != 1:
                    raise ValueError("No unique Polymarket winner market matches this game yet")
                event = matched[0]
            self.discovery[key] = (time.monotonic(), event)
        return event

    def fetch_game(self, game):
        event = self._discover_game_event(game)
        return self._fetch_event(event, game, event.get('slug', ''), None)

    def fetch_bet_contracts(self, game):
        """Full-game winner/spread/total references. /books POST is read-only batch data."""
        from concurrent.futures import ThreadPoolExecutor
        event = self._discover_game_event(game)
        specs = []
        for market in event.get('markets', []):
            spec = bet_contract_spec(market, game)
            if spec:
                specs.append((market, spec))
        # One most-liquid exact full-game line per type; no fuzzy quarter/half matches.
        selected = []
        for kind in ('moneyline', 'spread', 'total'):
            candidates = [(m,s) for m,s in specs if s['kind'] == kind and m.get('active') is True and m.get('closed') is False and m.get('acceptingOrders') is True]
            if candidates:
                selected.append(max(candidates, key=lambda item: float(item[0].get('liquidityNum') or item[0].get('liquidity') or 0)))
        tokens = [token for _, spec in selected for token in spec['tokens'].values()]
        if not tokens:
            return {'contracts': [], 'errors': ['No open matching full-game markets.']}
        response = self.session.post(CLOB+'/books', json=[{'token_id': t} for t in tokens], timeout=10,
                                     headers={'User-Agent': 'nba-agent/0.1 read-only'})
        response.raise_for_status()
        raw_books = response.json()
        if not isinstance(raw_books, list):
            raise ValueError('Invalid batch book response')
        books = {str(b.get('asset_id')): b for b in raw_books}
        def read_contract(item):
            market, spec = item
            condition = market['conditionId']
            try:
                info = self._get(CLOB+'/clob-markets/'+condition)
            except requests.RequestException:
                info = {}
            fees = fee_parameters(market, info)
            quotes = {}
            for side, token in spec['tokens'].items():
                try:
                    b = normalize_book(books[token], token, condition, utc(self.clock()))
                    quotes[side] = {**b, **fees, 'game_id': str(game.game_id), 'side': side,
                        'kind': spec['kind'], 'condition_id': condition, 'token_id': token, 'synthetic': False,
                        'rules_verified': True, 'includes_overtime': True, 'active': True,
                        'url': 'https://polymarket.com/event/'+event['slug'], 'rules': market.get('description', '')}
                except (ValueError, TypeError, KeyError):
                    continue
            return {'id': condition, 'kind': spec['kind'], 'line': spec['line'], 'quotes': quotes}
        with ThreadPoolExecutor(max_workers=3) as pool:
            contracts = list(pool.map(read_contract, selected))
        return {'contracts': contracts, 'errors': [], 'captured_at': utc(self.clock()).isoformat()}


    def _fetch_event(self, event, game, slug, rules_verified):
        market, tokens = match_market(event, game)
        condition = market["conditionId"]
        info, errors = {}, []
        try:
            info = self._get(CLOB+"/clob-markets/"+condition)
        except requests.RequestException:
            errors.append("market fee details unavailable")
        fees = fee_parameters(market, info)
        description = market.get("description", "")
        verified = standard_rules(market) if rules_verified is None else bool(rules_verified)
        quotes = {}
        for side, token in tokens.items():
            try:
                raw = self._get(CLOB+"/book", {"token_id": token})
                observed = utc(self.clock())
                book = normalize_book(raw, token, condition, observed)
                quotes[side] = {**book, **fees, "venue": "polymarket", "synthetic": False,
                                "game_id": str(game.game_id), "kind": "moneyline", "side": side,
                                "team": game.home_team if side == "home" else game.away_team,
                                "includes_overtime": verified, "rules_verified": verified,
                                "active": market.get("active") is True and market.get("closed") is False and market.get("acceptingOrders") is True,
                                "token_id": token, "condition_id": condition, "market_id": str(market.get("id", "")),
                                "event_slug": slug, "rules": description,
                                "url": "https://polymarket.com/event/"+slug, "raw_book": raw}
            except (requests.RequestException, ValueError, KeyError, TypeError):
                errors.append(side+" order book unavailable or invalid")
        return {"schema_version": 1, "quotes": quotes, "rules": description, "raw_market": market,
                "raw_fee_info": info, "errors": errors, "captured_at": utc(self.clock()).isoformat(),
                "note": "Read-only order books. Verify overtime, cancellation and postponement rules; no orders are sent."}

    def close(self):
        self.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-slug", required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--source", choices=("sample", "frozen"), default="frozen")
    parser.add_argument("--rules-verified", action="store_true", help="Use only after reviewing the exact full-game settlement rules")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    games = read_table("games", ROOT/"data"/"sample" if args.source == "sample" else FROZEN)
    matches = games[games.game_id.astype(str) == args.game_id]
    if len(matches) != 1:
        parser.error("game must match exactly once in the selected dataset")
    reader = PolymarketReader()
    try:
        result = reader.fetch(args.event_slug, next(matches.itertuples(index=False)), args.rules_verified)
    finally:
        reader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Captured {len(result['quotes'])} books; {result['errors']}; {args.output}")


if __name__ == "__main__":
    main()
