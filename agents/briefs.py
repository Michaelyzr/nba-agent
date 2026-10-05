"""One forecast core, four channel briefs (README section 1). Every number comes from forecast/api.py.

platform  fantasy and media platforms: distributions, win-probability change, freshness
media     broadcasters and writers: who gains and who loses; no stake or betting language
team      coaching and analytics: opponent rotation and matchup odds; no market signal
retail    bettors: model vs market with fees; confirm per order; never "lock" copy
"""
import re

import pandas as pd

MARKET_WORDS = re.compile(r"\b(bet|bets|betting|odds|stake|wager|price|market|kalshi|buy|sell)\b", re.I)


def outlook(forecaster, view, game, out: list) -> dict:
    """Win probability and player distributions before and after the news, for both teams."""
    before, after = forecaster.win(view, game, []), forecaster.win(view, game, out)
    teams = {}
    for team_id, code in ((game.home_team_id, game.home_team), (game.away_team_id, game.away_team)):
        rows = {}
        for label, o in (("before", []), ("after", out)):
            for f in forecaster.players(view, game, team_id, o):
                r = rows.setdefault(f["player_id"], {"player_id": f["player_id"]})
                r[f"{f['target']}_{label}"] = f["quantiles"]
                r[f"p_play_{label}"] = f["p_play"]
        for pid in out:
            if pid in rows:
                rows[pid].update(min_after=None, pts_after=None, p_play_after=0.02)
        teams[code] = list(rows.values())
    return {"p_home_before": before["p_home"], "p_home_after": after["p_home"], "missing": after["missing"],
            "teams": teams, "as_of": view.now}


def player_table(o: dict, names: dict) -> pd.DataFrame:
    rows = []
    for team, players in o["teams"].items():
        for r in players:
            mb, ma = (r.get("min_before") or {}), (r.get("min_after") or {})
            pb, pa = (r.get("pts_before") or {}), (r.get("pts_after") or {})
            rows.append({"team": team, "player": names.get(r["player_id"], str(r["player_id"])),
                         "p_play": r.get("p_play_after", r.get("p_play_before")),
                         "min_p50_before": mb.get("p50"), "min_p50_after": ma.get("p50"),
                         "min_p10_p90_after": f"{ma.get('p10', 0):.0f}-{ma.get('p90', 0):.0f}" if ma else "out",
                         "pts_p50_before": pb.get("p50"), "pts_p50_after": pa.get("p50")})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["min_change"] = (t.min_p50_after.fillna(0) - t.min_p50_before.fillna(0)).round(1)
    return t.sort_values(["team", "min_change"], ascending=[True, False]).reset_index(drop=True)


def _movers(table: pd.DataFrame, n=3):
    played = table[table.min_p50_after.notna()]
    return played.nlargest(n, "min_change"), table[table.min_p50_after.isna()]


def channel_briefs(game, o: dict, news: pd.DataFrame, names: dict, market: dict | None = None,
                   order_text: str | None = None) -> dict:
    """market: {"bid", "ask"} for the home team's game-winner contract, if any."""
    table = player_table(o, names)
    home, away = game.home_team, game.away_team
    pb, pa = o["p_home_before"], o["p_home_after"]
    news_line = " ".join(news.sort_values("published_at").text.tolist()) or "No injury news yet."
    fresh = (f"as of {pd.Timestamp(o['as_of']):%Y-%m-%d %H:%M} UTC; latest news "
             f"{pd.Timestamp(news.published_at.max()):%H:%M} UTC" if len(news) else
             f"as of {pd.Timestamp(o['as_of']):%Y-%m-%d %H:%M} UTC; no news")
    gainers, outs = _movers(table) if len(table) else (table, table)
    gain_text = ", ".join(f"{r.player} ({r.team}) {r.min_p50_before:.0f} to {r.min_p50_after:.0f} minutes"
                          for r in gainers.itertuples() if r.min_change > 0.5) or "no clear minutes winner"
    out_text = ", ".join(f"{r.player} ({r.team})" for r in outs.itertuples()) or "nobody"

    platform = (f"{away} at {home}. {news_line}\nHome win probability {pb:.0%} before the news, {pa:.0%} after. "
                f"Biggest minutes gains: {gain_text}. Full distributions in the table ({fresh}).")
    media = (f"{away} at {home}: {out_text} will not play. {news_line} Who gains: {gain_text}. "
             f"Our model gives {home} a {pa:.0%} chance to win, {'down' if pa < pb else 'up'} from {pb:.0%}.")
    team = (f"Internal scouting note, {away} at {home}. Out: {out_text}. Expected rotation shifts: {gain_text}. "
            f"Matchup win chance for {home}: {pa:.0%} (was {pb:.0%}).")
    if market:
        mid = (market["bid"] + market["ask"]) / 2
        retail = (f"{away} at {home}. {news_line}\nModel: {home} {pa:.0%} to win ({pb:.0%} before the news). "
                  f"Market: {market['bid']:.0%}-{market['ask']:.0%} (mid {mid:.0%}). "
                  + (order_text + " Confirm to place this order; caps and a post-tip block apply." if order_text
                     else "No order: the gap after fees is too small or the price already moved."))
    else:
        retail = f"{away} at {home}. No market price is available, so no order can be suggested."
    for name, text in (("media", media), ("team", team)):
        assert not MARKET_WORDS.search(text), f"{name} brief contains market language"
    return {"platform": platform, "media": media, "team": team, "retail": retail, "table": table}
