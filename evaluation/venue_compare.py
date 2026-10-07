"""NBA game-winner prices across venues: Kalshi, Polymarket and DraftKings (via ESPN).

    python -m data_sources.polymarket --start 2026-02-01 --end 2026-04-12 --by-slug   # once
    python -m evaluation.venue_compare

Per test-period game (1 Feb - 12 Apr 2026) and snapshot time (tip - 24 h,
tip - 1 h, close = last quote before tip) this collects:

  kalshi      real top-of-book bid/ask for both team contracts (1-minute data)
  polymarket  last traded price per minute from CLOB prices-history; there is
              no book, so the ask is the traded price + HALF_SPREAD (synthetic)
  draftkings  moneylines from the ESPN summary `pickcenter` (open and close
              only; ESPN keeps no timestamped movement for these games)

The fair home-win probability is the mid with the two sides normalised to sum
to 1 (Kalshi, Polymarket) or the vig-free implied probability (DraftKings,
proportional normalisation). The all-in cost of a $1 payout is the ask plus
the venue's taker fee, or 1 / decimal odds for the sportsbook (vig included).

Writes data/frozen/venue_prices.parquet (per game, git-ignored) and
evaluation/results/venue_compare.{csv,md,png}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import FROZEN, RAW
from data_sources.polymarket import HALF_SPREAD, parse_slug
from evaluation.stats import REPS, SEED

RESULTS = Path(__file__).resolve().parent / "results"
SUMMARY = RAW / "espn" / "summary"
START, END = "2026-02-01", "2026-04-12"
VENUES = ("kalshi", "polymarket", "draftkings")
TIMES = {"t24h": pd.Timedelta(hours=24), "t1h": pd.Timedelta(hours=1), "close": pd.Timedelta(0)}
MAX_AGE = {"t24h": pd.Timedelta(hours=6), "t1h": pd.Timedelta(hours=2), "close": pd.Timedelta(hours=2)}
KALSHI_RATE = 0.07
POLYMARKET_RATE = 0.05     # current sports taker fee (docs.polymarket.com/trading/fees), makers free
ESPN_ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX",
             "BRK": "BKN"}


# ---------------------------------------------------------------- price maths

def american_to_prob(odds: float) -> float:
    """Implied probability with the vig still in (= cost of a $1 payout)."""
    odds = float(odds)
    if np.isnan(odds) or odds == 0:
        return np.nan
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def devig(p_home: float, p_away: float) -> tuple[float, float]:
    """Proportional normalisation: divide each side by the overround."""
    total = p_home + p_away
    return p_home / total, p_away / total


def kalshi_fee(p: float) -> float:
    """Per-contract Kalshi taker fee (unrounded; real orders round up to the cent)."""
    return KALSHI_RATE * p * (1 - p)


def polymarket_fee(p: float) -> float:
    return POLYMARKET_RATE * p * (1 - p)


def all_in(venue: str, price: float) -> float:
    """Cost of a $1 payout: ask plus taker fee, or the sportsbook's vig-inclusive price."""
    if venue == "kalshi":
        return price + kalshi_fee(price)
    if venue == "polymarket":
        return price + polymarket_fee(price)
    return price


def arbitrage(cost_home: float, cost_away: float) -> float:
    """Locked-in profit per $1 payout of buying home on one venue and away on another (< 0: none)."""
    return 1.0 - (cost_home + cost_away)


def match_games(events: pd.DataFrame, games: pd.DataFrame, max_days: int = 1) -> pd.Series:
    """game_id for each (date, team_a, team_b) row, home/away order ignored, date within +-max_days."""
    pair = lambda a, b: "-".join(sorted((a, b)))
    g = games.assign(key=[pair(h, a) for h, a in zip(games.home_team, games.away_team)],
                     d=pd.to_datetime(games.date))
    by_key = {k: v for k, v in g.groupby("key")}
    out = []
    for date, a, b in zip(pd.to_datetime(events.date), events.team_a, events.team_b):
        cand = by_key.get(pair(a, b))
        if cand is None:
            out.append(None)
            continue
        gap = (cand.d - date).abs()
        out.append(cand.game_id.iloc[int(gap.argmin())] if gap.min() <= pd.Timedelta(days=max_days) else None)
    return pd.Series(out, index=events.index, dtype=object)


# ---------------------------------------------------------------- data

def _last(series: tuple, t: np.datetime64, max_age: pd.Timedelta):
    """(bid, ask) of the last quote strictly before t, or None if missing or older than max_age."""
    ts, bid, ask = series
    i = int(np.searchsorted(ts, t, side="left"))
    if i == 0 or t - ts[i - 1] > max_age.to_timedelta64():
        return None
    return bid[i - 1], ask[i - 1]


def _quotes(markets: pd.DataFrame, prices: pd.DataFrame, games: pd.DataFrame, venue: str) -> pd.DataFrame:
    prices = prices.assign(ts=pd.to_datetime(prices.ts, utc=True).dt.tz_convert(None)).sort_values("ts")
    by_ticker = {k: (v.ts.to_numpy(), v.bid.to_numpy(float), v.ask.to_numpy(float))
                 for k, v in prices.groupby("market_ticker")}
    tickers = {(g, team): t for g, team, t in zip(markets.game_id, markets.team, markets.market_ticker)}
    rows = []
    for g in games.itertuples():
        home, away = tickers.get((g.game_id, g.home_team)), tickers.get((g.game_id, g.away_team))
        if home is None or away is None:
            continue
        tip = g.tip_time.tz_convert(None)
        row = {"game_id": g.game_id}
        for name, lead in TIMES.items():
            t = (tip - lead).to_datetime64()
            for side, ticker in (("home", home), ("away", away)):
                q = _last(by_ticker[ticker], t, MAX_AGE[name]) if ticker in by_ticker else None
                row[f"{venue}_{name}_{side}_bid"] = np.nan if q is None else q[0]
                row[f"{venue}_{name}_{side}_ask"] = np.nan if q is None else q[1]
        rows.append(row)
    out = pd.DataFrame(rows)
    for name in TIMES:
        c = lambda s, k: out[f"{venue}_{name}_{s}_{k}"]
        home_mid, away_mid = (c("home", "bid") + c("home", "ask")) / 2, (c("away", "bid") + c("away", "ask")) / 2
        out[f"{venue}_{name}_p_home"] = home_mid / (home_mid + away_mid)
    return out


def kalshi(games: pd.DataFrame) -> pd.DataFrame:
    markets = pd.read_parquet(FROZEN / "markets.parquet")
    markets = markets[(markets.venue == "kalshi") & (markets.kind == "game") & markets.game_id.isin(games.game_id)]
    prices = pd.read_parquet(FROZEN / "prices.parquet", filters=[("market_ticker", "in", list(markets.market_ticker))])
    return _quotes(markets, prices, games, "kalshi")


def polymarket(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    path = FROZEN / "markets_polymarket.parquet"
    if not path.exists():
        return pd.DataFrame({"game_id": []}), {"markets": 0, "matched": 0}
    markets = pd.read_parquet(path)
    info = pd.DataFrame([parse_slug(t[3:].rsplit("-", 1)[0]) for t in markets.market_ticker])
    markets = markets.assign(game_id=match_games(
        info.rename(columns={"home_team": "team_a", "away_team": "team_b"}), games).to_numpy())
    stats = {"markets": len(markets) // 2, "matched": int(markets.game_id.notna().sum()) // 2}
    markets = markets.dropna(subset=["game_id"])
    prices = pd.read_parquet(FROZEN / "prices_polymarket.parquet")
    prices = prices[prices.market_ticker.isin(markets.market_ticker)]
    return _quotes(markets, prices, games, "polymarket"), stats


def draftkings(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid in games.game_id:
        path = SUMMARY / f"{gid}.json"
        if not path.exists():
            continue
        pick = next((p for p in json.loads(path.read_text()).get("pickcenter") or []
                     if "draft" in p.get("provider", {}).get("name", "").lower()), None)
        if pick is None:
            continue
        ml = pick.get("moneyline") or {}
        row = {"game_id": gid}
        for when in ("open", "close"):
            for side in ("home", "away"):
                odds = (ml.get(side, {}).get(when) or {}).get("odds")
                if when == "close" and odds is None:
                    odds = pick.get(f"{side}TeamOdds", {}).get("moneyLine")
                try:
                    odds = float(str(odds).replace("+", "")) if odds not in (None, "", "OFF", "EVEN") else (
                        100.0 if odds == "EVEN" else np.nan)
                except ValueError:
                    odds = np.nan
                row[f"draftkings_{when}_{side}_odds"] = odds
                row[f"draftkings_{when}_{side}_ask"] = american_to_prob(odds)
            h, a = row[f"draftkings_{when}_home_ask"], row[f"draftkings_{when}_away_ask"]
            row[f"draftkings_{when}_overround"] = h + a - 1
            row[f"draftkings_{when}_p_home"] = devig(h, a)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def build_table() -> tuple[pd.DataFrame, dict]:
    games = pd.read_parquet(FROZEN / "games.parquet")
    games = games[(games.date >= START) & (games.date <= END)].copy()
    games["home_win"] = (games.home_pts > games.away_pts).astype(int)
    pm, pm_stats = polymarket(games)
    table = games[["game_id", "date", "tip_time", "home_team", "away_team", "home_win"]]
    for part in (kalshi(games), pm, draftkings(games)):
        if len(part):
            table = table.merge(part, on="game_id", how="left")
    return table.reset_index(drop=True), {"games": len(games), "polymarket": pm_stats}


# ---------------------------------------------------------------- analysis

def p_col(venue, t):
    return f"{venue}_{t}_p_home"


def present(table, venue, t):
    c = p_col(venue, t)
    return table[c].notna() if c in table else pd.Series(False, index=table.index)


def coverage(table) -> pd.DataFrame:
    rows = []
    for v in VENUES:
        for t in ("t24h", "t1h", "close", "open"):
            if p_col(v, t) in table:
                rows.append({"venue": v, "time": t, "games": int(present(table, v, t).sum())})
    return pd.DataFrame(rows)


def gaps(table) -> pd.DataFrame:
    rows = []
    for t in TIMES:
        for a, b in (("kalshi", "polymarket"), ("kalshi", "draftkings"), ("polymarket", "draftkings")):
            if p_col(a, t) not in table or p_col(b, t) not in table:
                continue
            d = (table[p_col(a, t)] - table[p_col(b, t)]).abs().dropna() * 100
            if len(d):
                rows.append({"time": t, "pair": f"{a} vs {b}", "games": len(d), "mean_pp": d.mean(),
                             "median_pp": d.median(), "p90_pp": d.quantile(0.9)})
    return pd.DataFrame(rows)


def costs(table, t) -> dict:
    """All-in cost of a $1 payout per venue and side at time t."""
    out = {}
    for v in VENUES:
        for side in ("home", "away"):
            c = f"{v}_{t}_{side}_ask"
            if c in table:
                out[(v, side)] = table[c].map(lambda p, v=v: all_in(v, p) if pd.notna(p) else np.nan)
    return out


def best_venue(table, t, venues) -> pd.DataFrame:
    """Share of (game, side) pairs where each venue is cheapest after fees, among games quoted by all."""
    c = costs(table, t)
    if any((v, "home") not in c for v in venues):
        return pd.DataFrame()
    rows = []
    for side in ("home", "away"):
        frame = pd.DataFrame({v: c[(v, side)] for v in venues}).dropna()
        best = frame.min(axis=1)
        for v in venues:
            rows.append({"time": t, "venues": " / ".join(venues), "side": side, "venue": v, "n": len(frame),
                         "cheapest": int((frame[v] <= best + 1e-12).sum()),
                         "mean_cost_over_best_c": float((frame[v] - best).mean() * 100)})
    out = pd.DataFrame(rows).groupby(["time", "venues", "venue"], as_index=False).agg(
        n=("n", "sum"), cheapest=("cheapest", "sum"), mean_cost_over_best_c=("mean_cost_over_best_c", "mean"))
    out["share"] = out.cheapest / out.n
    return out


def arbs(table, t) -> tuple[pd.DataFrame, pd.Series]:
    """Best cross-venue home/away pair per game; positive edge = locked-in profit per $1."""
    c = costs(table, t)
    names = sorted({v for v, _ in c})
    edges = {}
    for a in names:
        for b in names:
            if a != b:
                edges[f"{a} home + {b} away"] = arbitrage(c[(a, "home")], c[(b, "away")])
    frame = pd.DataFrame(edges)
    rows = []
    for col in frame:
        e = frame[col].dropna()
        rows.append({"time": t, "combo": col, "games": len(e), "arb_games": int((e > 0).sum()),
                     "arb_share": float((e > 0).mean()) if len(e) else np.nan,
                     "median_edge_c": float(e[e > 0].median() * 100) if (e > 0).any() else np.nan,
                     "max_edge_c": float(e.max() * 100) if len(e) else np.nan,
                     "median_cost_c": float((1 - e).median() * 100) if len(e) else np.nan})
    return pd.DataFrame(rows), frame.max(axis=1, skipna=True)


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def sharpness(table, t="close", reps=REPS, seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier / log loss per venue on games all venues quote, and paired differences with a game-day bootstrap."""
    venues = [v for v in VENUES if p_col(v, t) in table]
    common = table.dropna(subset=[p_col(v, t) for v in venues])
    y = common.home_win.to_numpy(float)
    days = common.date.to_numpy()
    uniq = np.unique(days)
    pos = {d: np.flatnonzero(days == d) for d in uniq}
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(reps, len(uniq)))
    losses = {(v, m): f(common[p_col(v, t)].to_numpy(float), y) for v in venues
              for m, f in (("brier", _brier), ("log_loss", _logloss))}
    day_sums = {k: np.array([v[pos[d]].sum() for d in uniq]) for k, v in losses.items()}
    day_n = np.array([len(pos[d]) for d in uniq])
    n_boot = day_n[draws].sum(axis=1)

    def ci(sums):
        b = sums[draws].sum(axis=1) / n_boot
        return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))

    level = []
    for (v, m), l in losses.items():
        lo, hi = ci(day_sums[(v, m)])
        level.append({"time": t, "venue": v, "metric": m, "games": len(y), "days": len(uniq),
                      "value": float(l.mean()), "ci_low": lo, "ci_high": hi})
    diff = []
    for i, a in enumerate(venues):
        for b in venues[i + 1:]:
            for m in ("brier", "log_loss"):
                d = day_sums[(a, m)] - day_sums[(b, m)]
                lo, hi = ci(d)
                diff.append({"time": t, "pair": f"{a} - {b}", "metric": m, "value": float(d.sum() / day_n.sum()),
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(level), pd.DataFrame(diff)


# ---------------------------------------------------------------- report

def figure(table, sharp, sharp_diff, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    pairs = [("kalshi", "draftkings"), ("kalshi", "polymarket"), ("polymarket", "draftkings")]
    bins = np.linspace(0, 10, 41)
    for a, b in pairs:
        if p_col(a, "close") in table and p_col(b, "close") in table:
            d = ((table[p_col(a, "close")] - table[p_col(b, "close")]).abs() * 100).dropna()
            if len(d):
                ax[0].hist(d.clip(upper=10), bins=bins, histtype="step", lw=1.6, label=f"{a} vs {b} (n={len(d)})")
    ax[0].set(xlabel="|gap| in home-win probability at close (pp, clipped at 10)", ylabel="games",
              title="Cross-venue gap at close")
    ax[0].legend(fontsize=8)

    k, dk = p_col("kalshi", "close"), p_col("draftkings", "close")
    sub = table.dropna(subset=[k, dk])
    ax[1].scatter(sub[dk], sub[k], s=8, alpha=0.5)
    ax[1].plot([0, 1], [0, 1], color="grey", lw=1)
    ax[1].set(xlabel="DraftKings close (vig-free)", ylabel="Kalshi close (normalised mid)",
              title="Kalshi vs DraftKings, home win")

    s = sharp_diff[sharp_diff.metric == "brier"]
    v, lo, hi = s.value * 1000, s.ci_low * 1000, s.ci_high * 1000
    ax[2].errorbar(s.pair, v, yerr=[v - lo, hi - v], fmt="o", capsize=5)
    ax[2].axhline(0, color="grey", lw=1)
    ax[2].set(ylabel="Brier difference x 1000 (< 0: first venue sharper)",
              title=f"Paired Brier at close, 95% game-day CI (n={int(sharp.games.iloc[0])})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md(frame, floatfmt=".3f"):
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    table, meta = build_table()
    FROZEN.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FROZEN / "venue_prices.parquet", index=False)

    cov = coverage(table)
    gap = gaps(table)
    best = pd.concat([best_venue(table, "close", VENUES), best_venue(table, "close", ("kalshi", "draftkings")),
                      best_venue(table, "t1h", ("kalshi", "polymarket")),
                      best_venue(table, "t24h", ("kalshi", "polymarket"))], ignore_index=True)
    arb_rows, any_arb = [], {}
    for t in TIMES:
        rows, best_edge = arbs(table, t)
        arb_rows.append(rows)
        e = best_edge.dropna()
        any_arb[t] = {"games": len(e), "arb_games": int((e > 0).sum()),
                      "median_edge_c": float(e[e > 0].median() * 100) if (e > 0).any() else np.nan}
    arb = pd.concat(arb_rows, ignore_index=True)
    arb = arb[arb.games > 0]
    sharp, sharp_diff = sharpness(table, "close")

    RESULTS.mkdir(parents=True, exist_ok=True)
    agg = pd.concat([cov.assign(section="coverage"), gap.assign(section="gap"), best.assign(section="best_price"),
                     arb.assign(section="arbitrage"), sharp.assign(section="sharpness"),
                     sharp_diff.assign(section="sharpness_diff")], ignore_index=True)
    agg.to_csv(RESULTS / "venue_compare.csv", index=False)
    figure(table, sharp, sharp_diff, RESULTS / "venue_compare.png")
    write_md(table, meta, cov, gap, best, arb, any_arb, sharp, sharp_diff)
    print((RESULTS / "venue_compare.md").read_text())


def write_md(table, meta, cov, gap, best, arb, any_arb, sharp, sharp_diff):
    pm = meta["polymarket"]
    over = table["draftkings_close_overround"].dropna() * 100
    k_spread = ((table.kalshi_close_home_ask - table.kalshi_close_home_bid) * 100).dropna()
    arb_any = pd.DataFrame([{"time": t, **v} for t, v in any_arb.items()])
    arb_any["arb_share"] = arb_any.arb_games / arb_any.games.replace(0, np.nan)
    text = f"""# Game-winner prices across venues (test period {START} to {END})

`python -m evaluation.venue_compare` · per-game data in `data/frozen/venue_prices.parquet` (git-ignored) ·
aggregates in `venue_compare.csv` · figure `venue_compare.png`.

![Venue comparison](venue_compare.png)

## Sources

| Venue | Source | What we have |
|---|---|---|
| Kalshi | `KXNBAGAME` 1-minute candles (`data/frozen/prices.parquet`) | real top-of-book bid and ask, both team contracts |
| Polymarket | public CLOB `prices-history` (1-minute traded price), games looked up by slug `nba-<away>-<home>-<ET date>` | traded price only; ask = price + {HALF_SPREAD:.2f} (synthetic) |
| DraftKings | ESPN summary `pickcenter` (public, no key) | moneyline open and close only; no timestamps, so no tip - 24 h / tip - 1 h |

No other sportsbook, and no sharp book such as Pinnacle, is available from a free public no-key historical
source (ESPN's core odds API lists only DraftKings for these games and returns no line movement; historical
multi-book odds need a keyed API such as The Odds API, which we did not use).

Snapshots: last quote before tip - 24 h (max age 6 h), before tip - 1 h (max age 2 h) and before the
scheduled tip (close, max age 2 h). Fair home-win probability = mids normalised to sum to 1 (prediction
markets) or proportional vig removal (DraftKings). Fees: Kalshi taker 0.07·p·(1-p) per contract (unrounded;
real orders round up to the cent); Polymarket **current** sports taker fee 0.05·p·(1-p), makers free
(docs.polymarket.com/trading/fees, read 7 Oct 2026; we assume it applied in the test period, which may
overstate Polymarket's historical cost); DraftKings cost = 1 / decimal odds (vig inside the price).

## Coverage

{meta['games']} regular-season games in the test period. Polymarket: {pm['markets']} moneyline markets found,
{pm['matched']} matched to a game by team pair and date (±1 day).

{_md(cov.pivot(index="venue", columns="time", values="games").reset_index().fillna(0), ".0f")}

Median Kalshi home-contract spread at close: {k_spread.median():.1f}c (mean {k_spread.mean():.2f}c).
DraftKings overround at close: median {over.median():.1f}%, mean {over.mean():.1f}%.

## Cross-venue gap in fair home-win probability (percentage points)

{_md(gap, ".2f")}

## Best price after fees (who is cheapest for the side being bought)

Share of (game, side) pairs where each venue has the lowest all-in cost of a $1 payout, on games quoted by
every venue in the group; `mean_cost_over_best_c` is how much more than the cheapest venue it costs on average.

{_md(best[["time", "venues", "venue", "n", "share", "mean_cost_over_best_c"]], ".3f")}

## Arbitrage check (buy home on one venue, away on another)

Edge = 1 - (all-in home cost + all-in away cost); > 0 means a locked-in profit per $1 payout before
settlement risk. Any cross-venue pair, per game:

{_md(arb_any, ".3f")}

By combination:

{_md(arb, ".3f")}

Caveats: Polymarket's ask is synthetic (traded price + {HALF_SPREAD:.2f}), so its "arbs" are not executable
evidence: the real book may be wider, and a stale last trade can sit far from the live quote. DraftKings
lines have no timestamp and the "close" is ESPN's final pre-game line; real bet limits and line moves on
arrival matter. Settlement rules differ (Kalshi and Polymarket settle on the official result incl. overtime,
as does the sportsbook moneyline, but void/postponement rules differ). Latency, top-of-book size (often a few
hundred dollars on Kalshi), capital locked until settlement, and Polymarket USDC transfer costs are ignored.

## Which close is sharpest (vs actual outcomes)

Games quoted by every venue at close; 95% CIs from a game-day bootstrap ({REPS} reps).

{_md(sharp[["venue", "metric", "games", "days", "value", "ci_low", "ci_high"]], ".4f")}

Paired differences (negative = first venue sharper):

{_md(sharp_diff[["pair", "metric", "value", "ci_low", "ci_high"]], ".4f")}

DraftKings open (vig-free) for reference: Brier {_brier(table.draftkings_open_p_home, table.home_win).mean():.4f}
on {int(table.draftkings_open_p_home.notna().sum())} games.
"""
    (RESULTS / "venue_compare.md").write_text(text)


if __name__ == "__main__":
    main()
