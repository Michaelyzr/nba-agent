"""Shared helpers for the edge scan: as-of Kalshi quotes, fee-adjusted CLV, day-block bootstrap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "edge_scan"
RESULTS = ROOT / "evaluation" / "results"
FALLBACK = Path("/tmp/nba_frozen")
STALE = pd.Timedelta("30min")


def fee(p):
    return 0.07 * p * (1 - p)


def frozen(name: str) -> pd.DataFrame:
    from data_sources import FROZEN
    path = FROZEN / f"{name}.parquet"
    return pd.read_parquet(path if path.exists() else FALLBACK / f"{name}.parquet")


class Quotes:
    """As-of lookups on 1-minute Kalshi game-winner quotes."""

    def __init__(self):
        p = frozen("prices")
        p = p[(p.bid > 0) & (p.ask < 1) & (p.ask >= p.bid)].sort_values(["market_ticker", "ts"])
        self.ts, self.bid, self.ask = {}, {}, {}
        for t, g in p.groupby("market_ticker", sort=False):
            self.ts[t] = g.ts.to_numpy("datetime64[ns]")
            self.bid[t], self.ask[t] = g.bid.to_numpy(), g.ask.to_numpy()
        m = frozen("markets")
        m = m[m.kind == "game"]
        self.ticker = {(str(g), t): k for g, t, k in zip(m.game_id, m.team, m.market_ticker)}
        games = frozen("games")
        self.games = games.assign(game_id=games.game_id.astype(str), tip_time=pd.to_datetime(games.tip_time, utc=True),
                                  date=games.date.astype(str).str[:10]).set_index("game_id")

    def quote(self, ticker, when, after=False):
        """(bid, ask, ts) of the last quote at or before `when` (or first at/after it if after=True)."""
        ts = self.ts.get(ticker)
        if ts is None:
            return None
        w = np.datetime64(pd.Timestamp(when).tz_convert("UTC").tz_localize(None), "ns")
        if after:
            i = np.searchsorted(ts, w, side="left")
            if i >= len(ts) or ts[i] - w > STALE.to_timedelta64():
                return None
        else:
            i = np.searchsorted(ts, w, side="right") - 1
            if i < 0 or w - ts[i] > STALE.to_timedelta64():
                return None
        return self.bid[ticker][i], self.ask[ticker][i], ts[i]

    def mid(self, ticker, when, after=False):
        q = self.quote(ticker, when, after)
        return None if q is None else (q[0] + q[1]) / 2

    def tip(self, game_id):
        return self.games.tip_time[str(game_id)]

    def trade_clv(self, ticker, side, when, after=False, gross=False):
        """Buy `side` at the touch at `when`; CLV per contract after fee vs the side's mid at tip.

        gross=True: mid-to-mid move in the side's favour, before spread and fee."""
        q = self.quote(ticker, when, after)
        game = ticker_game(self, ticker)
        close = self.mid(ticker, self.tip(game)) if game else None
        if q is None or close is None:
            return np.nan
        price = q[1] if side == "yes" else 1 - q[0]
        value = close if side == "yes" else 1 - close
        if gross:
            mid = (q[0] + q[1]) / 2
            return value - (mid if side == "yes" else 1 - mid)
        return value - price - fee(price)


def ticker_game(quotes: Quotes, ticker: str):
    if not hasattr(quotes, "_tg"):
        quotes._tg = {k: g for (g, _), k in quotes.ticker.items()}
    return quotes._tg.get(ticker)


def day_boot(values, days, reps=2000, seed=7606):
    """Mean of per-trade values with a game-day block bootstrap 95% CI."""
    df = pd.DataFrame({"v": np.asarray(values, float), "d": list(days)}).dropna()
    if df.empty:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "days": 0, "share_pos": np.nan}
    g = df.groupby("d").v.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(reps, len(g)))
    boots = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return {"n": len(df), "mean": float(df.v.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "days": len(g), "share_pos": float((df.v > 0).mean())}
