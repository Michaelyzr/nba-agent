"""Part 3b: trade timing. CLV after fees by minutes-to-tip and news age, plus two simple timing rules.

Inputs: the anchor agent's decision points with their model-vs-market gap (runs/quick_wins/prefilter_*.parquet,
Dec-Jan = tune, Feb-Apr 12 = test) and its filled trades (runs/quick_wins/shift_fixed4_empty/anchor).
    python -m evaluation.edge_scan_timing   -> runs/edge_scan/timing.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.edge_scan_common import ROOT, RUNS, Quotes, day_boot, frozen

QW = ROOT / "runs" / "quick_wins"
TTT_BINS = [0, 30, 60, 120, 240, 2000]
DELAYS = (0, 5, 10, 20)
FADE_KS = (0.03, 0.05, 0.08, 0.10, 0.15)


def decision_points(q: Quotes, news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, f in (("tune", "prefilter_dec_jan"), ("test", "prefilter_test")):
        d = pd.read_parquet(QW / f"{f}.parquet").dropna(subset=["gap"]).reset_index(drop=True)
        d["as_of"] = pd.to_datetime(d.as_of, utc=True)
        d = d.loc[d.gap.abs().groupby([d.game_id, d.as_of]).idxmax()]
        for r in d.itertuples():
            t = q.ticker.get((str(r.game_id), r.team))
            if t is None:
                continue
            side = "yes" if r.gap > 0 else "no"
            tip = q.tip(r.game_id)
            nw = news[(news.game_id == str(r.game_id)) & (news.published_at <= r.as_of)]
            last = nw.published_at.max() if len(nw) else pd.NaT
            row = {"split": split, "game_id": str(r.game_id), "date": str(q.games.date[str(r.game_id)])[:10],
                   "as_of": r.as_of, "gap": r.gap, "abs_gap": abs(r.gap), "ticker": t, "side": side,
                   "mins_to_tip": (tip - r.as_of).total_seconds() / 60,
                   "news_age": (r.as_of - last).total_seconds() / 60 if pd.notna(last) else np.nan,
                   "clv_trigger": q.trade_clv(t, side, r.as_of),
                   "gross_trigger": q.trade_clv(t, side, r.as_of, gross=True),
                   "clv_after_news": q.trade_clv(t, side, last, after=True) if pd.notna(last) else np.nan}
            for dl in DELAYS:
                when = min(r.as_of + pd.Timedelta(minutes=dl), tip - pd.Timedelta(minutes=1))
                row[f"clv_d{dl}"] = q.trade_clv(t, side, when)
                row[f"gross_d{dl}"] = q.trade_clv(t, side, when, gross=True)
            rows.append(row)
    return pd.DataFrame(rows)


def fade_points(q: Quotes) -> pd.DataFrame:
    """Every game, decision at tip-120/-60/-30 min: move since the 24 h anchor; fade = buy the side that fell."""
    g = q.games
    g = g[(g.date >= "2025-12-01") & (g.date <= "2026-04-12")]
    rows = []
    for gid, r in g.iterrows():
        t = q.ticker.get((gid, r.home_team))
        if t is None:
            continue
        for m in (120, 60, 30):
            when = r.tip_time - pd.Timedelta(minutes=m)
            now, anchor = q.mid(t, when), q.mid(t, r.tip_time - pd.Timedelta(hours=24), after=True)
            if now is None or anchor is None:
                continue
            move = now - anchor
            side = "no" if move > 0 else "yes"
            rows.append({"split": "tune" if str(r.date)[:10] < "2026-02-01" else "test", "game_id": gid,
                         "date": str(r.date)[:10], "mins_to_tip": m, "move": move,
                         "clv_fade": q.trade_clv(t, side, when),
                         "gross_fade": q.trade_clv(t, side, when, gross=True),
                         "clv_follow": q.trade_clv(t, "yes" if side == "no" else "no", when)})
    return pd.DataFrame(rows)


def row(part, split, label, vals, days, gross=None, **extra):
    g = day_boot(gross, days) if gross is not None else {}
    return {"part": "3b", "analysis": part, "split": split, "bucket": label, **day_boot(vals, days),
            "gross_mean": g.get("mean", np.nan), "gross_lo": g.get("lo", np.nan), "gross_hi": g.get("hi", np.nan),
            **extra}


def main():
    q = Quotes()
    news = frozen("news")
    news = news.assign(game_id=news.game_id.astype(str))
    dp = decision_points(q, news)
    dp.to_parquet(RUNS / "timing_points.parquet", index=False)
    out = []
    for split in ("tune", "test"):
        s = dp[(dp.split == split) & (dp.abs_gap >= 0.02)]
        for lo_gap, name in ((0.04, "trades |gap|>=4pp"), (0.02, "trades+near-misses |gap|>=2pp")):
            x = s[s.abs_gap >= lo_gap]
            out.append(row("all", split, name, x.clv_trigger, x.date, x.gross_trigger))
            for a, b in zip(TTT_BINS[:-1], TTT_BINS[1:]):
                y = x[(x.mins_to_tip > a) & (x.mins_to_tip <= b)]
                if len(y):
                    out.append(row("mins_to_tip", split, f"{name}: {a}-{b} min", y.clv_trigger, y.date,
                                   y.gross_trigger))
            for label, y in (("ESPN inactive news present", x[x.news_age.notna()]), ("no news", x[x.news_age.isna()])):
                out.append(row("news_age", split, f"{name}: {label}", y.clv_trigger, y.date, y.gross_trigger))
            for d in DELAYS:
                out.append(row("entry_delay", split, f"{name}: enter trigger+{d} min", x[f"clv_d{d}"], x.date,
                               x[f"gross_d{d}"]))
    fp = fade_points(q)
    fp.to_parquet(RUNS / "fade_points.parquet", index=False)
    for split in ("tune", "test"):
        for k in FADE_KS:
            y = fp[(fp.split == split) & (fp.move.abs() >= k)]
            out.append(row("fade_24h_move", split, f"fade |move|>={k:.2f}", y.clv_fade, y.date, y.gross_fade))
            out.append(row("fade_24h_move", split, f"follow |move|>={k:.2f}", y.clv_follow, y.date, -y.gross_fade))
    fills = pd.read_parquet(QW / "shift_fixed4_empty" / "anchor" / "fills.parquet")
    fills = fills.assign(game_id=fills.game_id.astype(str))
    fills["mins_to_tip"] = [(q.tip(g) - a).total_seconds() / 60 for g, a in zip(fills.game_id, fills.as_of)]
    fills["clv_net"] = fills.clv - fills.fee / fills.contracts
    fills["date"] = [str(q.games.date[g])[:10] for g in fills.game_id]
    fills["gross"] = [q.trade_clv(t, s, a, gross=True) for t, s, a in zip(fills.market_ticker, fills.side, fills.as_of)]
    out.append(row("anchor_fills", "test", "published fixed4 anchor fills (all)", fills.clv_net, fills.date,
                   fills.gross))
    for a, b in zip(TTT_BINS[:-1], TTT_BINS[1:]):
        y = fills[(fills.mins_to_tip > a) & (fills.mins_to_tip <= b)]
        if len(y):
            out.append(row("anchor_fills", "test", f"anchor fills {a}-{b} min", y.clv_net, y.date, y.gross))
    res = pd.DataFrame(out)
    res.to_csv(RUNS / "timing.csv", index=False)
    print("decision-point minutes to tip:", dp.mins_to_tip.round().value_counts().head().to_dict(),
          "| news age at decision:", dp.news_age.round().value_counts().head().to_dict())
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res[["analysis", "split", "bucket", "n", "days", "mean", "lo", "hi", "gross_mean", "gross_lo",
                   "gross_hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
