"""Is a sports prediction market a fair game for a retail user? Evidence from our recorded Kalshi prices.

    python -m evaluation.consumer_fairness          # data/frozen, writes evaluation/results/fairness_* and PNGs

1. Cost burden: what a $20 retail order pays over the fair mid (half the bid-ask
   spread plus the Kalshi fee) at tip - 60 min, by price bucket; and how much of
   each replay setup's loss was costs (fills from runs/results/<setup>/).
2. Price discovery: on test games where M4 says the inactive list moved the
   home team's chances, how far the price had already moved in that direction
   before the list became public (tip - 30 min) vs after it.
3. Long-shot calibration: implied (mid) vs realised win rate by price bucket,
   and the return of buying every contract in a bucket at the ask plus fee.
4. Simple strategies a consumer might follow (every favourite, every underdog,
   every contract) at tip - 60 min.

Market-wide properties (1, 3, 4) use every settled 2025-26 game with a quote;
model-dependent ones (2, replay costs) use the 1 Feb - 12 Apr 2026 test period.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from replay import kalshi_fee, load_tables

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = Path(__file__).resolve().parent.parent / "runs" / "results"
STAKE = 20.0
RETAIL_LEAD = pd.Timedelta(minutes=60)      # when the replay's "plain" consumer looks at the price
ANCHOR_LEAD = pd.Timedelta(hours=24)
MAX_AGE = pd.Timedelta(minutes=60)          # ignore quotes older than this at the decision moment
BUCKETS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SETUPS = {"test-no-agent": "Plain model, no agent", "test-no-anchor": "Agent, raw model",
          "test-placeholder-model": "Agent, win-rate placeholder", "test-no-learning": "Agent, anchor, no learning",
          "test-full": "Full agent (anchor + learning)"}


def _ns(ts: pd.Series) -> np.ndarray:
    return pd.to_datetime(ts, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()


class Quotes:
    """Last recorded bid/ask at or before a time, per market."""

    def __init__(self, prices: pd.DataFrame):
        prices = prices.sort_values("ts")
        self.idx = {k: (_ns(v.ts), v.bid.to_numpy(float), v.ask.to_numpy(float))
                    for k, v in prices.groupby("market_ticker")}

    def at(self, ticker, t, max_age=None, earliest=False):
        if ticker not in self.idx:
            return None
        ts, bid, ask = self.idx[ticker]
        now = pd.Timestamp(t).tz_convert("UTC").value
        i = int(np.searchsorted(ts, now, side="right"))
        if i == 0:
            if not earliest:
                return None
            i = 1
        if max_age is not None and now - ts[i - 1] > max_age.value:
            return None
        b, a = bid[i - 1], ask[i - 1]
        if not (0 < b < a < 1):
            return None
        return b, a


def market_rows(tables, q: Quotes) -> pd.DataFrame:
    """One row per (game, team YES contract) with quotes at the anchor, tip - 60, news time and tip."""
    games = tables["games"].dropna(subset=["home_pts", "tip_time"]).set_index("game_id")
    news = tables["news"]
    first_news = news.groupby("game_id").published_at.min()
    outcomes = dict(zip(tables["settlements"].market_ticker, tables["settlements"].outcome))
    rows = []
    for m in tables["markets"][tables["markets"].kind == "game"].itertuples():
        if m.game_id not in games.index or m.market_ticker not in outcomes:
            continue
        g = games.loc[m.game_id]
        tip = pd.Timestamp(g.tip_time)
        row = {"game_id": m.game_id, "date": g.date, "ticker": m.market_ticker, "team": m.team,
               "is_home": m.team == g.home_team, "outcome": float(outcomes[m.market_ticker]),
               "has_news": m.game_id in first_news.index}
        for name, t, kw in (("anchor", tip - ANCHOR_LEAD, {"earliest": True}),
                            ("retail", tip - RETAIL_LEAD, {"max_age": MAX_AGE}),
                            ("tip", tip, {"max_age": MAX_AGE})):
            v = q.at(m.market_ticker, t, **kw)
            row[f"bid_{name}"], row[f"ask_{name}"] = v if v else (np.nan, np.nan)
        nt = first_news.get(m.game_id)
        v = q.at(m.market_ticker, nt, max_age=MAX_AGE) if nt is not None and nt < tip else None
        row["bid_news"], row["ask_news"] = v if v else (np.nan, np.nan)
        rows.append(row)
    t = pd.DataFrame(rows)
    for name in ("anchor", "retail", "news", "tip"):
        t[f"mid_{name}"] = (t[f"bid_{name}"] + t[f"ask_{name}"]) / 2
    return t


def buy_cost(ask: float) -> dict:
    """All-in cost of a $STAKE YES order at the ask, per contract."""
    contracts = int(np.floor(STAKE / ask + 1e-9))
    fee = kalshi_fee(contracts, ask) / contracts if contracts else np.nan
    return {"contracts": contracts, "fee_pc": fee}


def cost_burden(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    r = t.dropna(subset=["ask_retail"]).copy()
    r[["contracts", "fee_pc"]] = pd.DataFrame([buy_cost(a) for a in r.ask_retail], index=r.index)
    r["half_spread"] = r.ask_retail - r.mid_retail
    r["cost"] = r.half_spread + r.fee_pc
    r["cost_pct_price"] = r.cost / r.ask_retail
    r["breakeven"] = r.ask_retail + r.fee_pc
    r["bucket"] = pd.cut(r.mid_retail, BUCKETS, include_lowest=True)
    by = r.groupby("bucket", observed=True).agg(
        contracts=("cost", "size"), mid=("mid_retail", "mean"), half_spread=("half_spread", "mean"),
        fee=("fee_pc", "mean"), cost=("cost", "mean"), cost_pct_price=("cost_pct_price", "mean")).reset_index()
    by["bucket"] = by.bucket.astype(str)
    overall = {"contracts": len(r), "games": r.game_id.nunique(),
               "mean_spread": float((r.ask_retail - r.bid_retail).mean()),
               "mean_half_spread": float(r.half_spread.mean()), "mean_fee": float(r.fee_pc.mean()),
               "mean_cost": float(r.cost.mean()), "mean_cost_pct_price": float(r.cost_pct_price.mean()),
               "median_cost_pct_price": float(r.cost_pct_price.median())}
    return by, overall


def run_costs() -> pd.DataFrame:
    """For each replay setup: P&L after costs, fees paid, and half-spread paid over the mid at the fill."""
    rows = []
    tables = None
    for name, label in SETUPS.items():
        path = RUNS / name / "fills.parquet"
        if not path.exists():
            continue
        f = pd.read_parquet(path).dropna(subset=["outcome"])
        if f.empty:
            continue
        if tables is None:
            tables = load_tables()
            quotes = Quotes(tables["prices"])
        mids = []
        for x in f.itertuples():
            v = quotes.at(x.market_ticker, pd.Timestamp(x.quote_ts))
            mid = (v[0] + v[1]) / 2 if v else np.nan
            mids.append(mid if x.side == "yes" else 1 - mid)
        f["mid_side"] = mids
        spread_paid = float((f.contracts * (f.price - f.mid_side)).sum())
        fees = float(f.fee.sum())
        pnl = float(f.pnl.sum())
        rows.append({"setup": label, "trades": len(f), "staked": float((f.price * f.contracts).sum()),
                     "pnl_after_costs": pnl, "fees_paid": fees, "spread_paid": spread_paid,
                     "pnl_at_mid_before_fees": pnl + fees + spread_paid,
                     "beat_or_matched_close": int((f.clv >= 0).sum()),
                     "won": int(((f.outcome == 1) == (f.side == "yes")).sum())})
    return pd.DataFrame(rows)


def price_discovery(t: pd.DataFrame, m4: pd.DataFrame, min_shift=0.01) -> tuple[pd.DataFrame, dict]:
    """Move of the home price in the direction of the M4 news shift, before vs after the list is public."""
    home = t[t.is_home].dropna(subset=["mid_anchor", "mid_news", "mid_tip"])
    d = home.merge(m4[["game_id", "players_out", "m4_before", "m4_after"]], on="game_id")
    d["news_shift"] = d.m4_after - d.m4_before
    d = d[(d.players_out > 0) & (d.news_shift.abs() >= min_shift)].copy()
    sign = np.sign(d.news_shift)
    d["before_news"] = (d.mid_news - d.mid_anchor) * sign
    d["after_news"] = (d.mid_tip - d.mid_news) * sign
    d["total"] = d.before_news + d.after_news
    moved = d[d.total > 0]
    out = {"games": len(d), "mean_shift_abs": float(d.news_shift.abs().mean()),
           "mean_before": float(d.before_news.mean()), "mean_after": float(d.after_news.mean()),
           "share_before": float(d.before_news.sum() / d.total.sum()) if d.total.sum() > 0 else np.nan,
           "games_moved_with_news": len(moved),
           "share_before_moved": float(moved.before_news.sum() / moved.total.sum()) if len(moved) else np.nan,
           "after_news_abs_le_1pt": float((d.after_news.abs() <= 0.01).mean())}
    return d, out


def calibration(t: pd.DataFrame) -> pd.DataFrame:
    r = t.dropna(subset=["ask_retail"]).copy()
    r["fee_pc"] = [buy_cost(a)["fee_pc"] for a in r.ask_retail]
    r["ret"] = (r.outcome - r.ask_retail - r.fee_pc) / (r.ask_retail + r.fee_pc)
    r["bucket"] = pd.cut(r.mid_retail, BUCKETS, include_lowest=True)
    by = r.groupby("bucket", observed=True).agg(contracts=("outcome", "size"), implied=("mid_retail", "mean"),
                                               realised=("outcome", "mean"), breakeven=("ask_retail", "mean"),
                                               roi_at_ask_plus_fee=("ret", "mean")).reset_index()
    by["breakeven"] = by.breakeven + r.groupby("bucket", observed=True).fee_pc.mean().to_numpy()
    by["se"] = np.sqrt(by.implied * (1 - by.implied) / by.contracts)
    by["z_realised_minus_implied"] = (by.realised - by.implied) / by.se.replace(0, np.nan)
    by["bucket"] = by.bucket.astype(str)
    return by


def strategies(t: pd.DataFrame) -> pd.DataFrame:
    """$20 at the ask + fee on every contract matching a simple rule, at tip - 60 min."""
    r = t.dropna(subset=["ask_retail"]).copy()
    r[["contracts", "fee_pc"]] = pd.DataFrame([buy_cost(a) for a in r.ask_retail], index=r.index)
    r["pnl"] = r.contracts * (r.outcome - r.ask_retail - r.fee_pc)
    r["staked"] = r.contracts * (r.ask_retail + r.fee_pc)
    rules = {"Every contract (both sides)": r.mid_retail > 0,
             "Every favourite (price > 50¢)": r.mid_retail > 0.5,
             "Every underdog (price < 50¢)": r.mid_retail < 0.5,
             "Long shots (price ≤ 25¢)": r.mid_retail <= 0.25,
             "Heavy favourites (price ≥ 75¢)": r.mid_retail >= 0.75,
             "Every home team": r.is_home}
    rows = []
    for name, mask in rules.items():
        x = r[mask]
        rows.append({"strategy": name, "bets": len(x), "staked": float(x.staked.sum()), "pnl": float(x.pnl.sum()),
                     "roi": float(x.pnl.sum() / x.staked.sum()), "win_rate": float(x.outcome.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_costs(by: pd.DataFrame, overall: dict, runs: pd.DataFrame, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    x = np.arange(len(by))
    labels = [f"{int(round(lo * 100))}–{int(round(hi * 100))}¢" for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])][:len(by)]
    a.bar(x, by.half_spread * 100, color="#94a3b8", label="half the bid-ask spread")
    a.bar(x, by.fee * 100, bottom=by.half_spread * 100, color="#ea580c", label="Kalshi fee 7%·p·(1−p)")
    for i, r in by.iterrows():
        a.text(i, (r.cost) * 100 + 0.08, f"{r.cost_pct_price:.0%}\nof price", ha="center", fontsize=8.5)
    a.set_xticks(x, labels, fontsize=9)
    a.set(xlabel="contract price (mid) at tip − 60 min", ylabel="cost over the fair mid (¢ per contract)",
          ylim=(0, by.cost.max() * 100 * 1.35),
          title=f"What a $20 order pays over the mid: {overall['mean_cost'] * 100:.1f}¢ per contract on average")
    a.legend(loc="upper right", fontsize=9)

    a = ax[1]
    if not runs.empty:
        y = np.arange(len(runs))
        a.barh(y - 0.2, runs.pnl_at_mid_before_fees, 0.4, color="#94a3b8", label="at the mid, before fees")
        a.barh(y + 0.2, runs.pnl_after_costs, 0.4, color="#dc2626", label="after spread + fees (what you get)")
        for i, r in runs.reset_index(drop=True).iterrows():
            a.text(min(r.pnl_after_costs, r.pnl_at_mid_before_fees, 0) - 40, i + 0.2,
                   f"{'−' if r.pnl_after_costs < 0 else '+'}${abs(r.pnl_after_costs):,.0f}",
                   va="center", ha="right", fontsize=9)
        a.set_yticks(y, [f"{s}\n({n} trades)" for s, n in zip(runs.setup, runs.trades)], fontsize=9)
        a.axvline(0, color="black", lw=0.8)
        lo = min(runs.pnl_after_costs.min(), runs.pnl_at_mid_before_fees.min())
        a.set(xlim=(lo * 1.35 - 50, max(runs.pnl_at_mid_before_fees.max(), 0) + 150),
              xlabel="test-period P&L ($)", title="Replay setups: P&L before and after trading costs")
        a.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_discovery(d: pd.DataFrame, s: dict, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
    a = ax[0]
    vals = [s["mean_before"] * 100, s["mean_after"] * 100]
    bars = a.bar(["24 h before tip →\nlist public (tip − 30 min)", "list public →\ntip-off"], vals,
                 color=["#2563eb", "#ea580c"], width=0.55)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:+.2f} pts", ha="center", fontsize=11, weight="bold")
    a.axhline(0, color="black", lw=0.8)
    a.set(ylabel="mean price move in the news direction (pts)", ylim=(min(0, min(vals)) - 0.5, max(vals) * 1.3 + 0.3),
          title=f"{s['games']} test games where the inactive list moved M4 ≥ 1 pt")
    a = ax[1]
    a.scatter(d.before_news * 100, d.after_news * 100, s=14, alpha=0.5, color="#475569")
    lim = np.nanpercentile(np.abs(np.r_[d.before_news, d.after_news]) * 100, 98) + 1
    a.axhline(0, color="grey", lw=0.8)
    a.axvline(0, color="grey", lw=0.8)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="move before the list was public (pts, news direction)",
          ylabel="move after the list was public (pts)", title="Per game: most of the move is already in the price")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(cal: pd.DataFrame, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    a = ax[0]
    a.fill_between(cal.implied, cal.implied - 1.96 * cal.se, cal.implied + 1.96 * cal.se, color="#cbd5e1",
                   label="95% range if prices are fair")
    a.plot(cal.implied, cal.realised, "o-", color="#2563eb", label="realised win rate")
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="price (implied probability) at tip − 60 min",
          ylabel="share of contracts that won", title="Are prices calibrated? (every contract, 2025-26)")
    a.legend(loc="upper left", fontsize=9)
    a = ax[1]
    labels = [f"{int(round(lo * 100))}–{int(round(hi * 100))}¢" for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])][:len(cal)]
    colors = ["#dc2626" if v < 0 else "#16a34a" for v in cal.roi_at_ask_plus_fee]
    a.bar(labels, cal.roi_at_ask_plus_fee * 100, color=colors)
    for i, r in cal.iterrows():
        a.text(i, r.roi_at_ask_plus_fee * 100 + (1 if r.roi_at_ask_plus_fee >= 0 else -1), f"n={r.contracts}",
               ha="center", va="bottom" if r.roi_at_ask_plus_fee >= 0 else "top", fontsize=8)
    a.axhline(0, color="black", lw=0.8)
    a.set(xlabel="contract price bucket", ylabel="return on stake, buying at ask + fee (%)",
          title="Return of buying every contract in a bucket")
    a.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(overall, by, runs, disc, cal, strat) -> str:
    def tbl(df):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(4)
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)] +
                         ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])

    full = runs[runs.setup.str.startswith("Full agent")]
    full_line = ""
    if len(full):
        f = full.iloc[0]
        full_line = (f"- **Costs decide the sign.** At the mid with no fee the full agent would have made "
                     f"{'+' if f.pnl_at_mid_before_fees >= 0 else '−'}${abs(f.pnl_at_mid_before_fees):,.0f}; "
                     f"after ${f.fees_paid:,.0f} fees and ${f.spread_paid:,.0f} "
                     f"spread it lost ${-f.pnl_after_costs:,.0f}.\n")
    cheap = cal.iloc[0]
    sig = cal[cal.z_realised_minus_implied.abs() >= 2]
    longshot = (f"- **Long-shot bias: not clearly supported.** Only the cheapest bucket hints at it "
                f"({cheap.realised * cheap.contracts:.0f} win in {cheap.contracts} contracts vs {cheap.implied:.1%} "
                f"implied, z = {cheap.z_realised_minus_implied:.1f}); {len(cal) - len(sig)} of {len(cal)} buckets "
                f"are within 2 standard errors of fair"
                + (f" (outside: {', '.join(sig.bucket)}, the mirror side of the cheap contracts)" if len(sig) else "")
                + ". Long shots still lose more per dollar because costs "
                f"are a bigger share of a cheap price.\n")
    summary = (f"## Summary\n\n"
               f"- **Costs:** a $20 order pays {overall['mean_cost'] * 100:.2f}¢ per contract over the mid "
               f"({overall['mean_cost_pct_price']:.1%} of the price): the edge needed just to break even.\n"
               f"{full_line}"
               f"- **Information speed:** {disc['share_before']:.0%} of the news-direction price move came "
               f"before the inactive list was public ({disc['mean_before'] * 100:+.2f} vs "
               f"{disc['mean_after'] * 100:+.2f} pts, {disc['games']} test games).\n"
               f"{longshot}")
    return f"""# Consumer fairness: evidence from recorded Kalshi NBA game-winner prices

{summary}
Generated by `python -m evaluation.consumer_fairness`. Market-wide numbers use every settled
2025-26 game with a quote at tip − 60 min; replay and price-discovery numbers use the test
period (1 Feb – 12 Apr 2026). A $20 order buys at the ask and pays the Kalshi fee
(7% · contracts · p · (1 − p), rounded up to the cent).

## 1. Cost of a $20 order over the fair mid, at tip − 60 min

{overall['contracts']} contracts ({overall['games']} games). Mean bid-ask spread {overall['mean_spread'] * 100:.2f}¢;
mean half-spread {overall['mean_half_spread'] * 100:.2f}¢; mean fee {overall['mean_fee'] * 100:.2f}¢ per contract;
**mean all-in cost {overall['mean_cost'] * 100:.2f}¢ per contract = {overall['mean_cost_pct_price']:.1%} of the price**
(median {overall['median_cost_pct_price']:.1%}). To break even a buyer must be right more often than the price
says by that many points.

{tbl(by)}

## 2. Replay setups: how much of the loss was trading costs (test period)

`pnl_at_mid_before_fees` = P&L if every fill had been at the mid with no fee.

{tbl(runs)}

## 3. Price discovery around the inactive list (test period)

Games where at least one player was ruled out and M4's home win probability moved by at least
1 point when the inactive list arrived. Moves are signed in the direction of the M4 shift.
The inactive list is stamped tip − 30 min (our proxy for when a retail user sees it).

- games: {disc['games']}; mean |M4 shift| {disc['mean_shift_abs'] * 100:.2f} pts
- mean move 24 h before tip → list public: **{disc['mean_before'] * 100:+.2f} pts**
- mean move list public → tip: **{disc['mean_after'] * 100:+.2f} pts**
- share of the net news-direction move that happened before the list was public: {disc['share_before']:.0%}
- games where the price moved with the news overall: {disc['games_moved_with_news']}; share before the list in those: {disc['share_before_moved']:.0%}
- games where the price moved by at most 1 pt after the list: {disc['after_news_abs_le_1pt']:.0%}

## 4. Calibration by price bucket (long-shot check)

`z_realised_minus_implied` is (realised − implied) / binomial standard error at the implied
probability. |z| < 2 means the bucket is consistent with fair pricing at this sample size. Each
game has two mirror-image contracts, so the 0–10¢ bucket is the 90–100¢ bucket seen from the other side.

{tbl(cal)}

## 5. Simple consumer strategies at tip − 60 min ($20 at ask + fee)

{tbl(strat)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    tables = load_tables()
    q = Quotes(tables["prices"])
    t = market_rows(tables, q)
    by, overall = cost_burden(t)
    runs = run_costs()
    m4 = pd.read_csv(RESULTS / "m4_games.csv", dtype={"game_id": str})
    t["game_id"] = t.game_id.astype(str)
    d, disc = price_discovery(t, m4)
    cal = calibration(t)
    strat = strategies(t)

    by.to_csv(args.out / "fairness_costs.csv", index=False)
    pd.DataFrame([overall]).to_csv(args.out / "fairness_cost_summary.csv", index=False)
    runs.to_csv(args.out / "fairness_run_costs.csv", index=False)
    d.to_csv(args.out / "fairness_price_discovery_games.csv", index=False)
    pd.DataFrame([disc]).to_csv(args.out / "fairness_price_discovery.csv", index=False)
    cal.to_csv(args.out / "fairness_calibration.csv", index=False)
    strat.to_csv(args.out / "fairness_strategies.csv", index=False)
    fig_costs(by, overall, runs, args.out / "cost_burden.png")
    fig_discovery(d, disc, args.out / "price_discovery.png")
    fig_calibration(cal, args.out / "longshot_calibration.png")
    (args.out / "consumer_fairness.md").write_text(markdown(overall, by, runs, disc, cal, strat))
    print((args.out / "consumer_fairness.md").read_text())


if __name__ == "__main__":
    main()
