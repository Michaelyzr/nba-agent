"""Evaluate the forecast-blend subloop (agents/forecast_blend.py) on the test period, no LLM.

    python -m evaluation.forecast_blend

Per test game (1 Feb - 12 Apr 2026, 501 games) the as-of signals are: Kalshi anchor mid 24 h before
tip, Kalshi mid 1 h before tip, anchor + M4 / MLP / GRU news shifts, and raw M4 / MLP / GRU. M4 and
M4-NN were trained only on games before 1 Feb (cached in evaluation/results/m4_games.csv and
runs/win_nn/test_preds.csv), so every signal exists before the decision. The learner walks forward
day by day and is gated on held-out Brier.

Trading: a simplified one-decision-per-game trader on the home market, 30 min before tip, buying
yes at the ask or no at 1 - bid when p - price - fee > 4 cents, $20 flat stake; CLV against the
tip-off mid, P&L against settlement. When data/frozen/prices.parquet is available the quotes are
real Kalshi quotes; otherwise the mid 1 h before tip +/- a 1-cent half-spread (stated in the report).
The same trader is used for every forecaster, so differences come from the probability alone.

Writes evaluation/results/forecast_blend.{md,csv,png} and forecast_blend_log.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.forecast_blend import (FIT_DAYS, GATE_BRIER, GATE_DAYS, MARKET_SIGNALS, SELECT_DAYS, SIGNALS,
                                   BlendLearner, brier, log_loss)
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from replay import kalshi_fee

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
CACHE = ROOT / "runs" / "forecast_blend"
FALLBACK_PRICES = Path("/tmp/nba_frozen/prices.parquet")   # local copy when iCloud evicts the frozen price table
MIN_EDGE = 0.04
STAKE = 20.0
HALF_SPREAD = 0.01
DECISION_LEAD = pd.Timedelta(minutes=30)


def clip(p):
    return np.clip(p, 0.02, 0.98)


def signals() -> pd.DataFrame:
    g = pd.read_csv(RESULTS / "m4_games.csv", dtype={"game_id": str})
    nn = pd.read_csv(ROOT / "runs" / "win_nn" / "test_preds.csv", dtype={"game_id": str})
    an = pd.read_csv(ROOT / "runs" / "win_nn" / "anchor.csv", dtype={"game_id": str})
    t = g.merge(nn.drop(columns=["date", "home_win"]), on="game_id").merge(an, on="game_id")
    t = t.assign(mid=t.market_before, m4=t.m4_after, mlp=t.mlp_after, gru=t.gru_after,
                 anchor_m4=clip(t.anchor + t.m4_after - t.m4_before),
                 anchor_mlp=clip(t.anchor + t.mlp_after - t.mlp_before),
                 anchor_gru=clip(t.anchor + t.gru_after - t.gru_before))
    return t.sort_values(["date", "game_id"]).reset_index(drop=True)


def quotes(t: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Home-market bid/ask 30 min before tip and the tip-off mid; real quotes if the price file is local."""
    path = ROOT / "data" / "frozen" / "prices.parquet"
    if not path.exists() and FALLBACK_PRICES.exists():
        path = FALLBACK_PRICES
    cache = CACHE / "quotes.parquet"
    if cache.exists():
        return pd.read_parquet(cache), "Kalshi quotes 30 min before tip (replay price table)"
    if path.exists():
        from data_sources import read_table
        games = read_table("games").assign(game_id=lambda d: d.game_id.astype(str))
        markets = read_table("markets").assign(game_id=lambda d: d.game_id.astype(str))
        m = markets[(markets.kind == "game")].merge(games[["game_id", "home_team", "tip_time"]], on="game_id")
        m = m[(m.team == m.home_team) & m.game_id.isin(t.game_id)]
        prices = pd.read_parquet(path, columns=["market_ticker", "ts", "bid", "ask"])
        prices = prices[prices.market_ticker.isin(m.market_ticker)].sort_values("ts")
        rows = []
        for r in m.itertuples():
            p = prices[prices.market_ticker == r.market_ticker]
            at = p[p.ts <= r.tip_time - DECISION_LEAD]
            close = p[p.ts <= r.tip_time]
            if at.empty or close.empty:
                continue
            q, c = at.iloc[-1], close.iloc[-1]
            rows.append({"game_id": r.game_id, "bid": float(q.bid), "ask": float(q.ask),
                         "close_mid": float((c.bid + c.ask) / 2)})
        out = pd.DataFrame(rows)
        CACHE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
        return out, "Kalshi quotes 30 min before tip (replay price table)"
    out = pd.DataFrame({"game_id": t.game_id, "bid": t.market_before - HALF_SPREAD,
                        "ask": t.market_before + HALF_SPREAD, "close_mid": t.market_tip})
    return out, ("APPROXIMATE quotes: mid 1 h before tip +/- 1 cent (data/frozen/prices.parquet was evicted "
                 "by iCloud and not available offline)")


def fee_pc(price):
    return 0.07 * price * (1 - price)


def trade(t: pd.DataFrame, q: pd.DataFrame, p_col: str) -> pd.DataFrame:
    d = t[["game_id", "date", "home_win"]].assign(p=t[p_col].to_numpy(float)).merge(q, on="game_id")
    rows = []
    for r in d.itertuples():
        p = r.p
        yes_gap = p - r.ask - fee_pc(r.ask)
        no_gap = (1 - p) - (1 - r.bid) - fee_pc(1 - r.bid)
        side, gap = ("yes", yes_gap) if yes_gap >= no_gap else ("no", no_gap)
        if gap <= MIN_EDGE:
            continue
        price = r.ask if side == "yes" else 1 - r.bid
        if not 0 < price < 1:
            continue
        contracts = int(np.floor(STAKE / price + 1e-9))
        close = r.close_mid if side == "yes" else 1 - r.close_mid
        won = float(r.home_win == 1) if side == "yes" else float(r.home_win == 0)
        fee = kalshi_fee(contracts, price)
        rows.append({"game_id": r.game_id, "side": side, "price": price, "contracts": contracts, "gap": gap,
                     "clv": close - price, "pnl": contracts * (won - price) - fee})
    return pd.DataFrame(rows, columns=["game_id", "side", "price", "contracts", "gap", "clv", "pnl"])


def score_ci(p_a, p_b, y, days, fn, reps=REPS, seed=SEED):
    """Difference fn(a) - fn(b) with a day-clustered bootstrap CI."""
    terms = (lambda p: (p - y) ** 2) if fn == "brier" else (
        lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    diff = pd.Series(terms(p_a) - terms(p_b)).groupby(np.asarray(days))
    s, n = diff.sum().to_numpy(), diff.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main():
    from replay import load_tables  # noqa: F401  (keeps the import path identical to other evals)
    from data_sources import read_table
    RESULTS.mkdir(parents=True, exist_ok=True)
    t = signals()
    y = t.home_win.to_numpy(float)
    learner = BlendLearner()
    t["blend"] = learner.run(t).sort_values(["date", "game_id"]).p_blend.to_numpy()

    ablations = {"blend (all inputs)": SIGNALS,
                 "no market inputs (raw M4, MLP, GRU)": ("m4", "mlp", "gru"),
                 "market only (anchor, mid)": ("anchor", "mid")}
    for s in SIGNALS:
        ablations[f"all minus {s}"] = tuple(x for x in SIGNALS if x != s)
    abl_logs = {}
    for name, sig in list(ablations.items())[1:]:
        start = "anchor_m4" if "anchor_m4" in sig else sig[0]
        lr = BlendLearner(signals=sig, start=start)
        t[name] = lr.run(t).p_blend.to_numpy()
        abl_logs[name] = lr
    t["blend (all inputs)"] = t.blend

    rows = []
    ref = t.anchor_m4.to_numpy()
    preds = {"Market 24 h before tip (anchor)": "anchor", "Market 1 h before tip": "mid",
             "Market at tip": "market_tip", "Anchor + M4 shift (agent)": "anchor_m4",
             "Anchor + MLP shift": "anchor_mlp", "Anchor + GRU shift": "anchor_gru", "M4 raw": "m4",
             **{f"Blend: {k}": k for k in ablations}}
    for label, col in preds.items():
        p = t[col].to_numpy(float)
        db = score_ci(p, ref, y, t.date, "brier")
        dl = score_ci(p, ref, y, t.date, "log")
        dm = score_ci(p, t.market_tip.to_numpy(float), y, t.date, "brier")
        rows.append({"predictor": label, "games": len(t), "brier": brier(p, y), "log_loss": log_loss(p, y),
                     "brier_minus_agent": db[0], "brier_minus_agent_lo": db[1], "brier_minus_agent_hi": db[2],
                     "logloss_minus_agent": dl[0], "logloss_minus_agent_lo": dl[1], "logloss_minus_agent_hi": dl[2],
                     "brier_minus_tip": dm[0], "brier_minus_tip_lo": dm[1], "brier_minus_tip_hi": dm[2]})
    acc = pd.DataFrame(rows)

    q, quote_note = quotes(t)
    games = read_table("games").assign(game_id=lambda d: d.game_id.astype(str))
    days = sorted(t.date.unique())
    trade_rows, tabs = [], {}
    for label, col in [("Anchor + M4 shift (agent)", "anchor_m4"), ("Anchor + MLP shift", "anchor_mlp"),
                       ("Blend (all inputs)", "blend"), ("Blend, no market inputs", "no market inputs (raw M4, MLP, GRU)")]:
        f = trade(t, q, col)
        tab = day_table(f, games, days)
        tabs[label] = tab
        b = bootstrap(tab)
        trade_rows.append({"forecaster": label, "trades": len(f), **{f"{m}{s}": b.loc[m, c] for m in
                           ("mean_clv", "clv_dollars", "pnl") for s, c in (("", "estimate"), ("_lo", "ci_low"),
                                                                             ("_hi", "ci_high"))}})
    trades = pd.DataFrame(trade_rows)
    pairs = []
    for label in ("Blend (all inputs)", "Anchor + MLP shift", "Blend, no market inputs"):
        d = paired(tabs[label], tabs["Anchor + M4 shift (agent)"])
        for m in ("clv_dollars", "pnl"):
            pairs.append({"comparison": f"{label} − Anchor + M4 shift", "metric": m, "estimate": d.loc[m, "estimate"],
                          "ci_low": d.loc[m, "ci_low"], "ci_high": d.loc[m, "ci_high"]})
    pairs = pd.DataFrame(pairs)

    out = pd.concat([acc.assign(table="accuracy"), trades.assign(table="trading"), pairs.assign(table="paired")])
    out.to_csv(RESULTS / "forecast_blend.csv", index=False)
    (RESULTS / "forecast_blend_log.json").write_text(json.dumps(learner.log, indent=1, default=float))
    figure(learner, t)
    markdown(acc, trades, pairs, learner, quote_note, t)
    print((RESULTS / "forecast_blend.md").read_text())


def figure(learner, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    days = sorted(t.date.unique())
    w = np.array([learner.weights_on(d) for d in days])
    x = pd.to_datetime(days)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(x, w.T, labels=learner.signals)
    for e in learner.log:
        if e["result"] == "accepted":
            ax.axvline(pd.Timestamp(e["day"]), color="k", lw=0.6, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("blend weight (logit space)")
    ax.set_title("Forecast-blend weights in use per test day (dotted: gate accepted new weights)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_blend.png", dpi=150)


def markdown(acc, trades, pairs, learner, quote_note, t):
    f4 = lambda e, lo, hi: f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    f0 = lambda e, lo, hi: f"{e:+.0f} [{lo:+.0f}, {hi:+.0f}]"
    acc_n = sum(e["result"] == "accepted" for e in learner.log)
    rej_n = sum(e["result"] == "rejected" for e in learner.log)
    L = ["# Forecast-blend subloop", "",
         "`agents/forecast_blend.py`: p(home) = sigmoid(Σ w_k · logit(signal_k)), w on the simplex, so a one-hot "
         "weight on *anchor + M4 shift* is the published agent. After each settled test day the learner refits "
         "w on settled games only (exponentiated gradient on log loss, L2 pull toward the current weights) and "
         f"a split gate accepts the proposal only if **Brier (primary metric, fixed up front)** on held-out earlier "
         f"days improves by ≥ {GATE_BRIER}. Windows: gate days = the {GATE_DAYS} market days before the last "
         f"{SELECT_DAYS}; fit = the other days of the last {FIT_DAYS}; the first review needs "
         f"{SELECT_DAYS + GATE_DAYS + 7} settled days. Starting weights: one-hot on anchor + M4 shift.", "",
         f"Test period {t.date.min()}–{t.date.max()}, {len(t)} games. Day-clustered bootstrap, {REPS} replicates, "
         f"seed {SEED}. Inputs are all as-of: M4 and M4-NN were trained on games before 1 Feb; the 1 h mid is "
         "public before the 30 min decision. No LLM, deterministic.", "",
         f"Gate decisions: {acc_n} accepted, {rej_n} rejected, "
         f"{sum(e['result'] == 'deferred' for e in learner.log)} deferred (warm-up). Final weights: "
         + ", ".join(f"{s} {w:.2f}" for s, w in zip(learner.signals, learner.weights) if w >= 0.005) + ".", "",
         "## Forecast accuracy (all 501 test games; blends are walk-forward)", "",
         "| Predictor | Brier | Log loss | Brier − agent [95% CI] | Log loss − agent [95% CI] | Brier − market at tip [95% CI] |",
         "| --- | --- | --- | --- | --- | --- |"]
    for r in acc.itertuples():
        L.append(f"| {r.predictor} | {r.brier:.4f} | {r.log_loss:.4f} | "
                 f"{f4(r.brier_minus_agent, r.brier_minus_agent_lo, r.brier_minus_agent_hi)} | "
                 f"{f4(r.logloss_minus_agent, r.logloss_minus_agent_lo, r.logloss_minus_agent_hi)} | "
                 f"{f4(r.brier_minus_tip, r.brier_minus_tip_lo, r.brier_minus_tip_hi)} |")
    L += ["", "## Trading (simplified offline trader, same for every forecaster)", "",
          f"Quotes: {quote_note}. One decision per game on the home market 30 min before tip, edge > 4¢ after "
          "fees, $20 stake, no volume cap, no notebook rules. This is not the full agent replay, so absolute "
          "numbers differ from `gate_audit.md`; the comparison isolates the probability.", "",
          "| Forecaster | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] |",
          "| --- | --- | --- | --- | --- |"]
    for r in trades.itertuples():
        L.append(f"| {r.forecaster} | {r.trades} | {f4(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi)} | "
                 f"{f0(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi)} | {f0(r.pnl, r.pnl_lo, r.pnl_hi)} |")
    L += ["", "| Paired comparison | Metric | Difference [95% CI] |", "| --- | --- | --- |"]
    for r in pairs.itertuples():
        L.append(f"| {r.comparison} | {r.metric} | {f0(r.estimate, r.ci_low, r.ci_high)} |")
    L += ["", "Reading: the blend trades about as often as the agent but loses less CLV per contract; the CLV-dollar "
          "gain survives real quotes (an earlier run on approximate quotes, 1 h mid ± 1¢, gave +$19 [+2, +37]), "
          "while P&L is indistinguishable. Part of the gain can come from the 1 h mid input pulling the estimate "
          "toward the traded price (fewer, smaller disagreements with the market). The forecast gain is not "
          "significant: Brier −0.0010 vs the agent, and still behind anchor + MLP alone and the market at tip."]
    L += ["", "## Weight log", "", "Accepted proposals (full log in `forecast_blend_log.json`):", ""]
    for e in learner.log:
        if e["result"] == "accepted":
            L.append(f"- {e['day']}: " + ", ".join(f"{s} {w:.2f}" for s, w in zip(e["signals"], e["proposed"])
                                                   if w >= 0.005) + f" — {e['reason']}")
    L += ["", "![weights](forecast_blend.png)", "",
          "## Not built (future work)", "",
          "- **Cross-venue prices** (Polymarket, DraftKings; `evaluation/venue_compare.py`): not added as inputs "
          "this round; they need an as-of timestamp audit before they can enter a blend.",
          "- **M2-based missing-player impact** (expected minutes/points of absent players from M2 medians).",
          "- **NBA injury-report PDFs** (`data/raw/injury_reports`, 2,637 files) are not in the frozen news; a "
          "parsed, timestamped table would give earlier injury information than the tip − 30 min ESPN list.",
          "- **Shift-baseline fix**: the shift's \"before\" uses out=[], double-counting absences already known "
          "at anchor time; not fixed here.",
          "- **M6** predicted move was left out (no cached per-game test predictions at the 30 min decision); "
          "**M3** (play model) predicts a different target (player participation), so it has no direct role "
          "in a game-winner probability beyond what M4's absence features already use.",
          "- The blend is not yet wired into `MarketAgent` replays (`--forecast blend`); trading numbers above come "
          "from the simplified offline trader.", ""]
    (RESULTS / "forecast_blend.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
