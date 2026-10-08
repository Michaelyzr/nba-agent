"""Part 2 of the learning upgrades: a model-review subloop that learns which forecast blend to trade on.

    python -m evaluation.model_review                   # 3 arms on the test period (3 processes)
    python -m evaluation.model_review --report-only

Arms (split gate, flat $20, empty notebook on 1 Feb, all log the same forecast components):

  m4            the current agent: anchor + M4 news shift, split-gate rule learning
  review        + model review (settle -> review -> gate -> ... -> review_model -> gate_model -> save/reject)
  review_adapt  + model review + adaptive edge (part 1, base 4¢)

Pre-stated before any test result was seen (code in forecast/blend.py and agents/graph.py):
  * components (as-of; models trained before 1 Feb): anchor (home mid 24 h before tip), current home mid,
    M4 news shift, M4-NN MLP news shift (3 seeds), M6 ImpactNet median move to tip.
  * candidate blends: anchor + 1.0·M4 (default), anchor only, anchor + 0.5·M4, anchor + MLP,
    anchor + mean(M4, MLP), mid + M6, and a logistic stack (L2, C = 0.1) of
    [logit anchor, logit mid − logit anchor, M4, MLP, M6] fit on the selection days.
  * every 7 market days, once 21 market days from 1 Feb exist: the reviewer picks the lowest-Brier
    blend on the last 7 market days' decision points; the gate back-tests the notebook with vs
    without it on the 14 market days before and accepts only if CLV $ rises by >= $2 over >= 3
    changed trades AND the Brier of the blend on the gate days' decision points is not worse than
    the forecasts the agent actually used there.
  * metrics: forecasts actually used (Brier, log loss, accuracy) vs the market mid at the decision
    and at tip, per decision point and per game (last decision); trading CLV $ and P&L with
    day-clustered bootstrap CIs (2000 reps, seed 7606) and paired differences vs m4.

Writes evaluation/results/model_review.{md,csv,png} and model_review_loop.png.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.adaptive_edge import TEST, fmt, load_frozen
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import market_days
from replay import RUNS

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RUNS / "model_review"
# arm -> (model_review, adaptive_edge)
ARMS = {"m4": (False, False), "review": (True, False), "review_adapt": (True, True)}
LABELS = {"m4": "Fixed M4 shift + rule learning (current agent)",
          "review": "Model review subloop + rule learning",
          "review_adapt": "Model review + adaptive edge (parts 1+2)"}


def run_arm(arm: str) -> dict:
    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from forecast.api import Forecaster
    from forecast.blend import Components, load_mlp
    from forecast.impact import load_impact
    from replay import Replay

    t0 = time.time()
    tables, forecaster = load_frozen(), Forecaster.load()
    review, adaptive = ARMS[arm]
    comps = Components(forecaster, load_mlp(train_if_missing=False), load_impact())
    agent = MarketAgent(notebook=Notebook(), forecaster=forecaster, learn=True, gate="split",
                        adaptive_edge=adaptive, model_review=review, components=comps)
    rp = Replay(tables, agent.policy, on_day_end=agent.on_day_end)
    decisions, fills = rp.run(*TEST)
    folder = save_run(f"model_review/{arm}", decisions, fills, agent)
    pd.DataFrame(list(agent.component_log.values())).to_parquet(folder / "components.parquet", index=False)
    return {"arm": arm, "fills": len(fills), "minutes": (time.time() - t0) / 60,
            "rules": [(r["rule_id"], r["status"], r["do"]["action"]) for r in agent.notebook.rules]}


def run_all(workers: int, arms: list):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_arm, arms):
            print(f"  {r['arm']}: {r['fills']} fills in {r['minutes']:.1f} min; rules {r['rules']}", flush=True)


# ---------------- report ----------------

def close_mids(comp: pd.DataFrame, tables: dict) -> pd.Series:
    """Home mid at tip-off per game (the closing line)."""
    from replay import AsOf
    prices = tables["prices"].sort_values("ts")
    markets, games = tables["markets"], tables["games"].set_index("game_id")
    out = {}
    for gid in comp.game_id.unique():
        g = games.loc[gid]
        m = markets[(markets.game_id == gid) & (markets.kind == "game") & (markets.team == g.home_team)]
        if m.empty:
            continue
        p = prices[prices.market_ticker == m.market_ticker.iloc[0]]
        p = p[p.ts <= g.tip_time]
        out[gid] = float((p.bid.iloc[-1] + p.ask.iloc[-1]) / 2) if len(p) else np.nan
    return comp.game_id.map(out)


def forecast_scores(comp: pd.DataFrame, arm: str) -> list:
    from evaluation.win_nn_eval import brier_terms, diff_ci, log_terms
    rows = []
    for unit, frame in (("decision points", comp), ("games (last decision)", comp.sort_values("as_of")
                                                   .groupby("game_id").tail(1))):
        frame = frame.dropna(subset=["p_used", "mid", "close"])
        y, days = frame.home_win.to_numpy(), frame.date.to_numpy()
        for name, p in (("forecast used", frame.p_used), ("market at decision", frame["mid"]),
                        ("market at tip", frame.close)):
            p = p.to_numpy(float)
            r = {"arm": arm, "unit": unit, "predictor": name, "n": len(y), "brier": brier_terms(p, y).mean(),
                 "log_loss": log_terms(p, y).mean(), "accuracy": float(np.mean((p > 0.5) == y))}
            if name == "forecast used":
                for ref, q in (("decision", frame["mid"]), ("tip", frame.close)):
                    e, lo, hi = diff_ci(brier_terms(p, y), brier_terms(q.to_numpy(float), y), days)
                    r.update({f"brier_minus_mkt_{ref}": e, f"brier_minus_mkt_{ref}_lo": lo,
                              f"brier_minus_mkt_{ref}_hi": hi})
            rows.append(r)
    return rows


def posthoc_blends(comp: pd.DataFrame) -> pd.DataFrame:
    """Each fixed candidate blend over the whole test period (descriptive only; the agent never sees this)."""
    from evaluation.win_nn_eval import brier_terms, diff_ci
    from forecast.blend import CANDIDATE_BLENDS, blend_home
    y, days = comp.home_win.to_numpy(), comp.date.to_numpy()
    mkt = brier_terms(comp["mid"].to_numpy(float), y)
    rows = []
    for b in CANDIDATE_BLENDS:
        if b["name"] == "stack":
            continue
        terms = brier_terms(blend_home(b, comp), y)
        e, lo, hi = diff_ci(terms, mkt, days)
        rows.append({"blend": b["name"], "brier": terms.mean(), "minus_mkt": e, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def blend_rules(nb: dict) -> list:
    out = []
    for r in nb.get("rules", []):
        if r["do"]["action"] != "forecast_blend":
            continue
        g = r.get("gate", {})
        sel = r.get("selection", {})
        out.append({"rule": r["rule_id"], "blend": r["do"]["params"]["name"], "status": r["status"],
                    "decided": str(g.get("decided_at", ""))[:10], "valid_until": str(r.get("valid_until") or "")[:10],
                    "sel_days": sel.get("days"), "sel_rows": sel.get("rows"), "sel_brier": sel.get("brier", {}),
                    "gate_days": g.get("backtest_days"), "clv_before": g.get("clv_dollars_before"),
                    "clv_after": g.get("clv_dollars_after"), "brier_before": g.get("brier_before"),
                    "brier_after": g.get("brier_after"), "cases": g.get("cases"),
                    "coef": r["do"]["params"].get("coef")})
    return out


def blend_timeline(nb: dict, days: list) -> pd.DataFrame:
    from agents.graph import _recency
    from agents.notebook import Notebook
    book = Notebook(nb.get("rules", []))
    rows = []
    for d in days:
        now = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)
        live = [r for r in book.active(now, "forecast") if r["do"]["action"] == "forecast_blend"]
        rows.append({"date": d, "blend": max(live, key=_recency)["do"]["params"]["name"] if live else "m4"})
    return pd.DataFrame(rows)


def loop_diagram(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = {"settle": (0.3, 1.5), "review": (2.0, 2.5), "gate": (3.7, 2.5), "save_rule / reject_rule": (5.6, 2.5),
             "review_model": (2.0, 0.6), "gate_model": (3.9, 0.6), "save_model / reject_model": (6.0, 0.6),
             "notebook": (9.0, 1.5)}
    colors = {"review_model": "#d3f9d8", "gate_model": "#d3f9d8", "save_model / reject_model": "#d3f9d8",
              "notebook": "#fff3bf"}
    for name, (x, y) in boxes.items():
        w = 1.75 if "/" in name else 1.4
        ax.add_patch(FancyBboxPatch((x, y), w, 0.6, boxstyle="round,pad=0.05", fc=colors.get(name, "#e7f5ff"),
                                    ec="#333"))
        ax.text(x + w / 2, y + 0.3, name, ha="center", va="center", fontsize=9)

    def arrow(a, b, text=""):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=12, color="#333"))
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, text, fontsize=7, ha="center")
    arrow((1.7, 1.9), (2.0, 2.7))
    arrow((3.4, 2.8), (3.7, 2.8), "rule")
    arrow((5.1, 2.8), (5.6, 2.8))
    arrow((6.5, 2.5), (2.7, 1.2), "then (sequential)")
    arrow((3.4, 0.9), (3.9, 0.9), "blend")
    arrow((5.3, 0.9), (6.0, 0.9))
    arrow((7.35, 2.8), (9.0, 2.0), "trader rules")
    arrow((7.75, 0.9), (9.0, 1.6), "forecast rules")
    ax.text(0.3, 0.1, "review_model: every 7 market days, lowest-Brier blend on the last 7 days (stack fit there); "
            "gate_model: back-test on the 14 days before, CLV $ +$2 and Brier not worse. "
            "decide → forecast uses the newest active forecast rule.", fontsize=7.5)
    ax.set_title("Learn phase with the model-review subloop (--model-review)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart(tabs, nbs, days, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"m4": "#1f77b4", "review": "#2ca02c", "review_adapt": "#d62728"}
    names = ["m4", "market", "m4_half", "mlp", "m4_mlp_mean", "m6", "stack"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    x = pd.to_datetime(days)
    for i, arm in enumerate(ARMS):
        if arm not in nbs:
            continue
        tl = blend_timeline(nbs[arm], days)
        ax1.step(x, [names.index(b) + (i - 1) * 0.08 for b in tl.blend], where="post", color=colors[arm],
                 label=LABELS[arm], lw=2)
        ax2.plot(x, tabs[arm].clv_dollars.cumsum().to_numpy(), color=colors[arm], label=LABELS[arm], lw=2)
    ax1.set_yticks(range(len(names)), names)
    ax1.set_title("Active forecast blend (accepted forecast rules)")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("cumulative CLV $")
    ax2.set_title("Cumulative closing-line value, test period")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report():
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *TEST)
    rows, tabs, nbs, fc_rows = [], {}, {}, []
    for arm in ARMS:
        folder = OUT / arm
        if not (folder / "fills.parquet").exists():
            continue
        fills = pd.read_parquet(folder / "fills.parquet")
        nb = json.loads((folder / "notebook.json").read_text())
        table = day_table(fills, games, days)
        boot = bootstrap(table)
        rules = nb.get("rules", [])
        row = {"arm": arm, "setup": LABELS[arm], "trades": int(table.trades.sum()),
               "trader_rules": f"{sum(r['status'] == 'active' and r['kind'] == 'trader' for r in rules)}/"
                               f"{sum(r['kind'] == 'trader' for r in rules)}",
               "forecast_rules": f"{sum(r['status'] == 'active' and r['kind'] == 'forecast' for r in rules)}/"
                                 f"{sum(r['kind'] == 'forecast' for r in rules)}"}
        for m in ("mean_clv", "clv_dollars", "pnl"):
            row.update({m: boot.loc[m, "estimate"], f"{m}_lo": boot.loc[m, "ci_low"], f"{m}_hi": boot.loc[m, "ci_high"]})
        rows.append(row)
        tabs[arm], nbs[arm] = table, nb
        comp = pd.read_parquet(folder / "components.parquet")
        comp = comp[comp.date.isin(days)]
        g = games.set_index("game_id")
        comp["home_win"] = (g.loc[comp.game_id, "home_pts"].to_numpy() > g.loc[comp.game_id, "away_pts"].to_numpy()
                            ).astype(float)
        comp["close"] = close_mids(comp, tables)
        fc_rows += forecast_scores(comp, arm)
        if arm == "m4":
            post = posthoc_blends(comp)
    frame, fc = pd.DataFrame(rows), pd.DataFrame(fc_rows)
    pairs = []
    for a, b in (("review", "m4"), ("review_adapt", "m4"), ("review_adapt", "review")):
        if a in tabs and b in tabs:
            d = paired(tabs[a], tabs[b])
            for m in ("clv_dollars", "pnl"):
                pairs.append({"comparison": f"{LABELS[a]} − {LABELS[b]}", "metric": m, "estimate": d.loc[m, "estimate"],
                              "ci_low": d.loc[m, "ci_low"], "ci_high": d.loc[m, "ci_high"], "p": d.loc[m, "p_a_gt_b"]})
    pairs = pd.DataFrame(pairs)
    frame.to_csv(RESULTS / "model_review.csv", index=False)
    fc.to_csv(RESULTS / "model_review_forecasts.csv", index=False)
    pairs.to_csv(RESULTS / "model_review_pairs.csv", index=False)
    chart(tabs, nbs, days, RESULTS / "model_review.png")
    loop_diagram(RESULTS / "model_review_loop.png")

    lines = ["# Part 2: a model-review subloop that learns how to combine the models", "",
             "**Question.** The agent trades on one forecast: the 24 h market anchor plus M4's news shift. Can a "
             "second learning subloop, reusing the notebook and the split gate, learn to use the other models "
             "(M4-NN MLP, M6 ImpactNet) and the market itself better?", "",
             "![model-review loop](model_review_loop.png)", "",
             "**Design (pre-stated, fixed in code before the test period was run).**",
             "- Components (as-of; every model trained before 1 Feb): anchor (home mid 24 h before tip), current home "
             "mid, M4 news shift, M4-NN MLP news shift (3 seeds, the validated config from `win_nn_eval`), M6 "
             "ImpactNet median move to tip. M1/M2 player projections are not added: M4's missing-minutes/points "
             "inputs already carry absent players' usual minutes and points. The official injury-report PDFs are "
             "future work (no as-of-safe parse in the time box).",
             "- Candidate blends: anchor + 1.0·M4 shift (default), anchor only, anchor + 0.5·M4, anchor + MLP, "
             "anchor + mean(M4, MLP), current mid + M6 move, and a logistic stack (L2, C = 0.1) of [logit anchor, "
             "logit mid − logit anchor, M4, MLP, M6] fit on the selection days only.",
             "- Review: every 7 market days once 21 market days from 1 Feb exist (so no window contains a game any "
             "model trained on), pick the lowest-Brier blend on the last 7 market days' decision points.",
             "- Gate (pre-stated criterion): back-test the notebook with vs without the blend on the 14 market days "
             "before; accept only if CLV $ rises by ≥ $2 over ≥ 3 changed trades **and** the blend's Brier on those "
             "days' decision points is not worse than the forecasts actually used. Accepted blends are notebook rules "
             "of kind `forecast` (`forecast_blend`), valid from the decision time; a new one supersedes the old.",
             f"- Test period {TEST[0]}–{TEST[1]}; day-clustered bootstrap ({REPS} reps, seed {SEED}).", "",
             "## Trading results", "",
             "| Setup | Trader rules accepted/proposed | Forecast rules accepted/proposed | Trades | Mean CLV/contract "
             "[95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in frame.itertuples():
        lines.append(f"| {r.setup} | {r.trader_rules} | {r.forecast_rules} | {r.trades} | "
                     f"{fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
                     f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi)} | {fmt(r.pnl, r.pnl_lo, r.pnl_hi)} |")
    lines += ["", "| Paired difference | Metric | Estimate [95% CI] | p (one-sided, a > b) |", "| --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        lines.append(f"| {r.comparison} | {r.metric} | {fmt(r.estimate, r.ci_low, r.ci_high)} | {r.p:.3f} |")
    lines += ["", "## Forecasts actually used vs the market", "",
              "Brier difference CIs resample game-days. Lower is better; 'market at tip' is the closing line.", "",
              "| Setup | Unit | Predictor | n | Brier | Log loss | Accuracy | Brier − market at decision [95% CI] | "
              "Brier − market at tip [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in fc.itertuples():
        d1 = (f"{r.brier_minus_mkt_decision:+.4f} [{r.brier_minus_mkt_decision_lo:+.4f}, "
              f"{r.brier_minus_mkt_decision_hi:+.4f}]") if r.predictor == "forecast used" else ""
        d2 = (f"{r.brier_minus_mkt_tip:+.4f} [{r.brier_minus_mkt_tip_lo:+.4f}, {r.brier_minus_mkt_tip_hi:+.4f}]"
              if r.predictor == "forecast used" else "")
        lines.append(f"| {LABELS[r.arm]} | {r.unit} | {r.predictor} | {r.n} | {r.brier:.4f} | {r.log_loss:.4f} | "
                     f"{r.accuracy:.3f} | {d1} | {d2} |")
    lines += ["", "## Which blend was active when", "", "![blend timeline and cumulative CLV](model_review.png)", ""]
    for arm in ("review", "review_adapt"):
        if arm not in nbs:
            continue
        tl = blend_timeline(nbs[arm], days)
        changes = tl[tl.blend != tl.blend.shift()]
        lines += [f"**{LABELS[arm]}** — active blend: " + " → ".join(f"{b} from {d}" for d, b in
                                                                       zip(changes.date, changes.blend)) + ".", ""]
        rules = blend_rules(nbs[arm])
        if rules:
            lines += ["| Rule | Blend | Status | Decided | Superseded | Selection days (rows) | Selection Brier (best 3) | "
                      "Gate days | CLV $ without → with | Brier used → blend (gate days) |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in rules:
                best3 = ", ".join(f"{k} {v:.4f}" for k, v in sorted(r["sel_brier"].items(), key=lambda kv: kv[1])[:3])
                lines.append(
                    f"| {r['rule']} | {r['blend']} | {r['status']} | {r['decided']} | {r['valid_until'] or '–'} | "
                    f"{r['sel_days'][0]} – {r['sel_days'][1]} ({r['sel_rows']}) | {best3} | "
                    f"{r['gate_days'][0]} – {r['gate_days'][1]} | {r['clv_before']:+.2f} → {r['clv_after']:+.2f} | "
                    f"{r['brier_before']:.4f} → {r['brier_after']:.4f} |")
            lines.append("")
    reviews = []
    for arm in ("review", "review_adapt"):
        trace = OUT / arm / "trace.jsonl"
        if trace.exists():
            for line in trace.read_text().splitlines():
                t = json.loads(line)
                if t.get("phase") == "review":
                    for s in t["trace"]:
                        if s["step"] == "review_model" and s["detail"] != "not due":
                            reviews.append(f"- {arm} {t['day']}: {s['detail']}")
    if reviews:
        lines += ["Every model review (selection-day Brier per candidate):", ""] + reviews + [""]
    lines += ["## Reading", "", "VERDICT_PLACEHOLDER", ""]
    path = RESULTS / "model_review.md"
    old = path.read_text() if path.exists() else ""
    verdict = old.split("## Reading\n\n", 1)[1].rstrip("\n") if "## Reading\n\n" in old else ""
    text = "\n".join(lines) + "\n"
    if verdict and verdict != "VERDICT_PLACEHOLDER":
        text = text.replace("VERDICT_PLACEHOLDER", verdict)
    path.write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_all(args.workers, args.arms)
    report()


if __name__ == "__main__":
    main()
