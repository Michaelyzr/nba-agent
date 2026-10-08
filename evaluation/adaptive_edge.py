"""Part 1 of the learning upgrades: does the agent learn its own edge threshold?

    python -m evaluation.adaptive_edge                  # 4 arms on the test period (~30 min, 4 processes)
    python -m evaluation.adaptive_edge --report-only    # re-score saved runs
    python -m evaluation.adaptive_edge --subsample      # also the 304-point subsample (seed 7606)

Arms (all M4 anchor + news shift, flat $20 stakes, split gate, empty notebook on 1 Feb):

  fixed4    fixed 4-point edge, no learning                        (gate_audit "No learning")
  split     published split-gate learning (REVIEW_TEMPLATES)       (gate_audit "Split gate + learning")
  adapt4    adaptive edge from a 4-point base (EDGE_TEMPLATES)
  adapt1    adaptive edge from a 1-point base (fees are already in the gap, so ~fee-only)

Pre-stated before any test result was seen (the design is fixed in agents/graph.py):
  * candidate thresholds: market-wide 2, 3, 5, 7, 10 points; underdogs (<= 35%) 8 points; long shots
    (<= 25%) 10 points; stale news (>= 30 min) 7 points; plus the published skip templates.
  * the reviewer proposes the candidate with the largest CLV-dollar gain on the last 7 market days
    (counterfactual CLV for decisions a lower threshold would let through); the split gate accepts it
    only if CLV dollars on the 14 market days before improve by >= $2 over >= 3 changed trades.
  * primary metric: CLV dollars over the test period, day-clustered bootstrap (2000 reps, seed 7606);
    paired differences vs fixed4 and vs split. Nothing is tuned on the test period.

Writes evaluation/results/adaptive_edge.{md,csv,png}; runs go to runs/adaptive_edge/.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import REGULAR_END, market_days
from replay import RUNS

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RUNS / "adaptive_edge"
TEST = ("2026-02-01", REGULAR_END)
FALLBACK_FROZEN = Path("/tmp/nba_frozen")       # local copy when iCloud evicts data/frozen/prices.parquet
SUB_FRAC, SUB_SEED = 0.4, 7606                  # the llm_agent_eval / edge_gate subsample
# arm -> (learn, adaptive_edge, base edge)
ARMS = {
    "fixed4": (False, False, 0.04),
    "split": (True, False, 0.04),
    "adapt4": (True, True, 0.04),
    "adapt1": (True, True, 0.01),
}
LABELS = {
    "fixed4": "(i) Fixed 4¢ edge, no learning",
    "split": "(ii) Split-gate learning (published templates)",
    "adapt4": "(iii) Adaptive edge, base 4¢",
    "adapt1": "(iv) Adaptive edge, base 1¢",
}


def load_frozen() -> dict:
    from data_sources import FROZEN
    from replay import load_tables
    use_fallback = not (FROZEN / "prices.parquet").exists() and (FALLBACK_FROZEN / "prices.parquet").exists()
    return load_tables(FALLBACK_FROZEN if use_fallback else FROZEN)


def sampled(policy, frac: float, seed: int):
    from evaluation.llm_agent_eval import sampled as keep
    return keep(policy, frac, seed)


def run_arm(job: dict) -> dict:
    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from forecast.api import Forecaster
    from replay import Replay

    t0 = time.time()
    tables, forecaster = load_frozen(), Forecaster.load()
    learn, adaptive, base = ARMS[job["arm"]]
    agent = MarketAgent(notebook=Notebook(), forecaster=forecaster, learn=learn, gate="split",
                        min_edge=base, adaptive_edge=adaptive)
    policy = sampled(agent.policy, SUB_FRAC, SUB_SEED) if job["sub"] else agent.policy
    rp = Replay(tables, policy, on_day_end=agent.on_day_end if learn else None)
    decisions, fills = rp.run(*TEST)
    folder = save_run(f"adaptive_edge/{job['name']}", decisions, fills, agent)
    return {"name": job["name"], "fills": len(fills), "minutes": (time.time() - t0) / 60,
            "active": sum(r["status"] == "active" for r in agent.notebook.rules), "rules": len(agent.notebook.rules),
            "folder": str(folder)}


def run_all(workers: int, sub: bool):
    jobs = [{"arm": a, "name": a, "sub": False} for a in ARMS]
    if sub:
        jobs += [{"arm": a, "name": f"sub-{a}", "sub": True} for a in ARMS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_arm, jobs):
            print(f"  {r['name']}: {r['fills']} fills, {r['active']}/{r['rules']} rules active, "
                  f"{r['minutes']:.1f} min", flush=True)


# ---------------- report ----------------

def threshold_timeline(notebook: dict, base: float, adaptive: bool, days: list) -> pd.DataFrame:
    """Market-wide edge in force at the start of each market day, from the saved notebook."""
    from agents.graph import effective_edge, is_global_edge
    from agents.notebook import Notebook
    nb = Notebook(notebook.get("rules", []))
    rows = []
    for d in days:
        now = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)       # before any tip that day
        glob = [r for r in nb.active(now, "trader") if is_global_edge(r)]
        rows.append({"date": d, "edge": effective_edge(base, glob, adaptive),
                     "slice_rules": sum(r["do"]["action"] == "min_edge" and not is_global_edge(r)
                                        for r in nb.active(now, "trader"))})
    return pd.DataFrame(rows)


def edge_rules(notebook: dict) -> list:
    out = []
    for r in notebook.get("rules", []):
        if r["do"]["action"] != "min_edge" or r["status"] not in ("active", "rejected"):
            continue
        g = r.get("gate", {})
        out.append({"rule": r["rule_id"], "status": r["status"], "when": {k: v for k, v in r["when"].items()
                                                                          if k != "market_kind"},
                    "edge": r["do"]["params"]["edge"], "decided": str(g.get("decided_at", ""))[:10],
                    "valid_until": str(r.get("valid_until") or "")[:10],
                    "clv_before": g.get("clv_dollars_before"), "clv_after": g.get("clv_dollars_after"),
                    "cases": g.get("cases"), "days": g.get("backtest_days")})
    return out


def fmt(est, lo, hi, kind="$"):
    if pd.isna(est):
        return "–"
    if kind == "clv":
        return f"{est * 100:+.2f}¢ [{lo * 100:+.2f}, {hi * 100:+.2f}]"
    return f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"


def score(prefix: str, tables: dict) -> tuple:
    games = tables["games"]
    days = market_days(tables, *TEST)
    rows, tabs, nbs, fills_by = [], {}, {}, {}
    for arm in ARMS:
        folder = OUT / f"{prefix}{arm}"
        if not (folder / "fills.parquet").exists():
            continue
        fills = pd.read_parquet(folder / "fills.parquet")
        nb = json.loads((folder / "notebook.json").read_text())
        table = day_table(fills, games, days)
        boot = bootstrap(table)
        rules = nb.get("rules", [])
        row = {"sample": "subsample" if prefix else "full", "arm": arm, "setup": LABELS[arm],
               "base_edge": ARMS[arm][2], "trades": int(table.trades.sum()),
               "rules_proposed": len(rules), "rules_accepted": sum(r["status"] == "active" for r in rules),
               "edge_rules_accepted": sum(r["status"] == "active" and r["do"]["action"] == "min_edge" for r in rules)}
        for m in ("mean_clv", "clv_dollars", "pnl", "roi"):
            row.update({m: boot.loc[m, "estimate"], f"{m}_lo": boot.loc[m, "ci_low"],
                        f"{m}_hi": boot.loc[m, "ci_high"]})
        rows.append(row)
        tabs[arm], nbs[arm], fills_by[arm] = table, nb, fills
    pairs = []
    for a in ARMS:
        for b in ("fixed4", "split"):
            if a == b or a not in tabs or b not in tabs or (b == "split" and a == "fixed4"):
                continue
            d = paired(tabs[a], tabs[b])
            for m in ("clv_dollars", "pnl", "mean_clv"):
                pairs.append({"sample": "subsample" if prefix else "full", "comparison": f"{a} − {b}",
                              "metric": m, "estimate": d.loc[m, "estimate"], "ci_low": d.loc[m, "ci_low"],
                              "ci_high": d.loc[m, "ci_high"], "p_a_gt_b": d.loc[m, "p_a_gt_b"]})
    return pd.DataFrame(rows), pd.DataFrame(pairs), tabs, nbs, days


def chart(tabs: dict, nbs: dict, days: list, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fixed4": "#555555", "split": "#1f77b4", "adapt4": "#d62728", "adapt1": "#2ca02c"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    x = pd.to_datetime(days)
    for arm, (learn, adaptive, base) in ARMS.items():
        if arm not in nbs:
            continue
        tl = threshold_timeline(nbs[arm], base, adaptive, days)
        ax1.step(x, tl.edge * 100, where="post", color=colors[arm], label=LABELS[arm], lw=2,
                 ls="--" if arm in ("fixed4", "split") else "-")
        ax2.plot(x, tabs[arm].clv_dollars.cumsum().to_numpy(), color=colors[arm], label=LABELS[arm], lw=2)
    ax1.set_ylabel("market-wide edge\nthreshold (¢)")
    ax1.set_title("Effective market-wide edge threshold (accepted min_edge rules; slice rules not shown)")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("cumulative CLV $")
    ax2.set_title("Cumulative closing-line value, test period 1 Feb – 12 Apr 2026")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report():
    tables = load_frozen()
    frame, pairs, tabs, nbs, days = score("", tables)
    sub_frame, sub_pairs, _, sub_nbs, _ = score("sub-", tables)
    frame = pd.concat([frame, sub_frame], ignore_index=True)
    pairs = pd.concat([pairs, sub_pairs], ignore_index=True)
    frame.to_csv(RESULTS / "adaptive_edge.csv", index=False)
    pairs.to_csv(RESULTS / "adaptive_edge_pairs.csv", index=False)
    chart(tabs, nbs, days, RESULTS / "adaptive_edge.png")

    lines = [
        "# Part 1: the agent learns its own edge threshold", "",
        "**Question.** The no-gate finding (`edge_gate.md`) was that CLV per contract is flat across gap sizes, so "
        "the 4¢ threshold mainly sets volume. Can the agent's own learning loop (reviewer → split gate → notebook) "
        "find a better threshold than a fixed 4¢?", "",
        "**Design (pre-stated, fixed in code before running the test period).**",
        "- New review templates (`EDGE_TEMPLATES` in `agents/graph.py`, on only with `adaptive_edge=True` / "
        "`--edge-templates`): market-wide thresholds 2, 3, 5, 7, 10¢; slice thresholds 8¢ on underdogs "
        "(price ≤ 35%), 10¢ on long shots (≤ 25%), 7¢ when the news is stale (≥ 30 min). The published skip "
        "templates stay; the published single 7¢ template is replaced by the market-wide set.",
        "- Semantics: the newest accepted market-wide `min_edge` rule *sets* the threshold (it can raise or "
        "lower the base); it supersedes the previous one through `valid_until` (history kept). Slice rules can "
        "only raise it. Default agent: unchanged (max of base and every rule).",
        "- Reviewer: largest CLV-dollar gain on the last 7 market days. Raising/skip: the CLV $ of the trades "
        "removed. Lowering: the counterfactual CLV $ (mid at tip − price paid) of the first decision per "
        "untraded game that the current threshold blocked and the new one would let through.",
        "- Gate (unchanged split gate): back-test the notebook with vs without the rule on the 14 market days "
        "before those 7; accept if CLV $ improves by ≥ $2 over ≥ 3 changed trades.",
        f"- Metric: CLV $ and P&L after fees on {TEST[0]}–{TEST[1]}, day-clustered bootstrap ({REPS} reps, "
        f"seed {SEED}). Empty notebook on 1 Feb; gate back-tests may use January days (M4 is trained "
        "before 1 Feb, so those days are in-sample for M4 — same as the published gate audit).", "",
        "## Results (full test period)", "",
        "| Setup | Rules proposed / accepted (edge) | Trades | Mean CLV/contract [95% CI] | CLV $ [95% CI] | "
        "P&L after fees [95% CI] |", "| --- | --- | --- | --- | --- | --- |"]
    for r in frame.itertuples():
        if r.sample != "full":
            continue
        lines.append(f"| {r.setup} | {r.rules_proposed} / {r.rules_accepted} ({r.edge_rules_accepted}) | "
                     f"{r.trades} | {fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
                     f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi)} | {fmt(r.pnl, r.pnl_lo, r.pnl_hi)} |")
    lines += ["", "## Paired differences (same game-days)", "",
              "| Sample | Comparison | Metric | Difference [95% CI] | p (one-sided, a > b) |",
              "| --- | --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        if r.metric == "mean_clv":
            continue
        lines.append(f"| {r.sample} | {LABELS[r.comparison.split(' − ')[0]]} − "
                     f"{LABELS[r.comparison.split(' − ')[1]]} | {r.metric} | "
                     f"{fmt(r.estimate, r.ci_low, r.ci_high)} | {r.p_a_gt_b:.3f} |")
    lines += ["", "## Threshold the agent ended up at", "",
              "![threshold and cumulative CLV](adaptive_edge.png)", ""]
    for arm in ("split", "adapt4", "adapt1"):
        if arm not in nbs:
            continue
        rules = edge_rules(nbs[arm])
        acc = [r for r in rules if r["status"] == "active"]
        tl = threshold_timeline(nbs[arm], ARMS[arm][2], ARMS[arm][1], days)
        changes = tl[tl.edge.diff().fillna(1) != 0]
        path = " → ".join(f"{e * 100:.0f}¢ from {d}" for d, e in zip(changes.date, changes.edge))
        lines += [f"**{LABELS[arm]}** — market-wide threshold: {path}. "
                  f"{len(acc)} accepted / {len(rules) - len(acc)} rejected `min_edge` proposals.", ""]
        if rules:
            lines += ["| Rule | Status | Slice | Edge | Gate decided | Superseded | Gate days | CLV $ without → with "
                      "| Changed trades |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in rules:
                ev = ("–" if r["clv_before"] is None else f"{r['clv_before']:+.2f} → {r['clv_after']:+.2f}")
                gd = "–" if not r["days"] else f"{r['days'][0]} – {r['days'][-1]}"
                lines.append(f"| {r['rule']} | {r['status']} | {r['when'] or 'all games'} | {r['edge'] * 100:.0f}¢ | "
                             f"{r['decided']} | {r['valid_until'] or '–'} | {gd} | {ev} | {r['cases']} |")
            lines.append("")
    if len(sub_frame):
        lines += ["## 304-point subsample (40%, seed 7606; same points as `llm_agent.md` / `edge_gate.md`)", "",
                  "| Setup | Rules accepted | Trades | CLV $ [95% CI] | P&L [95% CI] |", "| --- | --- | --- | --- | --- |"]
        for r in sub_frame.itertuples():
            lines.append(f"| {r.setup} | {r.rules_accepted} | {r.trades} | "
                         f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi)} | {fmt(r.pnl, r.pnl_lo, r.pnl_hi)} |")
        lines += ["", "On the subsample the learning arms see only the sampled decision points, but their gate "
                  "back-tests replay every decision point on the earlier days.", ""]
    lines += ["## Reading", "", "VERDICT_PLACEHOLDER", "",
              "Caveats: one test period (49 market days), one seed of the replay (deterministic); the counterfactual "
              "used by the reviewer for lowering ignores later re-entries, but the gate's back-test is exact. "
              "Gate back-tests early in February use January days on which M4 is in-sample.", ""]
    path = RESULTS / "adaptive_edge.md"
    old = path.read_text() if path.exists() else ""
    verdict = old.split("## Reading\n\n", 1)[1].split("\n\nCaveats:", 1)[0] if "## Reading\n\n" in old else ""
    text = "\n".join(lines) + "\n"
    if verdict and verdict != "VERDICT_PLACEHOLDER":
        text = text.replace("VERDICT_PLACEHOLDER", verdict)
    path.write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--subsample", action="store_true", help="also run the 304-point subsample arms")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_all(args.workers, args.subsample)
    report()


if __name__ == "__main__":
    main()
