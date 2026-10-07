"""H: evaluate the DSL reviewer against the template reviewer and no learning.

    python -m evaluation.rule_dsl_eval                 # ~1–2 h offline on frozen data
    python -m evaluation.rule_dsl_eval --report-only   # re-score runs/rule_dsl/

Pre-registered in docs/preregistration_rules_memory.md (committed before this run).
Uses the split (disjoint days, total CLV dollars) gate when MarketAgent accepts
gate="split"; otherwise falls back to legacy and records which one ran.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import MarketAgent, save_run
from agents.notebook import Notebook
from agents.rule_dsl import DSLReviewAgent, stub_proposer
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import fmt_ci, market_days
from evaluation.llm_agent_eval import load_frozen
from replay import RUNS, Replay

RESULTS = Path(__file__).resolve().parent / "results"
DEV = ("2025-11-01", "2026-01-31")
TEST = ("2026-02-01", "2026-04-12")
SETUPS = ("no_learning", "template", "dsl")


def pick_gate() -> str:
    """Prefer the new split gate; fall back if an older MarketAgent is somehow loaded."""
    try:
        MarketAgent(gate="split", learn=False)
        return "split"
    except (TypeError, ValueError):
        return "legacy"


def run_setup(name: str, tables, start, end, *, forecaster, notebook=None, learn=False, dsl=False,
              gate="split", proposer=None):
    t0 = time.time()
    nb = copy.deepcopy(notebook) if notebook is not None else Notebook()
    if dsl:
        agent = DSLReviewAgent(notebook=nb, forecaster=forecaster, learn=learn, gate=gate,
                               proposer=proposer)
    else:
        agent = MarketAgent(notebook=nb, forecaster=forecaster, learn=learn, gate=gate)
    hook = agent.on_day_end if learn else None
    decisions, fills = Replay(tables, agent.policy, on_day_end=hook).run(start, end)
    save_run(f"rule_dsl/{name}", decisions, fills, agent)
    meta = {"name": name, "fills": len(fills), "seconds": time.time() - t0, "gate": gate,
            "rules_active": sum(1 for r in agent.notebook.rules if r["status"] == "active"),
            "rules_total": len(agent.notebook.rules)}
    if isinstance(agent, DSLReviewAgent):
        props = pd.DataFrame(agent.proposals)
        meta.update(proposals=len(props), parse_valid=int(props.valid.sum()) if len(props) else 0,
                    gate_accepted=int((props.gate == "accepted").sum()) if len(props) else 0,
                    gate_rejected=int((props.gate == "rejected").sum()) if len(props) else 0,
                    proposer=agent.proposer_name)
        props.to_csv(RUNS / "rule_dsl" / name / "proposals.csv", index=False)
    (RUNS / "rule_dsl" / name / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  {name}: {meta['fills']} fills in {meta['seconds'] / 60:.1f} min "
          f"(active rules {meta['rules_active']})", flush=True)
    return agent, fills, meta


def run_all(gate: str):
    from forecast.api import Forecaster

    tables = load_frozen()
    forecaster = Forecaster.load()
    print(f"forecaster {forecaster.name}; gate={gate}; Gemini={'yes' if __import__('llm').available() else 'no'}")
    # Development warm-start
    print(f"development {DEV[0]}..{DEV[1]}")
    tpl_dev, _, _ = run_setup("dev-template", tables, *DEV, forecaster=forecaster, learn=True, gate=gate)
    dsl_dev, _, _ = run_setup("dev-dsl", tables, *DEV, forecaster=forecaster, learn=True, dsl=True, gate=gate)
    # Test
    print(f"test {TEST[0]}..{TEST[1]}")
    run_setup("test-no_learning", tables, *TEST, forecaster=forecaster, learn=False, gate=gate)
    run_setup("test-template", tables, *TEST, forecaster=forecaster, learn=True, gate=gate,
              notebook=tpl_dev.notebook)
    run_setup("test-dsl", tables, *TEST, forecaster=forecaster, learn=True, dsl=True, gate=gate,
              notebook=dsl_dev.notebook)


def summary_row(label, setup, table, meta=None) -> dict:
    b = bootstrap(table)
    row = {"window": label, "setup": setup, "days": len(table), "trades": int(table.trades.sum())}
    for m in b.index:
        row.update({m: b.loc[m, "estimate"], f"{m}_lo": b.loc[m, "ci_low"], f"{m}_hi": b.loc[m, "ci_high"]})
    row["p_clv_gt_0"] = b.loc["mean_clv", "p_gt_0"]
    if meta:
        row.update({k: meta.get(k) for k in ("proposals", "parse_valid", "gate_accepted", "gate_rejected",
                                             "rules_active", "proposer", "gate") if k in meta})
        if meta.get("proposals"):
            row["parse_valid_rate"] = meta["parse_valid"] / meta["proposals"]
            gated = meta["gate_accepted"] + meta["gate_rejected"]
            row["gate_pass_rate"] = meta["gate_accepted"] / gated if gated else float("nan")
    return row


def chart(tabs: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(tabs)
    clv = [float(t.clv_dollars.sum()) for t in tabs.values()]
    trades = [int(t.trades.sum()) for t in tabs.values()]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].bar(labels, clv, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    ax[0].axhline(0, color="black", lw=0.7)
    ax[0].set_ylabel("CLV dollars")
    ax[0].set_title("Test 1 Feb – 12 Apr: CLV $")
    ax[1].bar(labels, trades, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    ax[1].set_ylabel("trades")
    ax[1].set_title("Trades")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(gate_used: str):
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *TEST)
    tabs, metas, rows = {}, {}, []
    for s in SETUPS:
        fills = pd.read_parquet(RUNS / "rule_dsl" / f"test-{s}" / "fills.parquet") if (
            RUNS / "rule_dsl" / f"test-{s}" / "fills.parquet").exists() else pd.DataFrame()
        meta_path = RUNS / "rule_dsl" / f"test-{s}" / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"gate": gate_used}
        metas[s] = meta
        tabs[s] = day_table(fills, games, days)
        rows.append(summary_row("Test 1 Feb–12 Apr", s, tabs[s], meta if s == "dsl" else {**meta, "setup": s}))
    summary = pd.DataFrame(rows)
    pairs = []
    for a, b in (("dsl", "no_learning"), ("dsl", "template"), ("template", "no_learning")):
        d = paired(tabs[a], tabs[b])
        for m in ("mean_clv", "clv_dollars", "pnl"):
            r = d.loc[m]
            pairs.append({"comparison": f"{a} − {b}", "metric": m, "estimate": r.estimate,
                          "ci_low": r.ci_low, "ci_high": r.ci_high, "p_a_gt_b": r.p_a_gt_b})
    pairs = pd.DataFrame(pairs)
    summary.to_csv(RESULTS / "rule_dsl.csv", index=False)
    pairs.to_csv(RESULTS / "rule_dsl_pairs.csv", index=False)
    chart(tabs, RESULTS / "rule_dsl.png")

    dsl_meta = metas.get("dsl", {})
    props = 0, 0, 0, 0
    if (RUNS / "rule_dsl" / "test-dsl" / "proposals.csv").exists():
        p = pd.read_csv(RUNS / "rule_dsl" / "test-dsl" / "proposals.csv")
        props = len(p), int(p.valid.sum()), int((p.gate == "accepted").sum()), int((p.gate == "rejected").sum())
    beat_tpl = pairs[(pairs.comparison == "dsl − template") & (pairs.metric == "clv_dollars")].iloc[0]
    verdict = ("DSL beats the template on CLV $" if beat_tpl.ci_low > 0
               else "DSL does not beat the template on CLV $ (null / inconclusive)")

    lines = [
        "# H: LLM / stub rules compiled to a safe DSL", "",
        f"Pre-registered in `docs/preregistration_rules_memory.md`. Gate used: **{dsl_meta.get('gate', gate_used)}** "
        f"(split = disjoint selection and test days, judged on total CLV dollars; legacy = published mean-CLV gate). "
        f"Proposer: **{dsl_meta.get('proposer', 'n/a')}**. Offline unless a Gemini key is set; LLM answers are "
        f"cached under `runs/llm_cache/`. Bootstrap: {REPS} day-clustered replicates, seed {SEED}.", "",
        f"**Verdict.** {verdict}. Estimate dsl − template CLV $ = "
        f"{beat_tpl.estimate:+.1f} [{beat_tpl.ci_low:+.1f}, {beat_tpl.ci_high:+.1f}], "
        f"p(dsl > template) = {beat_tpl.p_a_gt_b:.3f}.", "",
        "## Test window totals", "",
        "| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L [95% CI] | Active rules |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.setup} | {r.trades} | {fmt_ci(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
            f"{fmt_ci(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} | "
            f"{fmt_ci(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | {getattr(r, 'rules_active', '–') or 0} |")
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Metric | Difference [95% CI] | p (one-sided) |", "| --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        kind = "clv" if r.metric == "mean_clv" else "$"
        lines.append(f"| {r.comparison} | {r.metric} | {fmt_ci(r.estimate, r.ci_low, r.ci_high, kind)} | "
                     f"{r.p_a_gt_b:.3f} |")
    n, v, acc, rej = props
    lines += ["", "## DSL process metrics (test window)", "",
              f"- Proposals written: {n}",
              f"- Parse-valid: {v} ({v / n:.0%} of written)" if n else "- Parse-valid: –",
              f"- Gate accepted / rejected: {acc} / {rej}"
              + (f" (pass rate {acc / (acc + rej):.0%} of gated)" if acc + rej else ""),
              f"- Active rules at end of test: {dsl_meta.get('rules_active', '–')}", "",
              "Invalid proposals never reach the gate. Valid ones still go through `MarketAgent.gate`.", "",
              "![CLV dollars and trade counts](rule_dsl.png)", ""]
    (RESULTS / "rule_dsl.md").write_text("\n".join(lines))
    print((RESULTS / "rule_dsl.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--gate", choices=["split", "legacy", "auto"], default="auto")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RUNS / "rule_dsl").mkdir(parents=True, exist_ok=True)
    gate = pick_gate() if args.gate == "auto" else args.gate
    if not args.report_only:
        run_all(gate)
    report(gate)


if __name__ == "__main__":
    main()
