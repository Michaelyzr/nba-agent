"""I: evaluate episodic retrieval memory against no learning and the template reviewer.

    python -m evaluation.memory_eval                 # ~1–2 h offline on frozen data
    python -m evaluation.memory_eval --report-only   # re-score runs/memory/

Pre-registered in docs/preregistration_rules_memory.md (committed before this run).
Memory alone (learn=False) and memory + template reviewer (learn=True, gate=split).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd

from agents.graph import MarketAgent, save_run
from agents.memory import EpisodicMemory, MemoryAgent
from agents.notebook import Notebook
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import fmt_ci, market_days
from evaluation.llm_agent_eval import load_frozen
from replay import RUNS, Replay

RESULTS = Path(__file__).resolve().parent / "results"
DEV = ("2025-11-01", "2026-01-31")
TEST = ("2026-02-01", "2026-04-12")
SETUPS = ("no_learning", "template", "memory", "memory_template")


def pick_gate() -> str:
    try:
        MarketAgent(gate="split", learn=False)
        return "split"
    except (TypeError, ValueError):
        return "legacy"


def run_setup(name, tables, start, end, *, forecaster, notebook=None, memory=None, learn=False,
              use_memory=False, gate="split"):
    t0 = time.time()
    nb = copy.deepcopy(notebook) if notebook is not None else Notebook()
    if use_memory:
        mem = copy.deepcopy(memory) if memory is not None else EpisodicMemory()
        agent = MemoryAgent(notebook=nb, forecaster=forecaster, learn=learn, memory=mem, gate=gate)
    else:
        agent = MarketAgent(notebook=nb, forecaster=forecaster, learn=learn, gate=gate)
    hook = agent.on_day_end  # memory always stores; the parent review only runs when learn=True
    if not use_memory and not learn:
        hook = None
    decisions, fills = Replay(tables, agent.policy, on_day_end=hook).run(start, end)
    save_run(f"memory/{name}", decisions, fills, agent)
    meta = {"name": name, "fills": len(fills), "seconds": time.time() - t0, "gate": gate,
            "rules_active": sum(1 for r in agent.notebook.rules if r["status"] == "active"),
            "episodes": len(getattr(agent, "memory", EpisodicMemory()).episodes),
            "vetoes": len(getattr(agent, "vetoes", [])),
            "recalls": getattr(agent, "recalls", 0)}
    if use_memory:
        agent.memory.save(RUNS / "memory" / name / "memory.json")
        vetoes = pd.DataFrame(agent.vetoes)
        if len(vetoes):
            vetoes.to_csv(RUNS / "memory" / name / "vetoes.csv", index=False)
            meta["veto_mean_clv"] = float(vetoes.clv.mean()) if "clv" in vetoes else float("nan")
    (RUNS / "memory" / name / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  {name}: {meta['fills']} fills, {meta.get('vetoes', 0)} vetoes in "
          f"{meta['seconds'] / 60:.1f} min", flush=True)
    return agent, fills, meta


def run_all(gate: str):
    from forecast.api import Forecaster

    tables = load_frozen()
    forecaster = Forecaster.load()
    print(f"forecaster {forecaster.name}; gate={gate}")
    print(f"development {DEV[0]}..{DEV[1]}")
    tpl_dev, _, _ = run_setup("dev-template", tables, *DEV, forecaster=forecaster, learn=True, gate=gate)
    mem_dev, _, _ = run_setup("dev-memory", tables, *DEV, forecaster=forecaster, learn=False,
                              use_memory=True, gate=gate)
    mem_tpl_dev, _, _ = run_setup("dev-memory_template", tables, *DEV, forecaster=forecaster, learn=True,
                                  use_memory=True, gate=gate)
    print(f"test {TEST[0]}..{TEST[1]}")
    run_setup("test-no_learning", tables, *TEST, forecaster=forecaster, learn=False, gate=gate)
    run_setup("test-template", tables, *TEST, forecaster=forecaster, learn=True, gate=gate,
              notebook=tpl_dev.notebook)
    run_setup("test-memory", tables, *TEST, forecaster=forecaster, learn=False, use_memory=True,
              memory=mem_dev.memory, gate=gate)
    run_setup("test-memory_template", tables, *TEST, forecaster=forecaster, learn=True, use_memory=True,
              notebook=mem_tpl_dev.notebook, memory=mem_tpl_dev.memory, gate=gate)


def run_one(name: str, gate: str):
    """One setup in its own process; test setups warm-start from the saved development notebook and memory."""
    from forecast.api import Forecaster

    tables, forecaster = load_frozen(), Forecaster.load()
    window, setup = name.split("-", 1)
    span = DEV if window == "dev" else TEST
    notebook = memory = None
    if window == "test" and setup != "no_learning":
        dev = RUNS / "memory" / f"dev-{setup}"
        notebook = Notebook.load(dev / "notebook.json")
        if setup.startswith("memory"):
            memory = EpisodicMemory.load(dev / "memory.json")
    print(f"{name}: gate={gate}", flush=True)
    run_setup(name, tables, *span, forecaster=forecaster, notebook=notebook, memory=memory,
              learn=setup in ("template", "memory_template"), use_memory=setup.startswith("memory"), gate=gate)


def summary_row(label, setup, table, meta=None) -> dict:
    b = bootstrap(table)
    row = {"window": label, "setup": setup, "days": len(table), "trades": int(table.trades.sum())}
    for m in b.index:
        row.update({m: b.loc[m, "estimate"], f"{m}_lo": b.loc[m, "ci_low"], f"{m}_hi": b.loc[m, "ci_high"]})
    row["p_clv_gt_0"] = b.loc["mean_clv", "p_gt_0"]
    if meta:
        row.update({k: meta.get(k) for k in ("vetoes", "recalls", "episodes", "veto_mean_clv", "rules_active",
                                             "gate")})
    return row


def chart(tabs: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(tabs)
    clv = [float(t.clv_dollars.sum()) for t in tabs.values()]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(labels, clv, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("CLV dollars")
    ax.set_title("Test 1 Feb – 12 Apr: CLV $ by setup")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(gate_used: str):
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *TEST)
    tabs, metas, rows = {}, {}, []
    for s in SETUPS:
        folder = RUNS / "memory" / f"test-{s}"
        fills = pd.read_parquet(folder / "fills.parquet") if (folder / "fills.parquet").exists() else pd.DataFrame()
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {"gate": gate_used}
        metas[s] = meta
        tabs[s] = day_table(fills, games, days)
        rows.append(summary_row("Test 1 Feb–12 Apr", s, tabs[s], meta))
    summary = pd.DataFrame(rows)
    pairs = []
    for a, b in (("memory", "no_learning"), ("memory", "template"), ("memory_template", "no_learning"),
                 ("memory_template", "template"), ("template", "no_learning")):
        d = paired(tabs[a], tabs[b])
        for m in ("mean_clv", "clv_dollars", "pnl"):
            r = d.loc[m]
            pairs.append({"comparison": f"{a} − {b}", "metric": m, "estimate": r.estimate,
                          "ci_low": r.ci_low, "ci_high": r.ci_high, "p_a_gt_b": r.p_a_gt_b})
    pairs = pd.DataFrame(pairs)
    summary.to_csv(RESULTS / "memory.csv", index=False)
    pairs.to_csv(RESULTS / "memory_pairs.csv", index=False)
    chart(tabs, RESULTS / "memory.png")

    beat = pairs[(pairs.comparison == "memory − template") & (pairs.metric == "clv_dollars")].iloc[0]
    verdict = ("Memory beats the template on CLV $" if beat.ci_low > 0
               else "Memory does not beat the template on CLV $ (null / inconclusive)")
    lines = [
        "# I: episodic retrieval memory", "",
        f"Pre-registered in `docs/preregistration_rules_memory.md`. Gate for learning arms: "
        f"**{metas.get('template', {}).get('gate', gate_used)}**. Memory veto: k=20 neighbours, "
        f"min 40 settled episodes, skip when mean CLV ≤ −0.01. Bootstrap: {REPS} replicates, seed {SEED}.", "",
        f"**Verdict.** {verdict}. Estimate memory − template CLV $ = "
        f"{beat.estimate:+.1f} [{beat.ci_low:+.1f}, {beat.ci_high:+.1f}], "
        f"p(memory > template) = {beat.p_a_gt_b:.3f}.", "",
        "## Test window totals", "",
        "| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L [95% CI] | Vetoes | Veto mean CLV |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.itertuples():
        vclv = getattr(r, "veto_mean_clv", float("nan"))
        vclv_s = "–" if pd.isna(vclv) else f"{vclv:+.4f}"
        lines.append(
            f"| {r.setup} | {r.trades} | {fmt_ci(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
            f"{fmt_ci(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} | "
            f"{fmt_ci(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | {getattr(r, 'vetoes', 0) or 0} | {vclv_s} |")
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Metric | Difference [95% CI] | p (one-sided) |", "| --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        kind = "clv" if r.metric == "mean_clv" else "$"
        lines.append(f"| {r.comparison} | {r.metric} | {fmt_ci(r.estimate, r.ci_low, r.ci_high, kind)} | "
                     f"{r.p_a_gt_b:.3f} |")
    lines += ["", "A negative veto mean CLV means the skipped trades would have lost closing-line value "
              "on average (the veto was directionally right). Shadow episodes keep those outcomes in "
              "memory for later recalls without ever filling them.", "",
              "![CLV dollars by setup](memory.png)", ""]
    (RESULTS / "memory.md").write_text("\n".join(lines))
    print((RESULTS / "memory.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--gate", choices=["split", "legacy", "auto"], default="auto")
    ap.add_argument("--only", help="run one setup, e.g. dev-memory or test-memory_template, then exit")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RUNS / "memory").mkdir(parents=True, exist_ok=True)
    gate = pick_gate() if args.gate == "auto" else args.gate
    if args.only:
        run_one(args.only, gate)
        return
    if not args.report_only:
        run_all(gate)
    report(gate)


if __name__ == "__main__":
    main()
