"""Edge-gate ablation: remove the 4¢ minimum edge (min_edge = 0; fees still charged) and compare.

    # deterministic sweep on the 304 test decision points (free), then the LLM arms at edge 0
    python -m evaluation.llm_agent_eval --backend heuristic --subsample 0.4 --setups anchor --min-edge 0 \
        --name edge-sweep-0 --report edge-sweep-0 --stem tmp            # repeat for 0.01 .. 0.04
    python -m evaluation.llm_agent_eval --backend deepseek --model deepseek-reasoner --subsample 0.4 \
        --setups tool_sceptic,tool --min-edge 0 --workers 6 --name test-deepseek-reasoner-edge0
    # full test period for context (no LLM): no learning, and learning with the split gate
    python -m evaluation.llm_agent_eval --backend heuristic --setups anchor --min-edge 0 --name edge-full-0
    python -m evaluation.edge_gate --run-split 0
    python -m evaluation.edge_gate                 # writes evaluation/results/edge_gate.{md,csv,png}

The 4¢ arms are the existing runs (runs/llm_agent/test-deepseek*/, runs/gate_audit/test-split_*).
A trade's gap after fees is p(side) − price − fee per contract; trades with gap ≤ 0.04 are the ones
the gate would have blocked at that decision point ("low-edge"). Day-clustered bootstrap CIs
(evaluation/stats.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, WINDOWS, close_clv, fmt, load_frozen, load_traces
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
RUNS = ROOT / "runs"
EDGES = (0.0, 0.01, 0.02, 0.03, 0.04)
GATE = 0.04
# arm -> (label, folder at 4¢, folder at edge 0)
ARMS = {
    "A": ("A. Deterministic anchor agent", OUT / "test-deepseek-reasoner" / "anchor", OUT / "edge-sweep-0" / "anchor"),
    "C": ("C. Reasoner tool agent, no sceptic", OUT / "test-deepseek-reasoner" / "tool",
          OUT / "test-deepseek-reasoner-edge0" / "tool"),
    "D": ("D. Reasoner tool agent + sceptic", OUT / "test-deepseek-reasoner" / "tool_sceptic",
          OUT / "test-deepseek-reasoner-edge0" / "tool_sceptic"),
    "D-chat": ("D-chat. Chat tool agent + sceptic", OUT / "test-deepseek" / "tool_sceptic",
               OUT / "test-deepseek-edge0" / "tool_sceptic"),
}
FULL = {
    "full-nolearn": ("A, full test period, no learning", RUNS / "gate_audit" / "test-split_nolearn",
                     OUT / "edge-full-0" / "anchor"),
    "full-split": ("Split gate + learning, full test period", RUNS / "gate_audit" / "test-split_full",
                   RUNS / "edge_gate" / "test-split_full-e0"),
}


def sweep_folder(e: float) -> Path:
    folder = OUT / f"edge-sweep-{e:g}" / "anchor"
    return ARMS["A"][1] if e == GATE and not (folder / "fills.parquet").exists() else folder


def fills_of(folder: Path, games, days) -> pd.DataFrame | None:
    path = folder / "fills.parquet"
    if not path.exists():
        return None
    f = pd.read_parquet(path)
    if not len(f):
        return f
    f = f[f.game_id.isin(games.loc[games.date.isin(days), "game_id"])].copy()
    p_side = np.where(f.side == "yes", f.p_model, 1 - f.p_model)
    f["gap"] = p_side - f.price - f.fee / f.contracts
    return f


def summary(f: pd.DataFrame, games, days) -> dict:
    t = day_table(f, games, days)
    b = bootstrap(t)
    row = {"trades": int(t.trades.sum()), "win_rate": float((f.pnl > 0).mean()) if len(f) else np.nan,
           "mean_gap": float(f.gap.mean()) if len(f) else np.nan}
    for m in ("mean_clv", "clv_dollars", "pnl", "roi"):
        row.update({m: b.loc[m, "estimate"], f"{m}_lo": b.loc[m, "ci_low"], f"{m}_hi": b.loc[m, "ci_high"]})
    row["p_clv_gt_0"] = b.loc["mean_clv", "p_gt_0"]
    return row, t


def run_split(edge: float):
    """Split gate + learning over the full test period at `edge` (no LLM: offline reviewer)."""
    from agents.graph import MarketAgent, save_run
    from forecast.api import Forecaster
    from replay import Replay
    tables = load_frozen()
    agent = MarketAgent(forecaster=Forecaster.load(), learn=True, gate="split", min_edge=edge)
    rp = Replay(tables, agent.policy, on_day_end=agent.on_day_end)
    decisions, fills = rp.run(*WINDOWS["test"])
    save_run(f"edge_gate/test-split_full-e{edge:g}", decisions, fills, agent, rp.kill_trips)
    print(f"split_full at edge {edge:g}: {len(fills)} fills, {len(agent.notebook.rules)} rules", flush=True)


def sceptic_table(folder: Path, fills: pd.DataFrame, tables, price_index) -> dict:
    traces = [t for t in load_traces(folder) if "sceptic" in t]
    rows = []
    for t in traces:
        c = t.get("candidate")
        clv = (close_clv(tables, price_index, c["ticker"], c["side"], c["price"], t["game_id"]) if c
               else np.nan)
        rows.append({"verdict": t["sceptic"]["verdict"], "gap": t.get("gap"), "clv": clv})
    s = pd.DataFrame(rows, columns=["verdict", "gap", "clv"])
    vetoed = s[s.verdict != "approve"]
    out = {"sceptic_calls": len(s), "vetoes": len(vetoed), "vetoed_mean_clv": vetoed.clv.mean(),
           "kept_mean_clv": fills.clv.mean() if len(fills) else np.nan}
    for name, part in (("low", s[s.gap <= GATE]), ("high", s[s.gap > GATE])):
        v = part[part.verdict != "approve"]
        out.update({f"{name}_calls": len(part), f"{name}_vetoes": len(v), f"{name}_vetoed_mean_clv": v.clv.mean()})
    return out


def spend(folders: list) -> dict:
    """LLM calls first fetched for these runs (not cache hits), at the list prices."""
    from agents.llm_client import cost_usd
    calls = tok_in = tok_out = 0
    cost = 0.0
    for folder in folders:
        meta = folder / "meta.json"
        if not meta.exists():
            continue
        model = json.loads(meta.read_text())["model"]
        for t in load_traces(folder):
            x = t.get("llm", {})
            fresh = x.get("calls", 0) - x.get("cached", 0)
            if fresh and x.get("calls"):
                share = fresh / x["calls"]
                calls += fresh
                tok_in += x.get("tokens_in", 0) * share
                tok_out += x.get("tokens_out", 0) * share
                cost += cost_usd(model, x.get("tokens_in", 0) * share, x.get("tokens_out", 0) * share)
    return {"fresh_calls": calls, "tokens_in": int(tok_in), "tokens_out": int(tok_out), "cost_usd": cost}


def chart(sweep: pd.DataFrame, rows: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = sweep.min_edge * 100
    for ax, col, title in ((axes[0], "clv_dollars", "CLV $ (sum of CLV × contracts)"),
                           (axes[1], "mean_clv", "Mean CLV per contract ($)"), (axes[2], "trades", "Trades")):
        if col == "trades":
            ax.plot(x, sweep[col], "o-", color="#1f77b4", lw=2, label="A. deterministic (sweep)")
        else:
            ax.errorbar(x, sweep[col], yerr=[sweep[col] - sweep[f"{col}_lo"], sweep[f"{col}_hi"] - sweep[col]],
                        fmt="o-", color="#1f77b4", lw=2, capsize=3, label="A. deterministic (sweep)")
        for arm, color, dx in (("C", "#ff7f0e", -0.12), ("D", "#2ca02c", 0.12), ("D-chat", "#8c564b", 0.24)):
            r = rows[rows.arm == arm].sort_values("min_edge")
            if len(r) < 2:
                continue
            xs = r.min_edge * 100 + dx
            if col == "trades":
                ax.plot(xs, r[col], "s--", color=color, label=r.label.iloc[0])
            else:
                ax.errorbar(xs, r[col], yerr=[(r[col] - r[f"{col}_lo"]).fillna(0), (r[f"{col}_hi"] - r[col]).fillna(0)],
                            fmt="s--", color=color, capsize=3, label=r.label.iloc[0])
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(GATE * 100, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("minimum edge after fees (¢)")
        ax.set_title(title)
    axes[2].legend(fontsize=8)
    fig.suptitle("Removing the 4¢ edge gate: 304 test decision points (40% subsample), day-clustered 95% CIs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report():
    from evaluation.walkforward import market_days
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *WINDOWS["test"])
    prices = tables["prices"].sort_values("ts")
    price_index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}

    sweep = []
    for e in EDGES:
        f = fills_of(sweep_folder(e), games, days)
        if f is not None:
            row, _ = summary(f, games, days)
            sweep.append({"kind": "sweep", "arm": "A", "label": ARMS["A"][0], "min_edge": e, **row})
    sweep = pd.DataFrame(sweep)

    rows, pairs, splits, scep = [], [], [], []
    for arm, (label, f4_dir, f0_dir) in {**ARMS, **FULL}.items():
        f4, f0 = fills_of(f4_dir, games, days), fills_of(f0_dir, games, days)
        if f4 is None or f0 is None:
            continue
        r4, t4 = summary(f4, games, days)
        rows.append({"kind": "arm", "arm": arm, "label": label, "min_edge": GATE, **r4})
        r0, t0 = summary(f0, games, days)
        rows.append({"kind": "arm", "arm": arm, "label": label, "min_edge": 0.0, **r0})
        d = paired(t0, t4)
        for m in ("mean_clv", "clv_dollars", "pnl"):
            pairs.append({"kind": "paired", "arm": arm, "label": label, "metric": m, "estimate": d.loc[m, "estimate"],
                          "ci_low": d.loc[m, "ci_low"], "ci_high": d.loc[m, "ci_high"], "p_a_gt_b": d.loc[m, "p_a_gt_b"]})
        key = ["game_id", "as_of", "market_ticker", "side"]
        if not len(f0) or not len(f4):
            in4, in0 = np.zeros(len(f0), bool), np.zeros(len(f4), bool)
        else:
            in4 = f0.set_index(key).index.isin(f4.set_index(key).index)
            in0 = f4.set_index(key).index.isin(f0.set_index(key).index)
        for part, sub in (("low-edge trades (gap ≤ 4¢), edge 0", f0[f0.gap <= GATE] if len(f0) else f0),
                          ("normal trades (gap > 4¢), edge 0", f0[f0.gap > GATE] if len(f0) else f0),
                          ("same trade as at 4¢", f0[in4]),
                          ("new at edge 0", f0[~in4]),
                          ("4¢ trades lost at edge 0", f4[~in0])):
            if not len(sub):
                splits.append({"kind": "split", "arm": arm, "label": label, "part": part, "trades": 0})
                continue
            r, _ = summary(sub, games, days)
            splits.append({"kind": "split", "arm": arm, "label": label, "part": part, **r})
        if arm.startswith("D"):
            scep.append({"kind": "sceptic", "arm": arm, "label": label, "min_edge": 0.0,
                         **sceptic_table(f0_dir, f0, tables, price_index)})
            scep.append({"kind": "sceptic", "arm": arm, "label": label, "min_edge": GATE,
                         **sceptic_table(f4_dir, f4, tables, price_index)})
    rows, pairs, splits, scep = map(pd.DataFrame, (rows, pairs, splits, scep))
    cost = spend([OUT / "test-deepseek-reasoner-edge0" / s for s in ("tool_sceptic", "tool")]
                 + [OUT / "test-deepseek-edge0" / "tool_sceptic"])
    pd.concat([sweep, rows, pairs, splits, scep], ignore_index=True).to_csv(RESULTS / "edge_gate.csv", index=False)
    chart(sweep, rows[rows.arm.isin(list(ARMS))], RESULTS / "edge_gate.png")

    def line(r):
        return (f"{r.trades} | {fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
                f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} | {fmt(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | "
                + ("–" if pd.isna(r.win_rate) else f"{r.win_rate:.0%}"))

    md = ["# Removing the 4¢ edge gate", "",
          "Question: do the extra trades the agents place when the minimum edge after fees is 0 (instead of 4¢) "
          "beat the closing line? Fees are still charged and every other setting is unchanged (stake $20, caps "
          "$50/$100/$300, fills at the ask plus the Kalshi fee, default M4, no learning unless stated). Same 304 "
          f"test decision points (fixed 40% subsample, seed {SEED}) as `llm_agent.md`. For the LLM arms the prompt "
          "text changes (\"more than 0\" instead of \"more than 0.04\"), so most calls are new. CLV is per contract "
          f"against the mid at tip; CLV $ = CLV × contracts. 95% CIs from a day-clustered bootstrap ({REPS} "
          f"replicates, seed {SEED}).", ""]

    def get(arm, e):
        r = rows[(rows.arm == arm) & (rows.min_edge == e)]
        return r.iloc[0] if len(r) else None

    a0, a4, d0, d4 = get("A", 0.0), get("A", GATE), get("D", 0.0), get("D", GATE)
    if a0 is not None and a4 is not None:
        dp = pairs[(pairs.arm == "A") & (pairs.metric == "clv_dollars")].iloc[0]
        md += [f"**Summary.** Without the gate the deterministic agent trades {a0.trades} times instead of "
               f"{a4.trades}; the extra trades lose to the close at the same rate per contract, so CLV $ falls from "
               f"{fmt(a4.clv_dollars, a4.clv_dollars_lo, a4.clv_dollars_hi, '$')} to "
               f"{fmt(a0.clv_dollars, a0.clv_dollars_lo, a0.clv_dollars_hi, '$')} (paired difference "
               f"{fmt(dp.estimate, dp.ci_low, dp.ci_high, '$')})."]
    if d0 is not None and d4 is not None and len(scep):
        s0 = scep[(scep.arm == "D") & (scep.min_edge == 0.0)].iloc[0]
        md[-1] += (f" With the sceptic, the reasoner agent goes from {d4.trades} to {d0.trades} trades: it proposes "
                   f"{s0.sceptic_calls} orders at edge 0 and the sceptic vetoes {s0.vetoes} of them ({s0.low_vetoes} "
                   f"of {s0.low_calls} low-edge ones), so it still holds the line without the gate.")
    md += ["", "## Deterministic agent (A): CLV vs minimum edge", "",
          "| Min edge | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | Win rate |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in sweep.itertuples():
        md.append(f"| {r.min_edge * 100:.0f}¢ | {line(r)} |")
    md += ["", "## Every arm at 4¢ vs 0", "",
           "| Arm | Min edge | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | Win rate |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows.itertuples():
        md.append(f"| {r.label} | {r.min_edge * 100:.0f}¢ | {line(r)} |")
    missing = [label for label, _, f0_dir in {**ARMS, **FULL}.values() if not (f0_dir / "fills.parquet").exists()]
    if missing:
        md += ["", "Not run at edge 0 (no row above): " + "; ".join(missing) + "."]
    md += ["", "Paired difference, edge 0 minus 4¢, on the same game-days (p one-sided for edge 0 > 4¢):", "",
           "| Arm | Metric | Difference [95% CI] | p |", "| --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        name = {"mean_clv": "mean CLV", "clv_dollars": "CLV $", "pnl": "P&L"}[r.metric]
        md.append(f"| {r.label} | {name} | {fmt(r.estimate, r.ci_low, r.ci_high, 'clv' if r.metric == 'mean_clv' else '$')} "
                  f"| {r.p_a_gt_b:.3f} |")
    md += ["", "## Where the difference comes from", "",
           "Trades at edge 0 split by their gap after fees, and matched to the 4¢ run on (game, decision time, "
           "contract, side). With one position per game, a low-edge trade can pre-empt a later, larger-edge trade "
           "on the same game, so a few 4¢ trades disappear.", "",
           "| Arm | Trades | n | Mean CLV [95% CI] | CLV $ [95% CI] | P&L [95% CI] | Win rate |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in splits.itertuples():
        md.append(f"| {r.label} | {r.part} | " + (line(r) if r.trades else "0 | – | – | – | –") + " |")
    if len(scep):
        md += ["", "## Sceptic (D)", "",
               "Vetoed CLV is counterfactual: the side's mid at tip minus the price the order would have paid.", "",
               "| Arm | Min edge | Sceptic calls | Vetoes | Vetoed mean CLV | Kept mean CLV | Low-edge (≤ 4¢) calls / vetoes "
               "(vetoed CLV) | Gap > 4¢ calls / vetoes (vetoed CLV) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]

        def c(v):
            return "–" if pd.isna(v) else f"{v:+.4f}"
        for r in scep.itertuples():
            md.append(f"| {r.label} | {r.min_edge * 100:.0f}¢ | {r.sceptic_calls} | {r.vetoes} | {c(r.vetoed_mean_clv)} | "
                      f"{c(r.kept_mean_clv)} | {r.low_calls} / {r.low_vetoes} ({c(r.low_vetoed_mean_clv)}) | "
                      f"{r.high_calls} / {r.high_vetoes} ({c(r.high_vetoed_mean_clv)}) |")
    md += ["", f"DeepSeek calls fetched for the edge-0 runs (not cache hits): {cost['fresh_calls']:,}, "
               f"{cost['tokens_in']:,} / {cost['tokens_out']:,} tokens in / out, about ${cost['cost_usd']:.2f} at the "
               "list prices in `agents/llm_client.PRICES`.", "", "![Edge gate](edge_gate.png)", ""]
    (RESULTS / "edge_gate.md").write_text("\n".join(md) + "\n")
    print((RESULTS / "edge_gate.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-split", type=float, default=None, help="run split gate + learning at this min edge")
    args = ap.parse_args()
    if args.run_split is not None:
        run_split(args.run_split)
        return
    report()


if __name__ == "__main__":
    main()
