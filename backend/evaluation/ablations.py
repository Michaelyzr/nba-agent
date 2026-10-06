"""E3: the README section 6 ablations, model-vs-market calibration and the headline chart, from one command.

    python -m evaluation.ablations                                     # real data in data/frozen
    python -m evaluation.ablations --source synthetic --test-end 2026-02-28 --out runs/abl-synthetic

1. Development period: the full agent learns rules (notebook saved).
2. Test period, same prices for every setup:
   full agent (starts from the development notebook, keeps learning),
   same agent with no learning (no rules at all), the same again trading the raw model
   probability instead of the market-anchored one,
   plain model with no agent (official statuses + the same models, trade when the
   gap after fees beats the threshold), and the agent on the win-rate placeholder
   instead of the trained models.
3. Win model vs the market on every test game; policy tests.
Writes CSV, Markdown and PNG to reports/results/ (or --out).
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import (DEFAULT_MIN_EDGE, DEFAULT_STAKE, OUT_STATUSES, MarketAgent, fee_per_contract,
                          load_source, record_forecaster, save_run)
from agents.notebook import Notebook
from evaluation.scorer import calibration_summary, model_vs_market, policy_tests, score_run
from replay import MAX_QUOTE_AGE, Order, Replay

from nba_agent_paths import RESULTS


def plain_policy(forecaster, edge=DEFAULT_MIN_EDGE, stake=DEFAULT_STAKE):
    """No agent: at every decision time, trade the best gap after fees if it beats the threshold."""
    held = set()

    def policy(view, game, now):
        if game.game_id in held:
            return []
        news = view.news(game.game_id)
        out = news.loc[news.status.astype(str).str.lower().isin(OUT_STATUSES), "player_id"].tolist()
        markets = view.markets(game.game_id)
        best = None
        for ticker, p in forecaster(view, game, markets, {"out": out}).items():
            q = view.quote(ticker)
            if q is None or now - q.ts > MAX_QUOTE_AGE:
                continue
            for side, price, p_side in (("yes", q.ask, p), ("no", 1 - q.bid, 1 - p)):
                gap = p_side - price - fee_per_contract(price)
                if best is None or gap > best[0]:
                    best = (gap, ticker, side, p)
        if best is None or best[0] <= edge:
            return []
        held.add(game.game_id)
        return [Order(best[1], best[2], best[3], stake, f"plain model gap {best[0]:+.3f}", news.news_id.tolist())]

    return policy


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def run_agent(name, tables, start, end, out_dir, **kwargs):
    agent = MarketAgent(**kwargs)
    hook = agent.on_day_end if agent.learn else None
    decisions, fills = Replay(tables, agent.policy, on_day_end=hook).run(start, end)
    save_run(f"{out_dir.name}/{name}", decisions, fills, agent)
    print(f"  {name}: {len(fills)} fills")
    return agent, fills


def headline_chart(fills_by_setup: dict, games: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, fills in fills_by_setup.items():
        if fills.empty:
            continue
        f = fills.merge(games[["game_id", "tip_time"]], on="game_id").sort_values("tip_time")
        ax.plot(f.tip_time, (f.clv * f.contracts).cumsum(), label=f"{label} ({len(f)} trades)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative closing-line value ($)")
    ax.set_title("Test period: cumulative closing-line value, with and without learning")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from forecast.api import Forecaster

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--dev-start", default="2025-11-01")
    ap.add_argument("--dev-end", default="2026-01-31")
    ap.add_argument("--test-start", default="2026-02-01")
    ap.add_argument("--test-end", default="2026-04-12")
    ap.add_argument("--llm", action="store_true", help="LLM steps for the full agent (needs a key)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = load_source(args.source, args.dev_start, args.test_end)
    forecaster = Forecaster.load()
    print(f"forecaster {forecaster.name}; development {args.dev_start}..{args.dev_end}")

    dev, _ = run_agent("dev-full", tables, args.dev_start, args.dev_end, args.out, forecaster=forecaster,
                       use_llm=args.llm)
    print(f"test {args.test_start}..{args.test_end}")
    t0, t1 = args.test_start, args.test_end
    _, full = run_agent("test-full", tables, t0, t1, args.out, notebook=copy.deepcopy(dev.notebook),
                        forecaster=forecaster, use_llm=args.llm)
    _, frozen = run_agent("test-frozen-rules", tables, t0, t1, args.out, notebook=copy.deepcopy(dev.notebook),
                          forecaster=forecaster, learn=False)
    _, no_learn = run_agent("test-no-learning", tables, t0, t1, args.out, notebook=Notebook(),
                            forecaster=forecaster, learn=False)
    run_agent("test-no-anchor", tables, t0, t1, args.out, notebook=Notebook(), forecaster=forecaster,
              learn=False, anchor=False)
    _, placeholder = run_agent("test-placeholder-model", tables, t0, t1, args.out, notebook=Notebook(),
                               forecaster=record_forecaster, learn=False)
    plain_d, plain = Replay(tables, plain_policy(forecaster)).run(t0, t1)
    save_run(f"{args.out.name}/test-no-agent", plain_d, plain)
    print(f"  test-no-agent: {len(plain)} fills")

    setups = {"test-full": "Full agent (learning)", "test-frozen-rules": "Agent, development rules frozen",
              "test-no-learning": "Agent, no learning",
              "test-no-anchor": "Agent, no learning, raw model (no market anchor)",
              "test-no-agent": "Plain model, no agent",
              "test-placeholder-model": "Agent on win-rate placeholder"}
    from replay import RUNS
    table = pd.DataFrame([{"setup": label, **score_run(RUNS / args.out.name / run, tables["games"])}
                          for run, label in setups.items()]).drop(columns="run")
    table.to_csv(args.out / "ablations.csv", index=False)
    cols = ["setup", "fills", "mean_clv", "clv_se", "pnl_after_fees", "roi", "brier_model", "brier_market",
            "max_drawdown", "kill_switch_trips", "rules_active"]
    (args.out / "ablations.md").write_text(markdown(table[[c for c in cols if c in table]].round(4)))
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))

    mvm = model_vs_market(tables, forecaster, t0, t1)
    mvm.to_csv(args.out / "model_vs_market.csv", index=False)
    cal = calibration_summary(mvm)
    cal.to_csv(args.out / "calibration.csv", index=False)
    print(cal.round(4).to_string(index=False))

    policy = policy_tests()
    policy.to_csv(args.out / "policy_tests.csv", index=False)
    print(f"policy tests blocked: {int(policy.blocked.sum())}/{len(policy)}")

    headline_chart({"with learning": full, "no learning": no_learn, "no agent": plain}, tables["games"],
                   args.out / "headline_clv.png")
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
