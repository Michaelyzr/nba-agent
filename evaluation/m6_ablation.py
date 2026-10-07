"""M6 trading ablation: the agent on the market-impact signal against the existing setups and never-trade.

    python -m evaluation.m6_ablation                      # runs into runs/m6/, tables to evaluation/results/
    python -m evaluation.m6_ablation --reuse              # rescore runs already in runs/m6/

Test period 1 Feb - 12 Apr 2026 and a holdout of every priced game after it
(play-in and playoffs). Setups:
  M6 agent, no learning; M6 agent with learning (empty notebook at the start of
  the test period, carried into the holdout); M6 with the edge threshold off
  (a diagnostic of what M6's direction is worth after costs, not a strategy);
  the anchor agent with and without learning, the raw M4 agent, and never-trade.
The anchor agent with learning on the test period is the existing
runs/results/test-full run (development notebook from Nov - Jan).
Confidence intervals resample game-days (evaluation/stats.py).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import MarketAgent, save_run
from agents.notebook import Notebook
from data_sources import FROZEN
from evaluation.ablations import markdown
from evaluation.stats import bootstrap, day_table
from replay import RUNS, Replay, load_tables

RESULTS = Path(__file__).resolve().parent / "results"
TEST = ("2026-02-01", "2026-04-12")
HOLDOUT = ("2026-04-13", "2026-06-30")
FORCED_EDGE = -1.0                 # take the better side at the first decision time of every game


def run(name, tables, period, reuse, **kwargs):
    out = RUNS / "m6" / name
    if reuse and (out / "fills.parquet").exists():
        notebook = Notebook.load(out / "notebook.json") if (out / "notebook.json").exists() else Notebook()
        return pd.read_parquet(out / "fills.parquet"), notebook
    agent = MarketAgent(**kwargs)
    decisions, fills = Replay(tables, agent.policy, on_day_end=agent.on_day_end if agent.learn else None).run(*period)
    save_run(f"m6/{name}", decisions, fills, agent)
    print(f"  {name}: {len(fills)} fills, P&L {fills.pnl.sum() if len(fills) else 0:+.2f}")
    return fills, agent.notebook


def score(label, period_name, fills, games, days, rules=None) -> dict:
    table = day_table(fills, games, days)
    b = bootstrap(table)
    staked = float(table.staked.sum())
    row = {"period": period_name, "setup": label, "trades": int(table.trades.sum()),
           "mean_clv": b.loc["mean_clv", "estimate"], "mean_clv_lo": b.loc["mean_clv", "ci_low"],
           "mean_clv_hi": b.loc["mean_clv", "ci_high"], "clv_dollars": b.loc["clv_dollars", "estimate"],
           "pnl": b.loc["pnl", "estimate"], "pnl_lo": b.loc["pnl", "ci_low"], "pnl_hi": b.loc["pnl", "ci_high"],
           "staked": staked, "roi": float(table.pnl.sum() / staked) if staked else 0.0,
           "rules_active": rules}
    if not row["trades"]:
        row.update(mean_clv=np.nan, mean_clv_lo=np.nan, mean_clv_hi=np.nan, pnl_lo=0.0, pnl_hi=0.0)
    return row


def active_rules(notebook) -> int:
    return sum(r.get("status") == "active" for r in notebook.rules)


def main():
    from forecast.api import Forecaster
    from forecast.impact import load_impact

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="rescore runs already saved in runs/m6/")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--data", type=Path, default=FROZEN, help="folder with the frozen tables")
    args = ap.parse_args()
    tables = load_tables(args.data)
    games = tables["games"]
    priced = games[games.game_id.isin(tables["markets"].game_id)]
    forecaster, impact = Forecaster.load(), load_impact()
    m6 = dict(forecaster=forecaster, impact=impact)
    rows, setups_nb = [], {}

    for period_name, period in (("test", TEST), ("holdout", HOLDOUT)):
        days = priced.loc[(priced.date >= period[0]) & (priced.date <= period[1]), "date"].unique()
        if not len(days):
            continue
        print(f"{period_name} {period[0]}..{period[1]}: {len(days)} game-days")
        tag = period_name
        setups = {}
        setups["M6 agent, no learning"] = run(f"{tag}-m6-no-learning", tables, period, args.reuse, learn=False, **m6)
        start_nb = Notebook() if tag == "test" else copy.deepcopy(setups_nb["m6"])
        setups["M6 agent + learning"] = run(f"{tag}-m6-learning", tables, period, args.reuse, notebook=start_nb, **m6)
        setups["M6, edge threshold off (diagnostic)"] = run(f"{tag}-m6-threshold-off", tables, period, args.reuse,
                                                            learn=False, min_edge=FORCED_EDGE, **m6)
        setups["Anchor agent, no learning"] = run(f"{tag}-anchor-no-learning", tables, period, args.reuse,
                                                  forecaster=forecaster, learn=False)
        setups["Raw M4 agent (no anchor), no learning"] = run(f"{tag}-raw-m4", tables, period, args.reuse,
                                                               forecaster=forecaster, learn=False, anchor=False)
        if tag == "test":
            existing = RUNS / "results" / "test-full"
            setups["Anchor agent + learning (existing run)"] = (pd.read_parquet(existing / "fills.parquet"),
                                                               Notebook.load(existing / "notebook.json"))
            setups_nb = {"m6": setups["M6 agent + learning"][1],
                         "anchor": setups["Anchor agent + learning (existing run)"][1]}
        else:
            setups["Anchor agent + learning"] = run(f"{tag}-anchor-learning", tables, period, args.reuse,
                                                    notebook=copy.deepcopy(setups_nb["anchor"]), forecaster=forecaster)
        for label, (fills, nb) in setups.items():
            learning = "learning" in label
            rows.append(score(label, period_name, fills, games, days, active_rules(nb) if learning else None))
        rows.append(score("Never trade", period_name, pd.DataFrame(), games, days))

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "m6_ablation.csv", index=False)
    shown = table.assign(**{
        "mean CLV [95% CI]": [("—" if np.isnan(r.mean_clv) else f"{r.mean_clv:+.4f} [{r.mean_clv_lo:+.4f}, {r.mean_clv_hi:+.4f}]")
                              for r in table.itertuples()],
        "CLV $": table.clv_dollars.map(lambda v: f"{v:+.2f}"),
        "P&L [95% CI]": [f"{r.pnl:+.2f} [{r.pnl_lo:+.2f}, {r.pnl_hi:+.2f}]" for r in table.itertuples()],
        "ROI": table.roi.map(lambda v: f"{v:+.1%}"),
        "rules active": table.rules_active.map(lambda v: "" if pd.isna(v) else int(v))})
    cols = ["period", "setup", "trades", "mean CLV [95% CI]", "CLV $", "P&L [95% CI]", "ROI", "rules active"]
    (args.out / "m6_ablation.md").write_text(
        "Day-clustered bootstrap (evaluation/stats.py, 2,000 resamples of game-days). CLV per contract against the "
        "mid at tip; CLV $ = CLV x contracts.\n\n" + markdown(shown[cols]))
    print(shown[cols].to_string(index=False))
    print(f"wrote {args.out / 'm6_ablation.csv'} and .md")


if __name__ == "__main__":
    main()
