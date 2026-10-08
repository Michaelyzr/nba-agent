"""Paired model comparison on the same decision points: deepseek-reasoner minus deepseek-chat, per setup.

    python -m evaluation.llm_model_compare --a test-deepseek-reasoner --b test-deepseek
"""
import argparse

import pandas as pd

from evaluation.llm_agent_eval import OUT, WINDOWS, fmt, load_frozen
from evaluation.stats import bootstrap, day_table, paired


def compare(a: str, b: str, setups=("plain", "tool", "tool_sceptic")) -> list:
    from evaluation.walkforward import load_fills, market_days
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *WINDOWS["test"])
    md = [f"| Setup | Trades ({a} / {b}) | Metric | {a} − {b} [95% CI] | p |", "| --- | --- | --- | --- | --- |"]
    for s in setups:
        if not ((OUT / a / s / "fills.parquet").exists() and (OUT / b / s / "fills.parquet").exists()):
            continue
        ta, tb = (day_table(load_fills(OUT / r / s), games, days) for r in (a, b))
        d = paired(ta, tb)
        for m, name in (("mean_clv", "mean CLV"), ("clv_dollars", "CLV $"), ("pnl", "P&L")):
            r = d.loc[m]
            md.append(f"| {s} | {int(ta.trades.sum())} / {int(tb.trades.sum())} | {name} | "
                      f"{fmt(r.estimate, r.ci_low, r.ci_high, 'clv' if m == 'mean_clv' else '$')} | "
                      f"{'–' if pd.isna(r.p_a_gt_b) else f'{r.p_a_gt_b:.3f}'} |")
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="test-deepseek-reasoner")
    ap.add_argument("--b", default="test-deepseek")
    args = ap.parse_args()
    print("\n".join(compare(args.a, args.b)))
