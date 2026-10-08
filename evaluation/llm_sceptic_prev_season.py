"""Arm D (reasoner tool agent + sceptic) vs arm A (anchor) on an extra, earlier window, and pooled with the test period.

No 2024-25 game prices exist (Kalshi coverage starts 20 Oct 2025), so the extra window is 1 Dec 2025 - 31 Jan 2026,
replayed with walk-forward M4/M6 fit only on data before 1 Dec (evaluation.walkforward_models):

    python -m evaluation.walkforward_models --split 2025-12-01 --val-from 2025-11-20
    NBA_MODEL_DIR=runs/walkforward/2025-12-01 python -m evaluation.llm_agent_eval --start 2025-12-01 \
        --end 2026-01-31 --name prev-anchor --setups anchor --backend deepseek --model deepseek-reasoner \
        --min-edge 0.04 --chunk-days 1          # and --name prev-tool_sceptic --setups tool_sceptic
    python -m evaluation.llm_sceptic_prev_season [--balance-before X --balance-after Y]

The test-period arms are the ones in llm_sceptic_full (finished full-sample runs, same settings, production models).
"""
import argparse
import json

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, RESULTS, WINDOWS, close_clv, load_frozen, load_traces
from evaluation.llm_sceptic_full import ANCHOR, MODEL, from_runs, summary
from evaluation.stats import REPS, SEED, day_table, paired

PREV = ("2025-12-01", "2026-01-31")
PREV_A = OUT / "prev-anchor" / "anchor"
PREV_D = OUT / "prev-tool_sceptic" / "tool_sceptic"
STEM = "llm_sceptic_prev_season"


def sceptic_stats(tables, pidx, traces, fills) -> dict:
    live = [t for t in traces if t["status"] not in ("held", "idle")]
    sc = [t for t in live if "sceptic" in t]
    vetoed = [t for t in sc if t["sceptic"]["verdict"] != "approve"]
    veto_clv = [close_clv(tables, pidx, t["candidate"]["ticker"], t["candidate"]["side"], t["candidate"]["price"],
                          t["game_id"]) for t in vetoed if "candidate" in t]
    return {"reviewed": len(sc), "approved": len(sc) - len(vetoed), "vetoed": len(vetoed),
            "vetoed_mean_clv": float(np.nanmean(veto_clv)) if veto_clv else np.nan,
            "kept_mean_clv": float(fills.clv.mean()) if len(fills) else np.nan}


def window_arms(tables, pidx, days, a_dir, d_parts) -> dict:
    games = tables["games"]
    day_of = games.set_index("game_id").date.astype(str)
    a_f = pd.read_parquet(a_dir / "fills.parquet")
    a_f = a_f[a_f.game_id.map(day_of).isin(days)]
    n_pts = sum(day_of.get(t["game_id"]) in days for t in load_traces(a_dir))
    d_f, d_tr = d_parts
    ta, td = day_table(a_f, games, days), day_table(d_f, games, days)
    out = {}
    for arm, t, f in (("A", ta, a_f), ("D", td, d_f)):
        s = summary(t, f)
        s["win_rate"] = float((f.pnl > 0).mean()) if len(f) else np.nan
        out[arm] = {"days": len(days), "decision_points": n_pts, **s}
    out["D"].update(sceptic_stats(tables, pidx, d_tr, d_f))
    return {"rows": out, "tables": (ta, td), "fills": (a_f, d_f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance-before", type=float)
    ap.add_argument("--balance-after", type=float)
    args = ap.parse_args()
    from evaluation.walkforward import market_days
    tables = load_frozen()
    games = tables["games"]
    prices = tables["prices"].sort_values("ts")
    pidx = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}

    meta = json.loads((PREV_D / "meta.json").read_text())
    assert (meta["start"], meta["end"], meta["min_edge"], meta["subsample"]) == (*PREV, 0.04, 1.0), meta
    prev_days = market_days(tables, *PREV)
    prev = window_arms(tables, pidx, prev_days, PREV_A,
                       (pd.read_parquet(PREV_D / "fills.parquet"), load_traces(PREV_D)))

    test_days = market_days(tables, *WINDOWS["test"])
    parts = {p["day"]: p for p in from_runs(games, test_days)}
    missing = sorted(set(test_days) - set(parts))
    if missing:
        raise SystemExit(f"test-period D runs do not cover {len(missing)} days, e.g. {missing[:3]}")
    fl = [p["fills"] for p in parts.values() if len(p["fills"])]
    test = window_arms(tables, pidx, test_days, ANCHOR,
                       (pd.concat(fl, ignore_index=True), [t for p in parts.values() for t in p["traces"]]))

    pooled_a = pd.concat([prev["tables"][0], test["tables"][0]]).sort_index()
    pooled_d = pd.concat([prev["tables"][1], test["tables"][1]]).sort_index()
    pa_f = pd.concat([prev["fills"][0], test["fills"][0]], ignore_index=True)
    pd_f = pd.concat([prev["fills"][1], test["fills"][1]], ignore_index=True)
    rows = []
    for name, w in (("Dec 2025 - Jan 2026 (walk-forward)", prev), ("Test, Feb - Apr 2026", test)):
        for arm, r in w["rows"].items():
            rows.append({"window": name, "arm": arm, **r})
    for arm, t, f in (("A", pooled_a, pa_f), ("D", pooled_d, pd_f)):
        s = summary(t, f)
        s["win_rate"] = float((f.pnl > 0).mean()) if len(f) else np.nan
        extra = {k: prev["rows"]["D"][k] + test["rows"]["D"][k] for k in ("reviewed", "approved", "vetoed")} \
            if arm == "D" else {}
        rows.append({"window": "Pooled", "arm": arm, "days": len(t),
                     "decision_points": prev["rows"]["A"]["decision_points"] + test["rows"]["A"]["decision_points"],
                     **s, **extra})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"{STEM}.csv", index=False)

    g = games.set_index("game_id")
    day_of = g.date.astype(str)
    tr = []
    for arm, f in (("A", prev["fills"][0]), ("D", prev["fills"][1])):
        if len(f):
            tr.append(f.assign(arm=arm, date=f.game_id.map(day_of),
                               game=f.game_id.map(lambda i: f"{g.away_team[i]} @ {g.home_team[i]}"),
                               team=f.market_ticker.str.split("-").str[-1],
                               staked=f.price * f.contracts + f.fee, clv_dollars=f.clv * f.contracts))
    trades = pd.concat(tr, ignore_index=True) if tr else pd.DataFrame()
    cols = ["arm", "date", "game", "team", "side", "price", "contracts", "staked", "outcome", "pnl", "clv", "clv_dollars"]
    if len(trades):
        trades[cols].sort_values(["arm", "date"]).to_csv(RESULTS / f"{STEM}_trades.csv", index=False)

    diffs = {"Dec 2025 - Jan 2026": paired(prev["tables"][1], prev["tables"][0]),
             "Test, Feb - Apr 2026": paired(test["tables"][1], test["tables"][0]),
             "Pooled": paired(pooled_d, pooled_a)}

    f0 = lambda v: f"{v:+,.0f}"
    md = ["# Arm D (reasoner tool agent + sceptic) vs arm A (anchor): extra window 1 Dec 2025 - 31 Jan 2026, and pooled",
          "", "**Why this window.** There are no recorded game prices for 2024-25 (Kalshi coverage in "
          "`data/frozen/prices.parquet` starts 20 Oct 2025; the Polymarket/venue tables cover only the test period), so "
          "the agent cannot trade that season. The extra window is the earlier part of 2025-26 instead: "
          f"{len(prev_days)} game-days, 1 Dec 2025 - 31 Jan 2026 (October is skipped: prices start mid-month and "
          "there is little injury news).", "",
          "**Leakage handling.** The production M4 and M6 are trained on data up to 1 Feb / 15 Jan 2026, which covers "
          "this window, so they are not used here. M4 is refit on the 2,936 games tipping before 1 Dec 2025 and M6 on "
          "Kalshi-era rows before 20 Nov 2025 (early stopping on 20-30 Nov), with the refit M4 feeding M6's news "
          "features (`evaluation/walkforward_models.py`, loaded through `NBA_MODEL_DIR`). The LLM prompts, the 4¢ "
          "minimum edge and the sceptic were fixed before this run; the only earlier contact with the window is a "
          "two-day Gemini prompt check on 5-6 Jan 2026.", "",
          f"Same replay settings as the full test run: `{MODEL}`, every decision point (no subsampling), $20 stake, "
          "caps $50/$100/$300, fills at the ask plus the Kalshi fee, min edge 4¢, no learning. Staked includes fees; "
          "a win is a trade with positive P&L. 95% CIs from a day-clustered bootstrap "
          f"({REPS} replicates, seed {SEED}); the pooled bootstrap resamples days from both windows.", "",
          "| Window | Arm | Days | Decision points | Trades | Staked | P&L [95% CI] | Return on staked | "
          "Return on $1,000 | CLV $ [95% CI] | Win rate |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in table.itertuples():
        if r.trades:
            md.append(f"| {r.window} | {r.arm} | {r.days} | {r.decision_points} | {r.trades} | ${r.staked:,.0f} | "
                      f"{f0(r.pnl)} [{f0(r.pnl_lo)}, {f0(r.pnl_hi)}] | {r.ret_staked:+.1%} | {r.ret_1000:+.1%} | "
                      f"{f0(r.clv_dollars)} [{f0(r.clv_lo)}, {f0(r.clv_hi)}] | {r.win_rate:.0%} |")
        else:
            md.append(f"| {r.window} | {r.arm} | {r.days} | {r.decision_points} | 0 | $0 | +0 | – | +0.0% | +0 | – |")
    md += ["", "Sceptic (arm D):", "", "| Window | Reviewed | Approved | Vetoed | Mean CLV / contract, vetoed | "
           "Mean CLV / contract, kept |", "| --- | --- | --- | --- | --- | --- |"]
    for name, w in (("Dec 2025 - Jan 2026", prev), ("Test, Feb - Apr 2026", test)):
        d = w["rows"]["D"]
        md.append(f"| {name} | {d['reviewed']} | {d['approved']} | {d['vetoed']} | {d['vetoed_mean_clv']:+.4f} | "
                  f"{d['kept_mean_clv']:+.4f} |")
    md += ["", "Paired D − A on the same days (day-clustered bootstrap, one-sided p for D > A):", "",
           "| Window | P&L difference [95% CI] | p | CLV $ difference [95% CI] | p |", "| --- | --- | --- | --- | --- |"]
    pv = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
    for name, d in diffs.items():
        p, c = d.loc["pnl"], d.loc["clv_dollars"]
        md.append(f"| {name} | {p.estimate:+,.2f} [{p.ci_low:+,.2f}, {p.ci_high:+,.2f}] | {pv(p.p_a_gt_b)} | "
                  f"{c.estimate:+,.2f} [{c.ci_low:+,.2f}, {c.ci_high:+,.2f}] | {pv(c.p_a_gt_b)} |")
    kept = trades[trades.arm == "D"] if len(trades) else trades
    md += ["", "Kept D trades, Dec 2025 - Jan 2026:", "",
           "| Date | Game | Bought | Price | Won | P&L | CLV / contract | CLV $ |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in kept.itertuples():
        md.append(f"| {x.date} | {x.game} | {x.team} {x.side} | {x.price:.2f} | {'won' if x.pnl > 0 else 'lost'} "
                  f"| {x.pnl:+.2f} | {x.clv:+.3f} | {x.clv_dollars:+.2f} |")
    if args.balance_before is not None and args.balance_after is not None:
        md += ["", f"DeepSeek spend for this window (account balance before minus after): "
                   f"${args.balance_before - args.balance_after:.2f} (${args.balance_before:.2f} → "
                   f"${args.balance_after:.2f})."]
    md.append("")
    (RESULTS / f"{STEM}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
