"""LLM tool agent vs the deterministic anchor agent vs a plain-LLM baseline vs never trade.

    # stub dry run, no network (the heuristic stub plays anchor + M4 shift through the tool protocol)
    python -m evaluation.llm_agent_eval --window test --backend heuristic --name dryrun
    # prompt freeze on development days, then the pre-registered test run, then the holdout once
    python -m evaluation.llm_agent_eval --window dev --start 2026-01-05 --end 2026-01-11 --name dev-freeze
    python -m evaluation.llm_agent_eval --window test
    python -m evaluation.llm_agent_eval --window holdout
    python -m evaluation.llm_agent_eval --probe 20      # does Gemini remember 2025-26 results?
    python -m evaluation.llm_agent_eval --report-only

Setups (docs/preregistration_llm_agent.md):
    anchor        A: deterministic MarketAgent, market anchor + M4 news shift, no learning
    plain         B: one LLM call with the same information as text (README "ChatGPT baseline")
    tool          C: tool-using LLM agent, no sceptic
    tool_sceptic  D: tool-using LLM agent + priced-in sceptic
    tool_anon     optional: C with TEAM_A/TEAM_B and no date (memorisation check)
    never         never trade

Every setup runs on the same Replay settings and decision times as the
existing test runs (stake $20, caps $50/$100/$300, fills at the ask plus the
Kalshi fee, default M4 trained before 1 Feb, no learning). Days are
independent without learning (caps reset daily, one position per game), so
the replay is split into day chunks run in parallel processes; results are
identical to one sequential replay. LLM responses are cached under
runs/llm_cache/, so re-runs are free and identical.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from agents.llm_client import DEFAULT_MODEL
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "runs" / "llm_agent"
WINDOWS = {"dev": ("2025-11-01", "2026-01-31"), "test": ("2026-02-01", "2026-04-12"),
           "holdout": ("2026-04-13", "2026-06-14")}
SETUPS = {"anchor": "A. Deterministic anchor agent (no learning)", "plain": "B. Plain LLM, one call (no tools)",
          "tool": "C. Tool agent, no sceptic", "tool_sceptic": "D. Tool agent + priced-in sceptic",
          "tool_anon": "C-anon. Tool agent, anonymised teams and date", "never": "Never trade"}
LLM_SETUPS = ("plain", "tool", "tool_sceptic", "tool_anon")
CHUNK_DAYS = 2

_STATE = {}
FALLBACK_FROZEN = Path("/tmp/nba_frozen")      # local copy when iCloud evicts data/frozen/prices.parquet


def load_frozen() -> dict:
    from data_sources import FROZEN
    from replay import load_tables
    use_fallback = not (FROZEN / "prices.parquet").exists() and (FALLBACK_FROZEN / "prices.parquet").exists()
    return load_tables(FALLBACK_FROZEN if use_fallback else FROZEN)


# ---------------- runs ----------------

def sampled(policy, frac: float, seed: int):
    """Keep a fixed pseudo-random share of decision points, identical for every setup."""
    if frac >= 1:
        return policy

    def wrapped(view, game, now):
        h = hashlib.sha256(f"{seed}|{game.game_id}|{pd.Timestamp(now).isoformat()}".encode()).hexdigest()
        return policy(view, game, now) if int(h[:12], 16) / 16 ** 12 < frac else []
    return wrapped


def _shared():
    if not _STATE:
        from forecast.api import Forecaster
        _STATE["tables"] = load_frozen()
        _STATE["forecaster"] = Forecaster.load()
        try:
            from forecast.impact import load_impact
            _STATE["impact"] = load_impact()
        except Exception:
            _STATE["impact"] = None
        path = ROOT / "data" / "frozen" / "players.parquet"
        players = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player_id", "player_name"])
        _STATE["players"] = dict(zip(players.player_id.astype(int), players.player_name))
    return _STATE


def make_backend(kind: str, model: str | None, min_interval: float):
    from agents.llm_client import GeminiBackend, StubBackend
    from agents.tool_agent import heuristic_reply
    if kind == "heuristic":
        return StubBackend(heuristic_reply, model="heuristic-stub")
    return GeminiBackend(model or DEFAULT_MODEL, min_interval=min_interval)


def make_agent(setup: str, backend, s: dict):
    from agents.graph import MarketAgent
    from agents.llm_client import CachedLLM
    from agents.tool_agent import PlainLLMAgent, ToolAgent
    if setup == "anchor":
        return MarketAgent(forecaster=s["forecaster"], learn=False)
    llm = CachedLLM(backend)
    kw = {"forecaster": s["forecaster"], "impact": s["impact"], "players": s["players"]}
    if setup == "plain":
        return PlainLLMAgent(llm, **kw)
    return ToolAgent(llm, sceptic=setup == "tool_sceptic", anonymise=setup == "tool_anon", **kw)


def _run_chunk(job: dict) -> dict:
    from replay import Replay
    s = _shared()
    backend = None if job["setup"] == "anchor" else make_backend(job["backend"], job["model"], job["min_interval"])
    agent = make_agent(job["setup"], backend, s)
    t0 = time.time()
    decisions, fills = Replay(s["tables"], sampled(agent.policy, job["frac"], job["seed"])).run(job["start"], job["end"])
    for frame in (decisions, fills):
        if len(frame):
            frame["decision_id"] = job["chunk"] + ":" + frame.decision_id.astype(str)
    return {"decisions": decisions, "fills": fills, "traces": agent.traces, "seconds": time.time() - t0}


def chunks(tables, start, end, size=CHUNK_DAYS) -> list:
    from evaluation.walkforward import market_days
    days = market_days(tables, start, end)
    return [(days[i], days[min(i + size, len(days)) - 1]) for i in range(0, len(days), size)]


def run_setup(setup, tag, start, end, args) -> Path:
    tables = _shared()["tables"] if _STATE else load_frozen()
    jobs = [{"setup": setup, "start": a, "end": b, "chunk": f"{a}", "backend": args.backend, "model": args.model,
             "frac": args.subsample, "seed": args.seed, "min_interval": args.min_interval}
            for a, b in chunks(tables, start, end)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(_run_chunk, jobs))
    decisions = pd.concat([p["decisions"] for p in parts], ignore_index=True)
    fills = pd.concat([p["fills"] for p in parts if len(p["fills"])], ignore_index=True) if any(
        len(p["fills"]) for p in parts) else pd.DataFrame()
    traces = [t for p in parts for t in p["traces"]]
    out = OUT / tag / setup
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(out / "decisions.parquet", index=False)
    fills.to_parquet(out / "fills.parquet", index=False)
    (out / "trace.jsonl").write_text("\n".join(json.dumps(t, default=str) for t in traces))
    meta = {"setup": setup, "start": start, "end": end, "backend": args.backend, "model": args.model or
            (DEFAULT_MODEL if args.backend == "gemini" else "heuristic-stub"), "subsample": args.subsample,
            "seed": args.seed, "wall_seconds": time.time() - t0, "fills": len(fills)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {setup}: {len(fills)} fills, {len(traces)} decision points in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return out


# ---------------- diagnostics ----------------

def load_traces(folder: Path) -> list:
    path = folder / "trace.jsonl"
    if not path.exists() or not path.read_text().strip():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close_clv(tables, price_index, ticker, side, price, game_id):
    """Counterfactual CLV of an order the sceptic vetoed: side's mid at tip minus the price it would have paid."""
    from replay import AsOf
    tip = tables["games"].set_index("game_id").tip_time[game_id]
    q = AsOf(tables, tip, price_index).quote(ticker)
    if q is None:
        return np.nan
    mid = (q.bid + q.ask) / 2
    return float((mid if side == "yes" else 1 - mid) - price)


def diagnostics(setup: str, traces: list, fills: pd.DataFrame, anchor_games: set, tables, price_index) -> dict:
    live = [t for t in traces if t["status"] not in ("held", "idle")]
    asked = [t for t in live if t.get("llm", {}).get("calls", 0) > 0]
    tool_calls = [len(t.get("tool_calls", [])) for t in asked]
    invalid = [t for t in live if t["status"] == "invalid"]
    by_reason = pd.Series([t["reason"] for t in invalid]).value_counts().to_dict() if invalid else {}
    tools_used = pd.Series([c["tool"] for t in asked for c in t.get("tool_calls", [])]).value_counts().to_dict()
    proposals = [t for t in live if (t.get("proposal") or {}).get("decision", "").lower() == "buy"
                 or (setup.startswith("plain") and t["status"] in ("order", "gap", "risk_block"))]
    sceptic = [t for t in live if "sceptic" in t]
    rejected = [t for t in sceptic if t["sceptic"]["verdict"] != "approve"]
    veto_clv = [close_clv(tables, price_index, t["candidate"]["ticker"], t["candidate"]["side"],
                          t["candidate"]["price"], t["game_id"]) for t in rejected if "candidate" in t]
    llm = [t.get("llm", {}) for t in traces]
    calls = sum(x.get("calls", 0) for x in llm)
    added = (~fills.game_id.isin(anchor_games)).mean() if len(fills) else np.nan
    return {"setup": setup, "decision_points": len(traces), "considered": len(live), "llm_decisions": len(asked),
            "tool_calls_per_decision": float(np.mean(tool_calls)) if tool_calls else 0.0,
            "turns_per_decision": float(np.mean([t.get("turns", 0) for t in asked])) if asked else 0.0,
            "buy_proposals": len(proposals), "invalid": len(invalid),
            "invalid_rate": len(invalid) / len(asked) if asked else np.nan, "invalid_by_reason": by_reason,
            "llm_errors": sum(t["status"] == "llm_error" for t in live),
            "gap_fail": sum(t["status"] == "gap" for t in live), "agent_pass": sum(t["status"] == "pass" for t in live),
            "sceptic_calls": len(sceptic), "sceptic_rejects": len(rejected),
            "sceptic_reject_rate": len(rejected) / len(sceptic) if sceptic else np.nan,
            "sceptic_invalid": sum(bool(t["sceptic"].get("invalid")) for t in sceptic),
            "vetoed_mean_clv": float(np.nanmean(veto_clv)) if len(veto_clv) and not np.all(np.isnan(veto_clv)) else np.nan,
            "kept_mean_clv": float(fills.clv.mean()) if len(fills) and sceptic else np.nan,
            "orders": sum(t["status"] == "order" for t in live), "fills": len(fills),
            "share_trades_not_in_anchor_games": float(added) if not pd.isna(added) else np.nan,
            "tools_used": tools_used, "llm_calls": calls, "cache_hits": sum(x.get("cached", 0) for x in llm),
            "tokens_in": sum(x.get("tokens_in", 0) for x in llm), "tokens_out": sum(x.get("tokens_out", 0) for x in llm),
            "cost_usd": sum(x.get("cost", 0.0) for x in llm),
            "latency_per_call_s": sum(x.get("latency", 0.0) for x in llm) / calls if calls else np.nan,
            "latency_per_llm_decision_s": (sum(x.get("latency", 0.0) for x in llm) / len(asked)) if asked else np.nan}


# ---------------- report ----------------

def fmt(est, lo, hi, kind):
    if pd.isna(est):
        return "–"
    if kind == "clv":
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"


def report_window(tag: str, start: str, end: str, tables, price_index) -> tuple:
    from evaluation.walkforward import load_fills, market_days
    games = tables["games"]
    days = market_days(tables, start, end)
    base = OUT / tag
    present = [s for s in SETUPS if s != "never" and (base / s / "fills.parquet").exists()]
    if not present:
        return None, None, None, None
    fills = {s: load_fills(base / s) for s in present}
    fills = {s: (f[f.game_id.isin(games.loc[games.date.isin(days), "game_id"])] if len(f) else f)
             for s, f in fills.items()}
    tabs = {s: day_table(f, games, days) for s, f in fills.items()}
    tabs["never"] = day_table(pd.DataFrame(), games, days)
    rows, pairs, diags = [], [], []
    anchor_games = set(fills["anchor"].game_id) if "anchor" in fills and len(fills["anchor"]) else set()
    for s, t in tabs.items():
        b = bootstrap(t)
        row = {"window": tag, "setup": s, "label": SETUPS[s], "days": len(t), "trades": int(t.trades.sum())}
        for m in b.index:
            row.update({m: b.loc[m, "estimate"], f"{m}_lo": b.loc[m, "ci_low"], f"{m}_hi": b.loc[m, "ci_high"]})
        rows.append(row)
        for ref in ("anchor", "never"):
            if ref == s or ref not in tabs or s == "never":
                continue
            d = paired(t, tabs[ref])
            for m in ("mean_clv", "clv_dollars", "pnl"):
                if ref == "never" and m == "mean_clv":
                    continue
                r = d.loc[m]
                pairs.append({"window": tag, "a": s, "b": ref, "metric": m, "estimate": r.estimate,
                              "ci_low": r.ci_low, "ci_high": r.ci_high, "p_a_gt_b": r.p_a_gt_b})
        if s in LLM_SETUPS:
            diags.append(diagnostics(s, load_traces(base / s), fills[s], anchor_games, tables, price_index))
    if "tool" in tabs and "tool_sceptic" in tabs:
        d = paired(tabs["tool_sceptic"], tabs["tool"])
        for m in ("mean_clv", "clv_dollars", "pnl"):
            r = d.loc[m]
            pairs.append({"window": tag, "a": "tool_sceptic", "b": "tool", "metric": m, "estimate": r.estimate,
                          "ci_low": r.ci_low, "ci_high": r.ci_high, "p_a_gt_b": r.p_a_gt_b})
    meta = {s: json.loads((base / s / "meta.json").read_text()) for s in present if (base / s / "meta.json").exists()}
    diags = pd.DataFrame(diags)
    if len(diags):
        from agents.llm_client import cost_usd
        models = diags.setup.map(lambda s: str(meta.get(s, {}).get("model", "")))
        diags["cost_usd"] = [cost_usd(m, i, o) for m, i, o in zip(models, diags.tokens_in, diags.tokens_out)]
    return pd.DataFrame(rows), pd.DataFrame(pairs), diags, {"tabs": tabs, "meta": meta, "days": days}


def chart(tabs_by_window: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"anchor": "#1f77b4", "plain": "#9467bd", "tool": "#ff7f0e", "tool_sceptic": "#2ca02c",
              "tool_anon": "#8c564b", "never": "#7f7f7f"}
    windows = list(tabs_by_window)
    fig, axes = plt.subplots(2, len(windows), figsize=(7 * len(windows), 7.5), squeeze=False)
    for j, w in enumerate(windows):
        for i, (col, title) in enumerate((("pnl", "cumulative P&L after fees ($)"), ("clv_dollars", "cumulative CLV ($)"))):
            ax = axes[i, j]
            for s, t in tabs_by_window[w].items():
                x = pd.to_datetime(t.index)
                ax.plot(x, t[col].cumsum(), label=f"{s} ({int(t.trades.sum())} trades)", color=colors.get(s),
                        lw=2 if s in ("tool_sceptic", "anchor") else 1.3, ls="--" if s == "never" else "-")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{w}: {title}")
            ax.tick_params(axis="x", rotation=30)
        axes[0, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(windows: list, probe: dict | None = None, stem: str = "llm_agent"):
    tables = load_frozen()
    prices = tables["prices"].sort_values("ts")
    price_index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}
    all_rows, all_pairs, all_diags, tabs_by_window, metas = [], [], [], {}, {}
    for tag in windows:
        start, end = WINDOWS.get(tag, (None, None))
        if start is None:
            meta = next((json.loads(p.read_text()) for p in (OUT / tag).glob("*/meta.json")), None)
            if meta is None:
                continue
            start, end = meta["start"], meta["end"]
        rows, pairs, diags, extra = report_window(tag, start, end, tables, price_index)
        if rows is None:
            continue
        all_rows.append(rows)
        all_pairs.append(pairs)
        all_diags.append(diags.assign(window=tag))
        tabs_by_window[tag] = extra["tabs"]
        metas[tag] = extra
    if not all_rows:
        print("no runs to report")
        return
    rows, pairs, diags = pd.concat(all_rows), pd.concat(all_pairs), pd.concat(all_diags)
    out = pd.concat([rows.assign(kind="setup"), pairs.assign(kind="paired"),
                     diags.assign(kind="diagnostics").astype({"invalid_by_reason": str, "tools_used": str})])
    out.to_csv(RESULTS / f"{stem}.csv", index=False)
    chart(tabs_by_window, RESULTS / f"{stem}.png")
    md = ["# LLM tool agent vs deterministic agent vs plain LLM vs never trade", "",
          "Pre-registration: `docs/preregistration_llm_agent.md`. Same replay as the existing test runs: stake $20, "
          "caps $50/$100/$300, fills at the ask (or 1 − bid) plus the Kalshi fee, capped by volume; default M4 "
          "(trained before 1 Feb); no learning. CLV is per contract against the mid at tip. 95% CIs and one-sided "
          f"p-values from a day-clustered bootstrap ({REPS} replicates, seed {SEED}).", ""]
    for tag, extra in metas.items():
        r = rows[rows.window == tag]
        m = extra["meta"]
        model = next((v["model"] for k, v in m.items() if k in LLM_SETUPS), "–")
        sub = next((v["subsample"] for v in m.values()), 1.0)
        md += [f"## {tag}: {extra['days'][0]} – {extra['days'][-1]} ({len(extra['days'])} game-days)", ""]
        if model == "heuristic-stub":
            md += ["**Stub results, not Gemini.** Every LLM call was answered by the deterministic heuristic stub "
                   "(`agents.tool_agent.heuristic_reply`: anchor + M4 news shift through the tool protocol; the "
                   "sceptic rejects when the price already moved 3¢ toward the trade). This proves the pipeline "
                   "end to end; it says nothing about how an LLM would trade. Tokens are character-count estimates "
                   "and latency is local.", ""]
        if model.startswith("gemini") and model != "gemini-2.5-flash":
            md += [f"**Model deviation from the pre-registration.** The pre-registered `gemini-2.5-flash` is no "
                   f"longer available to new API keys (404), and the free-tier quota for `gemini-3.8-flash` is "
                   f"20 requests per day. All LLM arms therefore used `{model}` (free tier, about 15 requests per "
                   f"minute). Prompts, tools, validation and the trading rule are unchanged.", ""]
        md += [
               f"LLM: `{model}`, temperature 0. Decision points: "
               + ("all" if sub >= 1 else f"fixed {sub:.0%} subsample (seed {next(iter(m.values()))['seed']}), same for every setup")
               + ".", "",
               "| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | p (CLV > 0) |",
               "| --- | --- | --- | --- | --- | --- |"]
        for x in r.itertuples():
            b = bootstrap(extra["tabs"][x.setup]).loc["mean_clv", "p_gt_0"]
            md.append(f"| {x.label} | {x.trades} | {fmt(x.mean_clv, x.mean_clv_lo, x.mean_clv_hi, 'clv')} | "
                      f"{fmt(x.clv_dollars, x.clv_dollars_lo, x.clv_dollars_hi, '$')} | "
                      f"{fmt(x.pnl, x.pnl_lo, x.pnl_hi, '$')} | {'–' if pd.isna(b) else f'{b:.3f}'} |")
        p = pairs[pairs.window == tag]
        md += ["", "Paired differences on the same game-days (first minus second; p is one-sided for first > second):", "",
               "| Comparison | Metric | Difference [95% CI] | p |", "| --- | --- | --- | --- |"]
        for x in p.itertuples():
            name = {"mean_clv": "mean CLV", "clv_dollars": "CLV $", "pnl": "P&L"}[x.metric]
            md.append(f"| {x.a} − {x.b} | {name} | {fmt(x.estimate, x.ci_low, x.ci_high, 'clv' if x.metric == 'mean_clv' else '$')} "
                      f"| {'–' if pd.isna(x.p_a_gt_b) else f'{x.p_a_gt_b:.3f}'} |")
        d = diags[diags.window == tag]
        if len(d):
            md += ["", "Agent diagnostics:", "",
                   "| Setup | Decision points (LLM asked) | Tool calls / decision | Buy proposals | Invalid (rate) | "
                   "Gap fails | Sceptic rejects / calls | Vetoed vs kept CLV | Trades outside A's games | LLM calls "
                   "(cache hits) | Tokens in / out | Cost | Latency / call |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for x in d.itertuples():
                rej = "–" if not x.sceptic_calls else f"{x.sceptic_rejects} / {x.sceptic_calls} ({x.sceptic_reject_rate:.0%})"
                veto = "–" if pd.isna(x.vetoed_mean_clv) else f"{x.vetoed_mean_clv:+.4f} vs {x.kept_mean_clv:+.4f}"
                share = "–" if pd.isna(x.share_trades_not_in_anchor_games) else f"{x.share_trades_not_in_anchor_games:.0%}"
                md.append(f"| {x.setup} | {x.considered} ({x.llm_decisions}) | {x.tool_calls_per_decision:.1f} | "
                          f"{x.buy_proposals} | {x.invalid} ({x.invalid_rate:.1%}) | {x.gap_fail} | {rej} | {veto} | "
                          f"{share} | {x.llm_calls:,} ({x.cache_hits:,}) | {x.tokens_in:,} / {x.tokens_out:,} | "
                          f"${x.cost_usd:.2f} | {x.latency_per_call_s:.2f}s |")
            md += ["", "Invalid outputs by reason: " + "; ".join(f"{x.setup}: {x.invalid_by_reason or 'none'}"
                                                             for x in d.itertuples()),
                   "", "Tool use: " + "; ".join(f"{x.setup}: {x.tools_used}" for x in d.itertuples() if x.tools_used)]
        md.append("")
    if probe:
        md += ["## Memorisation probe", "", probe["summary"], ""]
    md += [f"![Cumulative P&L and CLV by setup]({stem}.png)", "",
           "Costs are estimates from token counts at the list prices in `agents/llm_client.PRICES` "
           "(Gemini 3.5 Flash-Lite assumed at the 2.5 Flash-Lite price, $0.10 / $0.40 per million input / output "
           "tokens; the runs themselves used the free tier) and include calls served from the cache, i.e. the cost "
           "of a fresh run. Latency is per API call as measured when the response was first fetched."]
    (RESULTS / f"{stem}.md").write_text("\n".join(md) + "\n")
    print((RESULTS / f"{stem}.md").read_text())


# ---------------- memorisation probe ----------------

PROBE_SYSTEM = ("You answer questions about past NBA games from memory. If you do not remember the game, say so. "
                "Reply with one JSON object: {\"winner\": team abbreviation or \"unknown\", \"confidence\": 0 to 1}.")


def probe(n: int, args) -> dict:
    from agents.llm_client import CachedLLM
    tables = load_frozen()
    g = tables["games"]
    start, end = WINDOWS["test"]
    g = g[(g.date >= start) & (g.date <= end) & g.game_id.isin(tables["markets"].game_id)].dropna(subset=["home_pts"])
    g = g.sample(n=min(n, len(g)), random_state=args.seed)
    llm = CachedLLM(make_backend(args.backend, args.model, args.min_interval))
    rows = []
    for x in g.itertuples():
        q = f"Who won the NBA game {x.away_team} at {x.home_team} on {x.date}?"
        r = llm.ask("probe", PROBE_SYSTEM, q, {"game": x._asdict()})
        winner = str((r["json"] or {}).get("winner", "unknown")).upper()
        truth = x.home_team if x.home_pts > x.away_pts else x.away_team
        rows.append({"game_id": x.game_id, "answered": winner not in ("UNKNOWN", ""), "correct": winner == truth,
                     "home_won": x.home_pts > x.away_pts})
    df = pd.DataFrame(rows)
    ans = df[df.answered]
    summary = (f"{len(df)} randomly drawn test-period games (seed {args.seed}), asked \"Who won AWAY at HOME on DATE?\" "
               f"with an explicit \"unknown\" option. Gemini answered {len(ans)} of {len(df)}; "
               f"{int(ans.correct.sum())} of those were right ({ans.correct.mean():.0%} accuracy among answered, "
               f"{df.correct.mean():.0%} of all questions). Always picking the home team would score "
               f"{df.home_won.mean():.0%}. Accuracy near the home-team or favourite rate means no evidence of recall.")
    (OUT / "probe.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(summary)
    return {"summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS), default="test")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--name", help="run folder under runs/llm_agent/ (default: the window)")
    ap.add_argument("--setups", default="anchor,plain,tool,tool_sceptic")
    ap.add_argument("--backend", choices=["gemini", "heuristic"], default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between API calls per process")
    ap.add_argument("--subsample", type=float, default=1.0, help="share of decision points kept (fixed seed)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report", default="test,holdout", help="run folders to put in the report")
    ap.add_argument("--stem", default="llm_agent", help="results file name in evaluation/results/")
    ap.add_argument("--probe", type=int, default=0, help="ask about N test-period results, then exit")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe(args.probe, args)
        return
    if not args.report_only:
        start, end = args.start or WINDOWS[args.window][0], args.end or WINDOWS[args.window][1]
        tag = args.name or args.window
        print(f"{tag}: {start} .. {end}, backend {args.backend}, subsample {args.subsample}")
        for setup in args.setups.split(","):
            run_setup(setup.strip(), tag, start, end, args)
    p = OUT / "probe.json"
    write_report([w.strip() for w in args.report.split(",")], json.loads(p.read_text()) if p.exists() else None,
                 args.stem)


if __name__ == "__main__":
    main()
