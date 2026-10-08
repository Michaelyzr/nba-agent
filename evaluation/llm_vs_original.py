"""What the DeepSeek LLM agent changes, decision by decision, against the original deterministic agent (A).

    python -m evaluation.llm_vs_original            # reads runs/llm_agent/test-deepseek{,-reasoner}/, no API calls

Every arm ran on the same 304 test-period decision points (fixed 40% subsample, seed 7606), so decisions are
matched on (game_id, as_of). Sections:
    1. overlap with A: both trade (same / opposite side), A only (what the LLM filtered), LLM only (what it added);
       kept vs filtered CLV with a day-clustered bootstrap, and a random-filter baseline (same number of A's trades)
    2. probability estimates: plain LLM (B) p_home at every point vs A, M4, the market mid and the closing mid
    3. sceptic veto reasons by category, with the counterfactual CLV of the vetoed trades
    4. quality and guardrails: validation failures, tool use, citation validity, the funnel to fills
    5. side-by-side reasons on the same decision
    6. cost and latency per decision and per trade
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
MODELS = {"chat": ("test-deepseek", "deepseek-chat"), "reasoner": ("test-deepseek-reasoner", "deepseek-reasoner")}
ARMS = {"plain": "B plain", "tool": "C tools", "tool_sceptic": "D tools+sceptic"}
RANDOM_DRAWS = 20000
EPS = 0.01


def key(game_id, as_of) -> tuple:
    return str(game_id), pd.Timestamp(as_of).isoformat()


def load_traces(folder: Path) -> list:
    path = folder / "trace.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def load_fills(folder: Path, games: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_parquet(folder / "fills.parquet")
    if f.empty:
        return pd.DataFrame(columns=["key", "date", "team", "home_dir", "clv", "clv_usd", "pnl", "win", "price"])
    g = games.set_index("game_id")
    f["key"] = [key(a, b) for a, b in zip(f.game_id, f.as_of)]
    f["date"] = f.game_id.map(g.date)
    f["team"] = f.market_ticker.str.rsplit("-", n=1).str[-1]
    home = f.game_id.map(g.home_team)
    f["home_dir"] = np.where((f.team == home) == (f.side == "yes"), 1, -1)
    f["clv_usd"] = f.clv * f.contracts
    f["win"] = (f.pnl > 0).astype(float)
    return f


# ---------------- bootstrap helpers (day-clustered, like evaluation/stats.py) ----------------

def day_boot(frames: dict, fn, days, reps=REPS, seed=SEED) -> tuple:
    """fn(dict of resampled frames) -> float; frames are resampled by the same game-days. Returns (est, lo, hi)."""
    est = fn(frames)
    days = np.array(sorted(set(days)))
    idx = np.random.default_rng(seed).integers(0, len(days), size=(reps, len(days)))
    frames = {n: f.reset_index(drop=True) for n, f in frames.items()}
    rows_by_day = {n: [np.flatnonzero(f.date.to_numpy() == d) for d in days] for n, f in frames.items()}
    draws = []
    for row in idx:
        res = {n: frames[n].iloc[np.concatenate([r[i] for i in row]) if len(row) else []] for n, r in rows_by_day.items()}
        draws.append(fn(res))
    draws = np.array(draws, float)
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return est, np.nan, np.nan
    return est, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def mean_or_nan(s) -> float:
    return float(s.mean()) if len(s) else np.nan


def random_filter(values: pd.DataFrame, k: int, actual_mean: float, actual_usd: float, seed=SEED) -> dict:
    """Keep k of the trades at random: distribution of mean CLV and CLV $; p = share of draws at least as good."""
    if k <= 0 or k > len(values):
        return {}
    rng = np.random.default_rng(seed)
    clv, usd = values.clv.to_numpy(), values.clv_usd.to_numpy()
    picks = np.argsort(rng.random((RANDOM_DRAWS, len(values))), axis=1)[:, :k]
    means, totals = clv[picks].mean(axis=1), usd[picks].sum(axis=1)
    return {"k": k, "n_pool": len(values), "rand_mean_clv": float(means.mean()),
            "rand_mean_lo": float(np.quantile(means, 0.025)), "rand_mean_hi": float(np.quantile(means, 0.975)),
            "rand_usd": float(totals.mean()), "rand_usd_lo": float(np.quantile(totals, 0.025)),
            "rand_usd_hi": float(np.quantile(totals, 0.975)),
            "p_rand_mean_ge": float((means >= actual_mean - 1e-12).mean()),
            "p_rand_usd_ge": float((totals >= actual_usd - 1e-12).mean()), "_means": means}


# ---------------- 1. overlap ----------------

def cell_stats(f: pd.DataFrame) -> dict:
    return {"n": len(f), "mean_clv": mean_or_nan(f.clv), "clv_usd": float(f.clv_usd.sum()),
            "win_rate": mean_or_nan(f.win), "pnl": float(f.pnl.sum())}


def overlap(a: pd.DataFrame, l: pd.DataFrame, days) -> tuple:
    ak, lk = set(a.key), set(l.key)
    both = ak & lk
    ad, ld = a.set_index("key").home_dir, l.set_index("key").home_dir
    same = {k for k in both if ad[k] == ld[k]}
    opp = both - same
    cells = {"both_same_side (A's fills)": a[a.key.isin(same)], "both_same_side (LLM's fills)": l[l.key.isin(same)],
             "both_opposite (A's fills)": a[a.key.isin(opp)], "both_opposite (LLM's fills)": l[l.key.isin(opp)],
             "A only = filtered by LLM": a[~a.key.isin(lk)], "LLM only = added": l[~l.key.isin(ak)]}
    rows = {c: cell_stats(f) for c, f in cells.items()}
    kept, filtered = a[a.key.isin(lk)], a[~a.key.isin(lk)]
    diff = day_boot({"k": kept, "f": filtered}, lambda d: mean_or_nan(d["k"].clv) - mean_or_nan(d["f"].clv), days)
    if min(len(kept), len(filtered)) < 3:
        diff = (diff[0], np.nan, np.nan)
    added = l[~l.key.isin(ak)]
    sel = {"kept_n": len(kept), "kept_mean_clv": mean_or_nan(kept.clv), "filtered_n": len(filtered),
           "filtered_mean_clv": mean_or_nan(filtered.clv), "kept_minus_filtered": diff[0],
           "kept_minus_filtered_lo": diff[1], "kept_minus_filtered_hi": diff[2],
           "delta_usd_vs_A": float(l.clv_usd.sum() - a.clv_usd.sum()),
           "delta_from_skipping": -float(filtered.clv_usd.sum()), "delta_from_added": float(added.clv_usd.sum()),
           "delta_from_overlap": float(l[l.key.isin(ak)].clv_usd.sum() - kept.clv_usd.sum())}
    sel["random_kept"] = random_filter(a, len(kept), sel["kept_mean_clv"], float(kept.clv_usd.sum()))
    sel["random_portfolio"] = random_filter(a, len(l), mean_or_nan(l.clv), float(l.clv_usd.sum()), seed=SEED + 1)
    return rows, sel, {"kept": kept, "filtered": filtered, "added": l[~l.key.isin(ak)]}


# ---------------- 2. estimates ----------------

def quotes_at(tables, price_index, ticker, ts):
    from replay import AsOf
    q = AsOf(tables, pd.Timestamp(ts), price_index).quote(ticker)
    return None if q is None else (q.bid + q.ask) / 2


def estimate_frame(points, anchor_tr, plain_tr: dict, tool_tr: dict, tables, price_index) -> pd.DataFrame:
    games = tables["games"].set_index("game_id")
    mk = tables["markets"]
    mk = mk[(mk.venue == "kalshi") & (mk.kind == "game")] if "venue" in mk else mk[mk.kind == "game"]
    ticker = {(g, t): m for g, t, m in zip(mk.game_id.astype(str), mk.team, mk.market_ticker)}
    a_by = {key(t["game_id"], t["as_of"]): t for t in anchor_tr}
    rows = []
    for k in points:
        gid, as_of = k
        g = games.loc[gid]
        home, away = g.home_team, g.away_team
        tk = ticker.get((gid, home))
        row = {"key": k, "game_id": gid, "date": g.date, "home": home,
               "home_won": float(g.home_pts > g.away_pts) if pd.notna(g.home_pts) else np.nan}
        if tk is not None:
            row["mid"] = quotes_at(tables, price_index, tk, as_of)
            row["close"] = quotes_at(tables, price_index, tk, g.tip_time)
        t = a_by.get(k)
        if t is not None:
            est = {m.group(1): int(m.group(2)) / 100 for m in re.finditer(r"(\w+) win: (?:model|estimate) (\d+)%", t["brief"])}
            row["A"] = est.get(home, 1 - est[away] if away in est else np.nan)
            fc = next((s["detail"] for s in t["trace"] if s["step"] == "forecast"), "")
            m4 = {m.group(1): float(m.group(3)) for m in re.finditer(r"(\w+) ([\d.]+)->([\d.]+)", fc)}
            row["M4"] = m4.get(home, 1 - m4[away] if away in m4 else np.nan)
        for model, tr in plain_tr.items():
            p = (tr.get(k) or {}).get("proposal") or {}
            if isinstance(p.get("p_home"), (int, float)):
                row[f"B_{model}"] = float(p["p_home"])
        for model, tr in tool_tr.items():
            p = (tr.get(k) or {}).get("proposal") or {}
            if str(p.get("decision", "")).lower() == "buy" and isinstance(p.get("estimate_p"), (int, float)):
                row[f"C_{model}"] = p["estimate_p"] if p.get("team") == home else 1 - p["estimate_p"]
        rows.append(row)
    return pd.DataFrame(rows)


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_estimates(e: pd.DataFrame, days) -> tuple:
    cols = [c for c in ("A", "M4", "B_chat", "B_reasoner", "mid", "close") if c in e]
    common = e.dropna(subset=cols + ["home_won"])
    rows = []
    for c in cols:
        rows.append({"forecaster": c, "n": len(common), "brier": float(brier(common[c], common.home_won).mean()),
                     "log_loss": float(logloss(common[c], common.home_won).mean()),
                     "mean_abs_vs_close": float((common[c] - common.close).abs().mean()),
                     "mean_abs_vs_mid": float((common[c] - common.mid).abs().mean())})
    pairs = []
    for c in ("B_chat", "B_reasoner"):
        for ref in ("A", "mid"):
            d = common.assign(v=brier(common[c], common.home_won) - brier(common[ref], common.home_won))
            est, lo, hi = day_boot({"d": d}, lambda x: mean_or_nan(x["d"].v), days)
            pairs.append({"a": c, "b": ref, "metric": "brier", "estimate": est, "ci_low": lo, "ci_high": hi})
            d = common.assign(v=(common[c] - common.close).abs() - (common[ref] - common.close).abs())
            est, lo, hi = day_boot({"d": d}, lambda x: mean_or_nan(x["d"].v), days)
            pairs.append({"a": c, "b": ref, "metric": "abs_error_vs_close", "estimate": est, "ci_low": lo, "ci_high": hi})
    moves = []
    for c in ("B_chat", "B_reasoner"):
        x = common[(common[c] - common.A).abs() > 0.005]
        toward = ((x[c] - x.A) * (x.close - x.A) > 0).mean()
        closer_mid = ((x[c] - x.mid).abs() < (x.A - x.mid).abs()).mean()
        moves.append({"forecaster": c, "n_moved": len(x), "share_moved_toward_close": float(toward),
                      "share_closer_to_mid_than_A": float(closer_mid),
                      "mean_abs_A_minus_mid": float((common.A - common.mid).abs().mean()),
                      "mean_abs_LLM_minus_mid": float((common[c] - common.mid).abs().mean())})
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(moves), common


def tool_estimates(e: pd.DataFrame) -> pd.DataFrame:
    """On the tool agent's buy proposals: the LLM's estimate vs A, mid and close, signed in the trade's direction."""
    rows = []
    for model in MODELS:
        c = f"C_{model}"
        if c not in e:
            continue
        x = e.dropna(subset=[c, "A", "mid", "close"])
        sign = np.sign(x[c] - x.mid)
        rows.append({"model": model, "n_buy_proposals": len(x),
                     "llm_minus_mid": float((sign * (x[c] - x.mid)).mean()),
                     "A_minus_mid": float((sign * (x.A - x.mid)).mean()),
                     "close_minus_mid": float((sign * (x.close - x.mid)).mean()),
                     "abs_llm_vs_close": float((x[c] - x.close).abs().mean()),
                     "abs_A_vs_close": float((x.A - x.close).abs().mean()),
                     "abs_mid_vs_close": float((x.mid - x.close).abs().mean()),
                     "llm_equals_A_within_1pt": float(((x[c] - x.A).abs() <= 0.01).mean())})
    return pd.DataFrame(rows)


# ---------------- 3. sceptic veto reasons ----------------

NO_NEWS = r"no (fresh |published |listed )?news|zero (fresh )?news|no news|news shift is (exactly )?zero|zero (model |news )?(news )?shift|" \
          r"shift of 0\.0|model[- ]vs[- ]market|model-versus-market|stale model|no fresh|no corresponding news|no offsetting news|" \
          r"no news catalyst|unsupported by any news"
NEWS = r"injur|absence|absent|suspension|inactive|out news|outs are|news is|news shift|news-driven|queta|george"
AGAINST = r"against|fallen|lower|decline|adverse|drifted down|moved down|repriced|negative move|fell"
SMALL = r"barely|marginal|thin|only slightly|small re|too thin|minimum"
FLAGS = {"no_news_model_vs_market": NO_NEWS, "news_priced_in": NEWS, "market_moved_against": AGAINST,
         "small_edge": SMALL, "stale_anchor": r"stale|anchor"}


def veto_label(reason: str) -> tuple:
    r = reason.lower()
    flags = {f: bool(re.search(p, r)) for f, p in FLAGS.items()}
    if "fail closed" in r:
        primary = "sceptic failed (fail closed)"
    elif flags["no_news_model_vs_market"]:
        primary = "model-vs-market disagreement, no news"
    elif flags["news_priced_in"]:
        primary = "news already priced in"
    elif flags["market_moved_against"]:
        primary = "market moved against"
    elif flags["small_edge"]:
        primary = "small edge"
    else:
        primary = "other"
    return primary, flags


def sceptic_table(d_tr: list, c_fills: pd.DataFrame, a_fills: pd.DataFrame, tables, price_index) -> pd.DataFrame:
    from evaluation.llm_agent_eval import close_clv
    c = c_fills.set_index("key")
    a = a_fills.set_index("key")
    rows = []
    for t in d_tr:
        if "sceptic" not in t:
            continue
        k = key(t["game_id"], t["as_of"])
        primary, flags = veto_label(t["sceptic"]["reason"])
        cand = t.get("candidate") or t.get("order") or {}
        cf = close_clv(tables, price_index, cand["ticker"], cand["side"], cand["price"], t["game_id"]) if cand else np.nan
        rows.append({"key": k, "game_id": t["game_id"], "date": tables["games"].set_index("game_id").date[t["game_id"]],
                     "verdict": t["sceptic"]["verdict"], "primary": primary, **flags, "clv": cf,
                     "clv_usd": float(c.clv_usd[k]) if k in c.index else np.nan,
                     "c_win": float(c.win[k]) if k in c.index else np.nan, "A_traded": k in a.index,
                     "reason": t["sceptic"]["reason"][:160]})
    return pd.DataFrame(rows)


# ---------------- 4. quality and guardrails ----------------

def quality(setup: str, traces: list, fills: pd.DataFrame) -> dict:
    live = [t for t in traces if t["status"] not in ("held", "idle")]
    asked = [t for t in live if t.get("llm", {}).get("calls", 0) > 0]
    status = Counter(t["status"] for t in traces)
    invalid = Counter(t["reason"] for t in live if t["status"] == "invalid")
    calls = [len(t.get("tool_calls", [])) for t in asked]
    seqs = Counter(" > ".join(c["tool"] for c in t.get("tool_calls", [])[:4]) for t in asked if t.get("tool_calls"))
    tool_errors = sum(c.get("error", False) for t in asked for c in t.get("tool_calls", []))
    if setup == "plain":
        buys = [t for t in live if t["status"] in ("order", "gap", "risk_block")]
        cite_ok = len(buys)
    else:
        buys = [t for t in live if str((t.get("proposal") or {}).get("decision", "")).lower() == "buy"]
        cite_ok = sum(t["status"] not in ("invalid",) for t in buys)
    if setup == "plain":
        cite_ok = np.nan
    return {"setup": setup, "decision_points": len(traces), "llm_asked": len(asked), "held": status.get("held", 0),
            "pass": status.get("pass", 0), "invalid": sum(invalid.values()),
            "invalid_rate": sum(invalid.values()) / len(asked) if asked else np.nan,
            "invalid_by_reason": dict(invalid), "tool_calls_mean": float(np.mean(calls)) if calls else 0.0,
            "tool_calls_median": float(np.median(calls)) if calls else 0.0, "tool_calls_max": max(calls, default=0),
            "tool_errors": tool_errors, "top_sequence": seqs.most_common(1)[0] if seqs else None,
            "buy_proposals": len(buys), "citation_valid": cite_ok,
            "citation_valid_rate": cite_ok / len(buys) if buys else np.nan,
            "gap_fail": status.get("gap", 0), "sceptic_reject": status.get("sceptic_reject", 0),
            "risk_block": status.get("risk_block", 0), "orders": status.get("order", 0), "fills": len(fills)}


def cost_latency(model: str, traces: list, fills: pd.DataFrame) -> dict:
    from agents.llm_client import cost_usd
    llm = [t.get("llm", {}) for t in traces]
    tin, tout = sum(x.get("tokens_in", 0) for x in llm), sum(x.get("tokens_out", 0) for x in llm)
    cost = cost_usd(model, tin, tout)
    lat = np.array([x.get("latency", 0.0) for x in llm if x.get("calls", 0)])
    return {"llm_calls": sum(x.get("calls", 0) for x in llm), "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
            "cost_per_decision": cost / len(traces), "cost_per_trade": cost / len(fills) if len(fills) else np.nan,
            "latency_per_decision_s": float(lat.mean()) if len(lat) else np.nan,
            "latency_p90_s": float(np.quantile(lat, 0.9)) if len(lat) else np.nan,
            "llm_seconds_per_trade": float(lat.sum() / len(fills)) if len(fills) else np.nan}


# ---------------- 5. side by side ----------------

def short(text: str, n=170) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + " …"


def a_line(t: dict, team: str | None) -> str:
    lines = t["brief"].splitlines()[1:]
    hit = [x for x in lines if "buying" in x] or [x for x in lines if team and x.startswith(f"{team} win")] or lines
    return short(hit[0] if hit else t["brief"])


def examples(keys, a_tr, c_tr, d_tr, a_fills, c_fills, games) -> list:
    a_by = {key(t["game_id"], t["as_of"]): t for t in a_tr}
    c_by = {key(t["game_id"], t["as_of"]): t for t in c_tr}
    d_by = {key(t["game_id"], t["as_of"]): t for t in d_tr}
    af, cf = a_fills.set_index("key"), c_fills.set_index("key")
    g = games.set_index("game_id")
    out = []
    for k, why in keys:
        gid = k[0]
        c, d = c_by.get(k, {}), d_by.get(k, {})
        prop = c.get("proposal") or {}
        llm = prop.get("rationale") or c.get("reason", "")
        out.append({"why": why, "game": f"{g.away_team[gid]} @ {g.home_team[gid]}, {k[1][:16].replace('T', ' ')} UTC",
                    "A": a_line(a_by[k], prop.get("team")) if k in a_by else "–",
                    "A_clv": float(af.clv[k]) if k in af.index else np.nan,
                    "C": f"{c.get('status')}: " + short(llm), "C_clv": float(cf.clv[k]) if k in cf.index else np.nan,
                    "D": f"{d.get('status')}: " + short((d.get("sceptic") or {}).get("reason") or d.get("reason", ""))})
    return out


# ---------------- figure ----------------

def figure(groups: dict, rand: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    labels, vals, errs, ns = [], [], [], []
    for name, f in groups.items():
        labels.append(name)
        vals.append(f.clv.mean() * 100 if len(f) else np.nan)
        errs.append(f.clv.std(ddof=1) / np.sqrt(len(f)) * 196 if len(f) > 1 else 0)
        ns.append(len(f))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#2ca02c"][: len(labels)]
    ax.bar(range(len(labels)), vals, yerr=errs, color=colors, capsize=3)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{x}\n(n={n})" for x, n in zip(labels, ns)], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean CLV per contract (cents), ±1.96 SE")
    ax.set_title("Reasoner vs A: which of A's trades the LLM kept / filtered / added")
    ax = axes[1]
    for name, (r, actual, color) in rand.items():
        if r:
            ax.hist(r["_means"] * 100, bins=60, alpha=0.45, color=color, label=f"random {r['k']} of {r['n_pool']}")
            ax.axvline(actual * 100, color=color, lw=2, ls="--", label=f"{name}: {actual * 100:+.2f}¢ (p={r['p_rand_mean_ge']:.2f})")
    ax.set_xlabel("mean CLV per contract of the kept trades (cents)")
    ax.set_title("LLM filter vs keeping the same number at random")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------- report ----------------

def c(x, n=2):
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.{n}f}¢"


def usd(x):
    return "–" if pd.isna(x) else f"{x:+,.1f}"


def pct(x):
    return "–" if pd.isna(x) else f"{x:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "llm_agent"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from evaluation.llm_agent_eval import load_frozen
    tables = load_frozen()
    tables["games"] = tables["games"].assign(game_id=tables["games"].game_id.astype(str))
    tables["markets"] = tables["markets"].assign(game_id=tables["markets"].game_id.astype(str))
    games = tables["games"]
    prices = tables["prices"].sort_values("ts")
    price_index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}

    base_a = runs / MODELS["reasoner"][0] / "anchor"
    a_tr = load_traces(base_a)
    a_f = load_fills(base_a, games)
    points = sorted({key(t["game_id"], t["as_of"]) for t in a_tr})
    days = sorted({games.set_index("game_id").date[k[0]] for k in points})
    trs = {(m, s): load_traces(runs / MODELS[m][0] / s) for m in MODELS for s in ARMS}
    fills = {(m, s): load_fills(runs / MODELS[m][0] / s, games) for m in MODELS for s in ARMS}
    for (m, s), tr in trs.items():
        assert {key(t["game_id"], t["as_of"]) for t in tr} <= set(points), (m, s)
    csv = []

    # 1. overlap
    over_rows, sel_rows, groups_reasoner = [], [], {}
    for m in MODELS:
        for s in ARMS:
            cells, sel, grp = overlap(a_f, fills[(m, s)], days)
            for cell, v in cells.items():
                over_rows.append({"model": m, "arm": s, "cell": cell, **v})
            sel_rows.append({"model": m, "arm": s, "llm_trades": len(fills[(m, s)]),
                             "llm_mean_clv": mean_or_nan(fills[(m, s)].clv),
                             "llm_clv_usd": float(fills[(m, s)].clv_usd.sum()), **{k: v for k, v in sel.items()
                                                                                    if not k.startswith("random")},
                             **{f"keptrand_{k}": v for k, v in sel["random_kept"].items() if k != "_means"},
                             **{f"portrand_{k}": v for k, v in sel["random_portfolio"].items() if k != "_means"}})
            if m == "reasoner":
                groups_reasoner[s] = (grp, sel)
    over, sel = pd.DataFrame(over_rows), pd.DataFrame(sel_rows)
    csv += [over.assign(section="1_overlap"), sel.assign(section="1_selection")]

    # sceptic as a filter on C's trades (reasoner)
    c_f, d_f = fills[("reasoner", "tool")], fills[("reasoner", "tool_sceptic")]
    veto = sceptic_table(trs[("reasoner", "tool_sceptic")], c_f, a_f, tables, price_index)
    veto_chat = sceptic_table(trs[("chat", "tool_sceptic")], fills[("chat", "tool")], a_f, tables, price_index)
    kept_c = c_f[c_f.key.isin(d_f.key)]
    sk_rand = random_filter(c_f, len(kept_c), mean_or_nan(kept_c.clv), float(kept_c.clv_usd.sum()), seed=SEED + 2)
    vetoed_c = c_f[c_f.key.isin(veto.loc[veto.verdict != "approve", "key"])]
    sk_diff = day_boot({"k": kept_c, "v": vetoed_c}, lambda d: mean_or_nan(d["k"].clv) - mean_or_nan(d["v"].clv), days)
    csv.append(pd.DataFrame([{"section": "1_sceptic_filter", "model": "reasoner", "kept_n": len(kept_c),
                              "kept_mean_clv": mean_or_nan(kept_c.clv), "vetoed_n": len(vetoed_c),
                              "vetoed_mean_clv": mean_or_nan(vetoed_c.clv), "kept_minus_vetoed": sk_diff[0],
                              "kept_minus_vetoed_lo": sk_diff[1], "kept_minus_vetoed_hi": sk_diff[2],
                              **{f"rand_{k}": v for k, v in sk_rand.items() if k != "_means"}}]))

    # 2. estimates
    plain_tr = {m: {key(t["game_id"], t["as_of"]): t for t in trs[(m, "plain")]} for m in MODELS}
    tool_tr = {m: {key(t["game_id"], t["as_of"]): t for t in trs[(m, "tool")]} for m in MODELS}
    est = estimate_frame(points, a_tr, plain_tr, tool_tr, tables, price_index)
    scores, est_pairs, moves, common = score_estimates(est, days)
    tool_est = tool_estimates(est)
    csv += [scores.assign(section="2_scores"), est_pairs.assign(section="2_paired"), moves.assign(section="2_moves"),
            tool_est.assign(section="2_tool_proposals")]

    # 3. sceptic categories
    rej = veto[veto.verdict != "approve"]
    by_cat = rej.groupby("primary").agg(n=("clv", "size"), mean_clv=("clv", "mean"), clv_usd=("clv_usd", "sum"),
                                        share_positive=("clv", lambda x: (x > 0).mean()),
                                        A_also_traded=("A_traded", "mean")).reset_index().sort_values("n", ascending=False)
    flag_rows = [{"flag": f, "n": int(rej[f].sum()), "mean_clv": mean_or_nan(rej.loc[rej[f], "clv"]),
                  "share_positive": mean_or_nan((rej.loc[rej[f], "clv"] > 0))} for f in FLAGS]
    flags = pd.DataFrame(flag_rows)
    wrong, right = rej[rej.clv > 0], rej[rej.clv <= 0]
    csv += [by_cat.assign(section="3_veto_primary"), flags.assign(section="3_veto_flags"),
            veto.drop(columns=["reason"]).assign(section="3_veto_rows", key=veto.key.astype(str))]

    # 4. quality
    qual = pd.DataFrame([{"model": m, **quality(s, trs[(m, s)], fills[(m, s)])} for m in MODELS for s in ARMS])
    csv.append(qual.astype({"invalid_by_reason": str, "top_sequence": str}).assign(section="4_quality"))
    a_status = Counter(t["status"] for t in a_tr)
    a_checks = Counter(next((s["detail"] for s in t["trace"] if s["step"] == "checks"), "–") for t in a_tr)

    # 6. cost
    cost = pd.DataFrame([{"model": m, "arm": s, **cost_latency(MODELS[m][1], trs[(m, s)], fills[(m, s)])}
                         for m in MODELS for s in ARMS])
    csv.append(cost.assign(section="6_cost"))

    # 5. examples
    vetoed_wrong = rej.sort_values("clv", ascending=False)
    c_by = {key(t["game_id"], t["as_of"]): t for t in trs[("reasoner", "tool")]}
    filtered = a_f[~a_f.key.isin(c_f.key)].sort_values("clv")
    a_pass = next((k for k in filtered.key if c_by.get(k, {}).get("status") == "pass"), None)
    ex_keys = [(key("401810570", "2026-02-03 02:30:00+00:00"), "A and C both buy; the sceptic vetoes"),
               (d_f.sort_values("clv").key.iloc[-1], "the sceptic's best kept trade"),
               (a_pass, "A buys, the reasoner passes (A's worst trade among those the LLM passed on)")]
    if len(vetoed_wrong):
        ex_keys.append((vetoed_wrong.key.iloc[0], "veto the sceptic got most wrong (best counterfactual CLV)"))
    ex = examples([x for x in ex_keys if x[0]], a_tr, trs[("reasoner", "tool")], trs[("reasoner", "tool_sceptic")],
                  a_f, c_f, games)
    csv.append(pd.DataFrame(ex).assign(section="5_examples"))

    pd.concat(csv, ignore_index=True).to_csv(out / "llm_vs_original.csv", index=False)

    # figure
    g, s_ = groups_reasoner["tool"]
    fig_groups = {"A all": a_f, "A filtered by C": g["filtered"], "A kept by C": g["kept"], "C added": g["added"],
                  "C vetoed by sceptic": vetoed_c, "D kept": kept_c}
    figure(fig_groups, {"C keeps of A": (s_["random_kept"], s_["kept_mean_clv"], "#ff7f0e"),
                        "sceptic keeps of C": (sk_rand, mean_or_nan(kept_c.clv), "#2ca02c")},
           out / "llm_vs_original.png")

    # ---------------- markdown ----------------
    md = ["# What the DeepSeek LLM agent changes vs the original deterministic agent (A)", "",
          "Decision-level comparison on the **same 304 test decision points** (1 Feb – 12 Apr 2026, fixed 40% "
          "subsample, seed 7606). A = deterministic anchor agent (market price 24 h before tip + M4 news shift). "
          "B = plain LLM (one call), C = LLM tool agent, D = C + priced-in sceptic; `chat` = `deepseek-chat`, "
          "`reasoner` = `deepseek-reasoner` (V4.1-Flash, thinking on). CLV is per contract against the mid at tip "
          "(1¢ = 0.01). CIs are 95%, day-clustered bootstrap over the 64 game-days "
          f"({REPS} replicates, seed {SEED}); random-filter baselines use {RANDOM_DRAWS:,} draws. "
          "No new LLM calls: everything is read from the cached runs. Script: `evaluation/llm_vs_original.py`.", ""]

    r = sel.set_index(["model", "arm"])
    rc, rd = r.loc[("reasoner", "tool")], r.loc[("reasoner", "tool_sceptic")]
    sc = scores.set_index("forecaster")
    bp = est_pairs.set_index(["a", "b", "metric"])
    ba, bm = bp.loc[("B_chat", "A", "brier")], bp.loc[("B_chat", "mid", "brier")]
    te = tool_est.set_index("model").loc["reasoner"]
    q = qual.set_index(["model", "setup"])
    cl = cost.set_index(["model", "arm"])
    oc = over.set_index(["model", "arm", "cell"])
    added_c = oc.loc[("reasoner", "tool", "LLM only = added")]
    rc_kept_win = oc.loc[("reasoner", "tool", "both_same_side (A's fills)"), "win_rate"]
    rc_drop_win = oc.loc[("reasoner", "tool", "A only = filtered by LLM"), "win_rate"]
    md += ["## Verdict", "",
           "**What the LLM improves**", "",
           f"- **Fewer trades, and that is most of the gain.** A makes {len(a_f)} trades for CLV "
           f"{usd(a_f.clv_usd.sum())} $ (mean {c(a_f.clv.mean())} per contract). The reasoner with the sceptic (D) "
           f"makes {int(rd.llm_trades)} ({usd(rd.llm_clv_usd)} $). Its {usd(rd.delta_usd_vs_A)} $ over A splits "
           f"into {usd(rd.delta_from_skipping)} $ from skipping {int(rd.filtered_n)} of A's trades, "
           f"{usd(rd.delta_from_added)} $ from one added trade (a 14¢ longshot, 142 contracts) and "
           f"{usd(rd.delta_from_overlap)} $ on the one trade both made. Any filter that keeps 2 of A's trades "
           f"gets the skipping part: random 2 of 53 gives {usd(rd.portrand_rand_usd)} $ [{usd(rd.portrand_rand_usd_lo)}, "
           f"{usd(rd.portrand_rand_usd_hi)}].",
           f"- **D's 2 trades beat 2 random trades from A, but on n = 2.** Mean CLV {c(rd.llm_mean_clv)} vs random "
           f"{c(rd.portrand_rand_mean_clv)} (p = {rd.portrand_p_rand_mean_ge:.2f} that random is at least as good); "
           f"CLV $ {usd(rd.llm_clv_usd)} vs random [{usd(rd.portrand_rand_usd_lo)}, {usd(rd.portrand_rand_usd_hi)}] "
           "(the $ is driven by the contract count of one cheap longshot).",
           f"- **The sceptic's vetoes lean the right way, not significantly.** Of C's {len(c_f)} reasoner trades, the "
           f"{len(vetoed_c)} vetoed averaged {c(mean_or_nan(vetoed_c.clv))} and the {len(kept_c)} kept "
           f"{c(mean_or_nan(kept_c.clv))}: kept − vetoed {c(sk_diff[0])} [{c(sk_diff[1])}, {c(sk_diff[2])}]; "
           f"a random {len(kept_c)} of {len(c_f)} does at least as well {pct(sk_rand.get('p_rand_mean_ge'))} of the time.",
           "- **Explainability and auditability.** Every LLM trade carries a rationale that cites tool outputs, the "
           "citations are machine-checked against those outputs, and the sceptic's reasons name a concrete concern "
           "(stale 24-hour anchor, price already moved, no news behind the edge) — A's brief only shows the numbers.",
           "", "**What it does not improve (or makes worse)**", "",
           f"- **No better selection among A's trades.** The reasoner tool agent (C) keeps {int(rc.kept_n)} of A's "
           f"trades and drops {int(rc.filtered_n)}: kept {c(rc.kept_mean_clv)} vs dropped {c(rc.filtered_mean_clv)} "
           f"(kept − dropped {c(rc.kept_minus_filtered)} [{c(rc.kept_minus_filtered_lo)}, "
           f"{c(rc.kept_minus_filtered_hi)}]); a random keep of the same size is at least as good "
           f"{pct(rc.keptrand_p_rand_mean_ge)} of the time. The {int(added_c.n)} trades it adds average "
           f"{c(added_c.mean_clv)}, the same as A. On outcomes (noisy) the kept A trades won {pct(rc_kept_win)} vs "
           f"{pct(rc_drop_win)} for the dropped ones. D's one kept A trade was {c(rd.kept_mean_clv)}.",
           f"- **No better forecasts than the market.** The plain LLM's home-win probabilities beat A's Brier score "
           f"(chat {ba.estimate:+.4f} [{ba.ci_low:+.4f}, {ba.ci_high:+.4f}]) only because they copy the current "
           f"mid (mean distance {sc.loc['B_chat', 'mean_abs_vs_mid'] * 100:.2f} points); vs the mid itself: "
           f"{bm.estimate:+.4f} [{bm.ci_low:+.4f}, {bm.ci_high:+.4f}]. \"Moving toward the close\" is just moving to "
           "the market, which already sits near the close: more conservative, not more informed.",
           f"- **Tool-agent estimates are more overconfident than A's.** On its {int(te.n_buy_proposals)} buy "
           f"proposals the reasoner's estimate sits {c(te.llm_minus_mid, 1)} above the mid for the side it buys "
           f"(A at the same points: {c(te.A_minus_mid, 1)}); the closing price moved {c(te.close_minus_mid, 1)}.",
           f"- **Reliability, latency, cost.** {pct(q.loc[('reasoner', 'tool'), 'invalid_rate'])} of reasoner "
           "tool-agent decisions were invalid (empty/malformed JSON, often after spending the reasoning budget; "
           f"{int(q.loc[('reasoner', 'tool'), 'buy_proposals'] - q.loc[('reasoner', 'tool'), 'citation_valid'])} buy "
           "proposals failed the citation check), all treated as passes. The reasoner tool agent needs "
           f"{cl.loc[('reasoner', 'tool_sceptic'), 'latency_per_decision_s']:.0f} s of LLM time per decision and "
           f"${cl.loc[('reasoner', 'tool_sceptic'), 'cost_per_trade']:.2f} per trade with the sceptic; A runs in "
           "milliseconds for free. The guardrails (checks and risk limits) never had to block either agent.",
           ""]

    md += ["## 1. Decision-level overlap with A", "",
           "Cells by decision point. \"Filtered\" = A traded there and the LLM did not; \"added\" = the LLM traded "
           "where A did not. Same/opposite side compares the direction on the home team. Win = P&L > 0.", "",
           "| Model | Arm | Cell | n | Mean CLV | CLV $ | Win rate | P&L $ |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in over.itertuples():
        if x.n == 0 and "LLM's" in x.cell:
            continue
        md.append(f"| {x.model} | {ARMS[x.arm]} | {x.cell} | {x.n} | {c(x.mean_clv)} | {usd(x.clv_usd)} | "
                  f"{pct(x.win_rate)} | {usd(x.pnl)} |")
    md += ["", "**Selection vs fewer trades.** Kept/filtered use A's own fills, so the LLM's choice is the only "
           "difference. Random filter = keep the same number of A's 53 trades at random; p = share of random draws "
           "with mean CLV at least as high as the LLM's kept set (high p = no better than random). Portfolio = the "
           "LLM's actual trades vs the same number of A's trades at random.", "",
           "| Model | Arm | LLM trades (CLV $) | Kept / filtered | Kept mean CLV | Filtered mean CLV | Kept − filtered [95% CI] | "
           "Random keep: mean CLV [95%] | p (random ≥ LLM) | Random portfolio CLV $ [95%] | p (random $ ≥ LLM $) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in sel.itertuples():
        kr = f"{c(x.keptrand_rand_mean_clv)} [{c(x.keptrand_rand_mean_lo)}, {c(x.keptrand_rand_mean_hi)}]" \
            if not pd.isna(getattr(x, "keptrand_rand_mean_clv", np.nan)) else "–"
        pr = f"{usd(x.portrand_rand_usd)} [{usd(x.portrand_rand_usd_lo)}, {usd(x.portrand_rand_usd_hi)}]" \
            if not pd.isna(getattr(x, "portrand_rand_usd", np.nan)) else "–"
        md.append(f"| {x.model} | {ARMS[x.arm]} | {x.llm_trades} ({usd(x.llm_clv_usd)}) | {x.kept_n} / {x.filtered_n} | "
                  f"{c(x.kept_mean_clv)} | {c(x.filtered_mean_clv)} | {c(x.kept_minus_filtered)} "
                  f"[{c(x.kept_minus_filtered_lo)}, {c(x.kept_minus_filtered_hi)}] | {kr} | "
                  f"{pct(getattr(x, 'keptrand_p_rand_mean_ge', np.nan))} | {pr} | "
                  f"{pct(getattr(x, 'portrand_p_rand_usd_ge', np.nan))} |")
    md += ["", f"**The sceptic as a filter on C (reasoner).** Of C's {len(c_f)} trades the sceptic kept {len(kept_c)} "
           f"({c(mean_or_nan(kept_c.clv))}) and vetoed {len(vetoed_c)} ({c(mean_or_nan(vetoed_c.clv))}); kept − vetoed "
           f"{c(sk_diff[0])} [{c(sk_diff[1])}, {c(sk_diff[2])}]. Random {len(kept_c)} of {len(c_f)}: mean "
           f"{c(sk_rand.get('rand_mean_clv'))} [{c(sk_rand.get('rand_mean_lo'))}, {c(sk_rand.get('rand_mean_hi'))}], "
           f"p (random ≥ sceptic) = {sk_rand.get('p_rand_mean_ge', np.nan):.2f}; CLV $ random "
           f"{usd(sk_rand.get('rand_usd'))} [{usd(sk_rand.get('rand_usd_lo'))}, {usd(sk_rand.get('rand_usd_hi'))}] vs "
           f"kept {usd(kept_c.clv_usd.sum())}. With n = {len(kept_c)} this cannot show skill.", "",
           "![Kept vs filtered vs added, and the random-filter baseline](llm_vs_original.png)", ""]

    md += ["## 2. Probability estimates", "",
           f"Home-win probability at every decision point where all forecasters exist (n = {len(common)}). "
           "A's estimate is read from its brief (rounded to 1 point); M4 is the raw model; mid = Kalshi mid at the "
           "decision; close = mid at tip. Only the plain LLM (B) states a probability at every point; the tool "
           "agents give one only when they propose a trade (see below). Lower is better.", "",
           "| Forecaster | Brier | Log loss | Mean abs. distance to close | Mean abs. distance to mid |",
           "| --- | --- | --- | --- | --- |"]
    for x in scores.itertuples():
        md.append(f"| {x.forecaster} | {x.brier:.4f} | {x.log_loss:.4f} | {x.mean_abs_vs_close:.4f} | {x.mean_abs_vs_mid:.4f} |")
    md += ["", "Paired differences (first minus second, negative = LLM better):", "",
           "| LLM | vs | Metric | Difference [95% CI] |", "| --- | --- | --- | --- |"]
    for x in est_pairs.itertuples():
        md.append(f"| {x.a} | {x.b} | {x.metric} | {x.estimate:+.4f} [{x.ci_low:+.4f}, {x.ci_high:+.4f}] |")
    md += ["", "Direction of the LLM's change from A's estimate (points where it differs by > 0.5 points):", "",
           "| LLM | n moved | Moved toward close | Closer to the current mid than A | Mean abs. A − mid | Mean abs. LLM − mid |",
           "| --- | --- | --- | --- | --- | --- |"]
    for x in moves.itertuples():
        md.append(f"| {x.forecaster} | {x.n_moved} | {pct(x.share_moved_toward_close)} | "
                  f"{pct(x.share_closer_to_mid_than_A)} | {x.mean_abs_A_minus_mid:.4f} | {x.mean_abs_LLM_minus_mid:.4f} |")
    md += ["", "On the tool agent's buy proposals (C), signed in the direction of the proposed trade (positive = "
           "estimate above the mid for the side it buys):", "",
           "| Model | Buy proposals | LLM − mid | A − mid | Close − mid (what happened) | abs. LLM − close | "
           "abs. A − close | abs. mid − close | LLM within 1 pt of A |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in tool_est.itertuples():
        md.append(f"| {x.model} | {x.n_buy_proposals} | {c(x.llm_minus_mid, 1)} | {c(x.A_minus_mid, 1)} | "
                  f"{c(x.close_minus_mid, 1)} | {x.abs_llm_vs_close:.4f} | {x.abs_A_vs_close:.4f} | "
                  f"{x.abs_mid_vs_close:.4f} | {pct(x.llm_equals_A_within_1pt)} |")
    md.append("")

    md += ["## 3. Sceptic veto reasons (reasoner D)", "",
           f"{len(rej)} vetoes of {len(veto)} sceptic calls. Primary category by keyword rules in priority order "
           "(fail-closed → no-news model-vs-market → news already priced in → market moved against → small edge → "
           "other), then spot-checked by hand. Counterfactual CLV = side's mid at tip minus the price it would have "
           "paid; CLV $ uses C's fill for the same decision.", "",
           "| Primary reason | n | Mean CLV | CLV $ avoided | Share with positive CLV (veto wrong) | A also traded |",
           "| --- | --- | --- | --- | --- | --- |"]
    for x in by_cat.itertuples():
        md.append(f"| {x.primary} | {x.n} | {c(x.mean_clv)} | {usd(x.clv_usd)} | {pct(x.share_positive)} | {pct(x.A_also_traded)} |")
    md += ["", "Overlapping flags (a veto can mention several):", "", "| Flag | n | Mean CLV | Share positive |",
           "| --- | --- | --- | --- |"]
    for x in flags.itertuples():
        md.append(f"| {x.flag} | {x.n} | {c(x.mean_clv)} | {pct(x.share_positive)} |")
    wf = {f: mean_or_nan(wrong[f].astype(float)) for f in FLAGS}
    rf = {f: mean_or_nan(right[f].astype(float)) for f in FLAGS}
    md += ["", f"**Vetoes the sceptic got wrong** (positive counterfactual CLV): {len(wrong)} of {len(rej)}, mean "
           f"{c(mean_or_nan(wrong.clv))}, vs {len(right)} right ({c(mean_or_nan(right.clv))}). Flag rates wrong vs right: "
           + "; ".join(f"{f} {pct(wf[f])} vs {pct(rf[f])}" for f in FLAGS)
           + f". Chat D: {len(veto_chat[veto_chat.verdict != 'approve'])} vetoes of {len(veto_chat)}, mean "
           f"{c(mean_or_nan(veto_chat.loc[veto_chat.verdict != 'approve', 'clv']))}.", ""]

    md += ["## 4. Quality and guardrails", "",
           "| Model | Arm | Decision points (LLM asked) | Invalid (rate) | Invalid by reason | Tool calls mean / median / max | "
           "Buy proposals | Citations valid | Gap fails | Sceptic rejects | Risk blocks | Orders | Fills |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in qual.itertuples():
        md.append(f"| {x.model} | {ARMS[x.setup]} | {x.decision_points} ({x.llm_asked}) | {x.invalid} ({x.invalid_rate:.1%}) | "
                  f"{x.invalid_by_reason or 'none'} | {x.tool_calls_mean:.1f} / {x.tool_calls_median:.0f} / {x.tool_calls_max} | "
                  f"{x.buy_proposals} | {'–' if pd.isna(x.citation_valid) else f'{x.citation_valid:.0f} ({pct(x.citation_valid_rate)})'} | {x.gap_fail} | "
                  f"{x.sceptic_reject} | {x.risk_block} | {x.orders} | {x.fills} |")
    tops = "; ".join(f"{x.model} {x.setup}: `{x.top_sequence[0]}` ({x.top_sequence[1]}×)"
                     for x in qual.itertuples() if x.top_sequence)
    md += ["", f"Most common first four tool calls: {tops}.", "",
           f"A (deterministic): {dict(a_status)}; pre-trade checks {dict(a_checks)}; all {len(a_f)} orders "
           "approved by risk and filled. The guardrails never had to stop A or the LLM at the risk stage (0 risk "
           "blocks); for the LLM the binding guardrails are earlier: output validation (citations must match tool "
           "outputs, estimate within a band of the model/market references), the code-computed 4¢ gap after fees "
           "and the sceptic. Plain B proposes a probability at every point, but 97–99% fail the gap check. "
           "\"Citations valid\" = buy proposals that passed the citation / number / grounding checks (plain B: "
           "no citations; its \"buy proposals\" are the probabilities it stated).", ""]

    md += ["## 5. Same decision, side by side (reasoner)", ""]
    for e in ex:
        md += [f"**{e['game']}** — {e['why']}", "",
               f"- A: {e['A']}" + ("" if pd.isna(e["A_clv"]) else f" (CLV {c(e['A_clv'], 1)})"),
               f"- LLM C ({e['C']})" + ("" if pd.isna(e["C_clv"]) else f" (CLV {c(e['C_clv'], 1)})"),
               f"- Sceptic D ({e['D']})", ""]

    md += ["## 6. Cost and latency", "",
           "Token-based estimate of a fresh run at `agents/llm_client.PRICES` ($0.27 / $1.10 per M tokens). For "
           "C, the analyst calls were replayed from D's cache. Actual DeepSeek spend for all six arms: $3.49. "
           "Latency = LLM seconds per decision point where the LLM was asked (sequential calls within a decision). "
           "A: $0 and milliseconds per decision.", "",
           "| Model | Arm | LLM calls | Cost | $ / decision | $ / trade | Latency / decision (p90) | LLM seconds / trade |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in cost.itertuples():
        md.append(f"| {x.model} | {ARMS[x.arm]} | {x.llm_calls:,} | ${x.cost_usd:.2f} | ${x.cost_per_decision:.4f} | "
                  f"{'–' if pd.isna(x.cost_per_trade) else f'${x.cost_per_trade:.3f}'} | "
                  f"{x.latency_per_decision_s:.1f}s ({x.latency_p90_s:.1f}s) | "
                  f"{'–' if pd.isna(x.llm_seconds_per_trade) else f'{x.llm_seconds_per_trade:,.0f}s'} |")
    md += ["", "Caveats: one 70-day test window, a 40% subsample, n = 2 kept trades for D; the A estimate is "
           "rounded to 1 point; CLV is against the mid at tip, so it ignores the spread paid (P&L does not)."]
    (out / "llm_vs_original.md").write_text("\n".join(md) + "\n")
    print((out / "llm_vs_original.md").read_text())


if __name__ == "__main__":
    main()
