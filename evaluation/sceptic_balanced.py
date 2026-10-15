"""Approach A: a balanced, scored sceptic (score in [0, 1], approve iff score >= tau) vs the current approve/reject one.

Labelled set: every proposal the current deepseek-reasoner sceptic reviewed in the two walk-forward windows
(Dec 2025 - Jan 2026 with refit models, runs/llm_agent/prev-tool_sceptic; Feb - Apr 2026 test, the finished
full-sample runs used by llm_sceptic_full). The exact user prompt that sceptic saw is read back from
runs/llm_cache/, so the balanced sceptic gets the same evidence; only the system prompt and the last line change.
Each proposal's counterfactual fill, CLV and P&L come from Replay._fill / Replay._settle at the decision time
($20 stake, fill at the ask, Kalshi fee in P&L, CLV = mid at tip minus the price paid, as for real fills).

tau is tuned on one window (max CLV $ of the kept set, at least MIN_KEPT kept) and applied unchanged to the other.

    python -m evaluation.sceptic_balanced --score        # API calls (cached), then the report
    python -m evaluation.sceptic_balanced                # report from the cache only
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, RESULTS, WINDOWS, load_frozen, load_traces
from evaluation.llm_sceptic_full import from_runs
from evaluation.stats import REPS, SEED

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "sceptic_balanced"
STEM = "sceptic_balanced"
STAKE = 20.0
MIN_EDGE = 0.04
MIN_KEPT = 5
RANDOM_DRAWS = 20000
REASONER = "deepseek-reasoner"
SAMPLES = {REASONER: 1}
CONNECT_S, READ_S, ATTEMPT_S, FIRST_TOKEN_S = 10.0, 120.0, 180.0, 60.0
RETRIES = 4
LONG_MAX_TOKENS, LONG_ROLE = 32768, "@max32k"
WINDOW_NAMES = {"prev": "Dec 2025 - Jan 2026", "test": "Feb - Apr 2026 (test)"}
HEAD = re.compile(r"Game: (\w+) \(away\) at (\w+) \(home\), game date (\S+)\. Tip-off in (\d+) minutes")


# ---------------- labelled set ----------------

def reviewed_traces(games: pd.DataFrame) -> list:
    """(window, trace) for every proposal the reasoner sceptic reviewed, in both windows."""
    prev = load_traces(OUT / "prev-tool_sceptic" / "tool_sceptic")
    dates = games.date.astype(str)
    days = sorted(dates[(dates >= WINDOWS["test"][0]) & (dates <= WINDOWS["test"][1])].unique())
    test = [t for p in from_runs(games, days) for t in p["traces"]]
    return [("prev", t) for t in prev if "sceptic" in t] + [("test", t) for t in test if "sceptic" in t]


def cached_sceptic_prompts() -> dict:
    """(date, away, home, minutes to tip, side, price, estimate_p) -> list of cached reasoner sceptic entries."""
    from agents.llm_client import CACHE_DIR
    index = {}
    for p in CACHE_DIR.glob("*/*.json"):
        d = json.loads(p.read_text())
        if d.get("role") != "sceptic" or d.get("model") != REASONER:
            continue
        m = HEAD.search(d["prompt"])
        lines = d["prompt"].split("\n")
        ev = json.loads(lines[lines.index("EVIDENCE") + 1])
        e = ev["code_computed_edge"]
        if abs(e.get("min_edge", 0) - MIN_EDGE) > 1e-9:
            continue
        key = (m.group(3), m.group(1), m.group(2), int(m.group(4)), e["side"], round(e["price"], 4),
               round(float(e["estimate_p"]), 4))
        index.setdefault(key, []).append(d)
    return index


def labelled_set(tables: dict) -> pd.DataFrame:
    from agents.llm_client import parse_json
    from replay import Order, Replay
    games = tables["games"]
    g = games.set_index("game_id")
    index = cached_sceptic_prompts()
    rp = Replay(tables, lambda *a: [])
    rows = []
    for window, t in reviewed_traces(games):
        gid, now = t["game_id"], pd.Timestamp(t["as_of"])
        tip = g.tip_time[gid]
        cand = t.get("candidate") or t.get("order")
        if cand is None:
            raise SystemExit(f"approved proposal without an order: {gid} {now} {t['status']}")
        est = round(float(t["proposal"]["estimate_p"]), 4)
        key = (str(g.date[gid]), g.away_team[gid], g.home_team[gid], int(round((tip - now).total_seconds() / 60)),
               cand["side"], round(cand["price"], 4), est)
        hits = index.get(key, [])
        verdicts = {(parse_json(h["text"]) or {}).get("verdict") for h in hits}
        if not hits or len({h["prompt"] for h in hits}) > 1:
            raise SystemExit(f"{len(hits)} cached sceptic prompts for {key}")
        fill, why = rp._fill(Order(cand["ticker"], cand["side"], 0.0, STAKE, "counterfactual"), now, tip)
        if fill is None:
            raise SystemExit(f"no counterfactual fill for {key}: {why}")
        s = rp._settle({**fill, "market_ticker": cand["ticker"], "side": cand["side"], "game_id": gid}, tip)
        approved = t["sceptic"]["verdict"] == "approve"
        assert verdicts <= {t["sceptic"]["verdict"], None}, (key, verdicts)
        rows.append({"window": window, "date": str(g.date[gid]), "game_id": gid, "as_of": str(now),
                     "game": f"{g.away_team[gid]} @ {g.home_team[gid]}", "team": t["proposal"].get("team"),
                     "ticker": cand["ticker"], "side": cand["side"], "price": s["price"], "contracts": s["contracts"],
                     "fee": s["fee"], "close": s["close_price"], "clv": s["clv"], "clv_usd": s["clv"] * s["contracts"],
                     "clv_usd_after_fee": s["clv"] * s["contracts"] - s["fee"], "pnl": s["pnl"],
                     "gap": t.get("gap"), "estimate_p": est, "orig_approve": approved, "orig_status": t["status"],
                     "prompt": hits[0]["prompt"]})
    df = pd.DataFrame(rows)
    check_against_fills(df, games)
    return df


def check_against_fills(df: pd.DataFrame, games: pd.DataFrame):
    """Counterfactual fills of approved proposals must reproduce the runs' real fills."""
    dates = games.date.astype(str)
    days = sorted(dates[(dates >= WINDOWS["test"][0]) & (dates <= WINDOWS["test"][1])].unique())
    fills = [p["fills"] for p in from_runs(games, days) if len(p["fills"])]
    fills.append(pd.read_parquet(OUT / "prev-tool_sceptic" / "tool_sceptic" / "fills.parquet"))
    real = pd.concat(fills, ignore_index=True)
    kept = df[df.orig_approve].merge(real, left_on=["game_id", "ticker"], right_on=["game_id", "market_ticker"],
                                     suffixes=("", "_real"))
    assert len(kept) == len(real) == df.orig_approve.sum(), (len(kept), len(real), df.orig_approve.sum())
    for a, b in (("price", "price_real"), ("contracts", "contracts_real"), ("clv", "clv_real"), ("pnl", "pnl_real")):
        assert np.allclose(kept[a], kept[b]), (a, kept[[a, b]])


# ---------------- scoring ----------------

def api_answers(timeout=45.0) -> bool:
    """True if DeepSeek starts streaming a tiny reply within `timeout` s (when overloaded it sends only keep-alives)."""
    import os
    import time
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    t0 = time.time()
    body = {"model": REASONER, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 3, "stream": True}
    try:
        with httpx.stream("POST", "https://api.deepseek.com/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                          timeout=httpx.Timeout(30, connect=10)) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    return True
                if time.time() - t0 > timeout:
                    return False
    except httpx.HTTPError:
        return False
    return False


def wait_for_api(every=120, tries=90):
    import time
    for i in range(tries):
        if api_answers():
            print(f"DeepSeek answering ({time.strftime('%H:%M')})", flush=True)
            return
        print(f"DeepSeek not answering ({time.strftime('%H:%M')}), retry in {every} s", flush=True)
        time.sleep(every)
    raise SystemExit("DeepSeek did not answer")


class StreamingDeepSeek:
    """DeepSeekBackend's request body, streamed, with a wall-clock cap per attempt.

    During an overload DeepSeek answers 200 and then sends only keep-alives, which defeat a plain read timeout, so an
    attempt fails if no content or reasoning token arrives within FIRST_TOKEN_S or it runs past ATTEMPT_S."""

    def __init__(self, model: str = REASONER, retries: int = RETRIES, max_tokens: int | None = None):
        import httpx
        from agents.llm_client import DeepSeekBackend
        base = DeepSeekBackend(model)
        self._body_of, self.client = base._body, base.client
        self.model, self.retries, self.max_tokens = model, retries, max_tokens
        self.client.timeout = httpx.Timeout(READ_S, connect=CONNECT_S, write=30.0, pool=30.0)

    def _attempt(self, body: dict):
        import time
        t0, first, content, reasoning, usage = time.time(), None, [], 0, {}
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise _Retry(r.status_code, f"HTTP {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                elapsed = time.time() - t0
                if elapsed > ATTEMPT_S or (first is None and elapsed > FIRST_TOKEN_S):
                    raise _Retry(None, f"stalled after {elapsed:.0f} s ({'no token' if first is None else 'slow'})")
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        first = first or elapsed
                    content.append(d.get("content") or "")
                    reasoning += len(d.get("reasoning_content") or "")
        if first is None:
            raise _Retry(None, "stream ended without tokens")
        details = usage.get("completion_tokens_details") or {}
        return "".join(content), {"tokens_in": int(usage.get("prompt_tokens", 0) or 0),
                                  "tokens_out": int(usage.get("completion_tokens", 0) or 0),
                                  "reasoning_tokens": int(details.get("reasoning_tokens", 0) or 0),
                                  "reasoning_chars": reasoning}

    def generate(self, system: str, prompt: str, meta=None):
        import random
        import time
        import httpx
        body = {**self._body_of(system, prompt), "stream": True, "stream_options": {"include_usage": True}}
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                return self._attempt(body)
            except _Retry as exc:
                code, err = exc.code, str(exc)
            except httpx.TransportError as exc:
                code, err = None, f"{exc.__class__.__name__}: {exc}"
            if (code is not None and code not in (429, 500, 502, 503, 504)) or attempt == self.retries:
                raise RuntimeError(err)
            if code == 429:
                print("rate limited (429)", flush=True)
            time.sleep(delay * random.uniform(0.5, 1.5))
            delay = min(delay * 2, 60)


class _Retry(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def role(k: int) -> str:
    """Sample k of the same prompt; distinct roles give distinct cache keys (the reasoner samples)."""
    return "sceptic_balanced" if k == 0 else f"sceptic_balanced#{k + 1}"


def balanced_prompt(prompt: str) -> str:
    assert prompt.endswith("Reply with your JSON verdict.")
    return prompt[: -len("Reply with your JSON verdict.")] + "Reply with your JSON score."


def score_all(df: pd.DataFrame, models: dict, live: bool, workers: int) -> pd.DataFrame:
    from agents.llm_client import REASONER_MAX_TOKENS, CachedLLM
    from agents.tool_agent import SCEPTIC_BALANCED_SYSTEM, parse_balanced

    class CacheOnly:
        def __init__(self, model):
            self.model = model

        def generate(self, system, prompt, meta=None):
            raise RuntimeError("cache miss (run with --score)")

    out = df.copy()
    for model, n in models.items():
        llm = CachedLLM(StreamingDeepSeek(model) if live else CacheOnly(model))
        long = CachedLLM(StreamingDeepSeek(model, max_tokens=LONG_MAX_TOKENS) if live else CacheOnly(model))
        jobs = [(i, k) for i in range(len(df)) for k in range(n)]
        truncated = []

        def one(job):
            i, k = job
            prompt = balanced_prompt(df.prompt.iloc[i])
            r = llm.ask(role(k), SCEPTIC_BALANCED_SYSTEM, prompt)
            p = parse_balanced(r)
            if p.get("invalid") and not r["error"] and r["tokens_out"] >= REASONER_MAX_TOKENS:
                truncated.append(i)        # reasoning hit max_tokens before any answer: re-ask with a larger cap
                r = long.ask(role(k) + LONG_ROLE, SCEPTIC_BALANCED_SYSTEM, prompt)
                p = parse_balanced(r)
            return i, k, (np.nan if r["error"] else p["score"]), bool(p.get("invalid")) and not r["error"], \
                r["cached"], r["cost"], p.get("reason", "")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(one, jobs))
        fresh = [x for x in res if not x[4]]
        print(f"{model}: {len(res)} calls, {len(fresh)} new, est. ${sum(x[5] for x in fresh):.2f}, "
              f"{sum(np.isnan(x[2]) for x in res)} missing, {sum(x[3] for x in res)} invalid, "
              f"{len(truncated)} re-asked with max_tokens {LONG_MAX_TOKENS}", flush=True)
        tag = "reasoner"
        out["truncated_at_8k"] = out.index.isin(out.index[truncated])
        for i, k, sc, inv, *_rest in res:
            out.loc[out.index[i], f"{tag}_s{k + 1}"] = sc
            out.loc[out.index[i], f"{tag}_s{k + 1}_invalid"] = inv
            if k == 0:
                out.loc[out.index[i], f"{tag}_reason"] = _rest[-1]
        cols = [f"{tag}_s{k + 1}" for k in range(n)]
        out[f"score_{tag}"] = out[cols].mean(axis=1, skipna=False)
    return out


# ---------------- evaluation ----------------

def day_boot_idx(days: pd.Series, reps=REPS, seed=SEED):
    """Day-clustered bootstrap: for each replicate, how many times each row's day is drawn."""
    uniq = np.array(sorted(days.unique()))
    pos = pd.Index(uniq).get_indexer(days)
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(reps, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
    return counts[:, pos]                                           # (reps, rows)


def ci(v):
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def auc(score, y) -> float:
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def random_band(clv_usd: np.ndarray, k: int, draws=RANDOM_DRAWS, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + k)
    if k == 0:
        return np.zeros(draws)
    return np.array([clv_usd[rng.choice(len(clv_usd), k, replace=False)].sum() for _ in range(draws)])


def filter_stats(w: pd.DataFrame, keep: np.ndarray, base: np.ndarray | None, boot) -> dict:
    c, cf, pnl, clv = (w[x].to_numpy(float) for x in ("clv_usd", "clv_usd_after_fee", "pnl", "clv"))
    pos = clv > 0
    k = int(keep.sum())
    r = {"kept": k, "keep_rate": k / len(w), "clv_usd": c[keep].sum(), "clv_usd_after_fee": cf[keep].sum(),
         "mean_clv_c": 100 * clv[keep].mean() if k else np.nan, "pnl": pnl[keep].sum(),
         "wrong_veto": float(pos[~keep].mean()) if (~keep).any() else np.nan,
         "wrong_approve": float((~pos[keep]).mean()) if k else np.nan}
    bs = boot @ (c * keep)
    r["clv_usd_lo"], r["clv_usd_hi"] = ci(bs)
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = 100 * (boot @ (clv * keep)) / (boot @ keep.astype(float))
    r["mean_clv_lo"], r["mean_clv_hi"] = ci(bm)
    r["pnl_lo"], r["pnl_hi"] = ci(boot @ (pnl * keep))
    if base is not None:
        d = boot @ (c * keep) - boot @ (c * base)
        r["diff_vs_current"] = c[keep].sum() - c[base].sum()
        r["diff_lo"], r["diff_hi"] = ci(d)
    band = random_band(c, k)
    r["random_mean"], r["random_lo"], r["random_hi"] = band.mean(), *np.percentile(band, [2.5, 97.5])
    tie = np.isclose(band, r["clv_usd"], atol=1e-6)
    r["random_pctile"] = 100 * (((band < r["clv_usd"]) & ~tie).mean() + 0.5 * tie.mean())
    return r


def tune(w: pd.DataFrame, col: str, min_kept=MIN_KEPT) -> float:
    """tau maximising CLV $ of {score >= tau} with at least min_kept kept; ties go to the higher tau."""
    s = w[col].to_numpy(float)
    best = None
    for tau in sorted(set(s[s > 0])):
        keep = s >= tau
        if keep.sum() < min_kept:
            continue
        v = w.clv_usd.to_numpy(float)[keep].sum()
        if best is None or v >= best[0]:
            best = (v, tau)
    return best[1] if best else 1.0


def curve(w: pd.DataFrame, col: str) -> pd.DataFrame:
    s, c = w[col].to_numpy(float), w.clv_usd.to_numpy(float)
    rows = [{"tau": tau, "kept": int((s >= tau).sum()), "clv_usd": c[s >= tau].sum()} for tau in sorted(set(s))]
    return pd.DataFrame(rows).assign(keep_rate=lambda x: x.kept / len(w))


def evaluate(df: pd.DataFrame, scorers: list) -> tuple:
    rows, taus, aucs = [], {}, {}
    for train, test in (("prev", "test"), ("test", "prev")):
        w = df[df.window == test].reset_index(drop=True)
        boot = day_boot_idx(w.date)
        base = w.orig_approve.to_numpy()
        y = w.clv.to_numpy() > 0
        add = lambda name, keep, tau=np.nan, b=base: rows.append(
            {"direction": f"tune {train} -> test {test}", "window": test, "filter": name, "tau": tau,
             **filter_stats(w, keep, b, boot)})
        add("Keep all", np.ones(len(w), bool))
        add("Current sceptic (approve/reject)", base, b=None)
        add("Oracle (CLV > 0)", y)
        for col, label in scorers:
            tau = tune(df[df.window == train], col)
            taus[(col, train)] = tau
            add(f"Balanced {label}, tau tuned on {train}", w[col].to_numpy() >= tau, tau)
            add(f"Balanced {label}, tau = 0.5 (untuned)", w[col].to_numpy() >= 0.5, 0.5)
            s = w[col].to_numpy(float)
            bs = [auc(s[np.repeat(np.arange(len(w)), b)], y[np.repeat(np.arange(len(w)), b)]) for b in boot[:500]]
            aucs[(col, test)] = (auc(s, y), *ci(np.array(bs)))
    # Pooled leave-one-window-out: each window filtered with the tau tuned on the other one.
    w = df.reset_index(drop=True)
    boot = day_boot_idx(w.date)
    base = w.orig_approve.to_numpy()
    for name, keep in (("Keep all", np.ones(len(w), bool)), ("Current sceptic (approve/reject)", base),
                       ("Oracle (CLV > 0)", w.clv.to_numpy() > 0)):
        rows.append({"direction": "pooled leave-one-window-out", "window": "both", "filter": name, "tau": np.nan,
                     **filter_stats(w, keep, None if name.startswith("Current") else base, boot)})
    for col, label in scorers:
        other = {"prev": "test", "test": "prev"}
        tau_row = w.window.map(lambda x: taus[(col, other[x])]).to_numpy()
        rows.append({"direction": "pooled leave-one-window-out", "window": "both",
                     "filter": f"Balanced {label}, tau from the other window", "tau": np.nan,
                     **filter_stats(w, w[col].to_numpy() >= tau_row, base, boot)})
        s, y = w[col].to_numpy(float), w.clv.to_numpy() > 0
        bs = [auc(s[np.repeat(np.arange(len(w)), b)], y[np.repeat(np.arange(len(w)), b)]) for b in boot[:500]]
        aucs[(col, "both")] = (auc(s, y), *ci(np.array(bs)))
    return pd.DataFrame(rows), taus, aucs


def calibration(df: pd.DataFrame, col: str) -> pd.DataFrame:
    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0001]
    b = pd.cut(df[col], bins, right=False, labels=["0-0.2", "0.2-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.8", "0.8-1"])
    return df.assign(bucket=b, pos=df.clv > 0, pos_fee=df.clv_usd_after_fee > 0).groupby("bucket", observed=True).agg(
        n=("clv", "size"), mean_score=(col, "mean"), share_clv_pos=("pos", "mean"),
        share_clv_pos_after_fee=("pos_fee", "mean"), mean_clv_c=("clv", lambda x: 100 * x.mean())).reset_index()


def chart(df: pd.DataFrame, col: str, tau: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = df[df.window == "test"].reset_index(drop=True)
    c = w.clv_usd.to_numpy(float)
    cv = curve(w, col)
    ks = np.arange(0, len(w) + 1)
    bands = np.array([np.percentile(random_band(c, k, 4000), [2.5, 50, 97.5]) for k in ks])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(ks / len(w), bands[:, 0], bands[:, 2], color="0.85", label="Random filter, 95% band")
    ax.plot(ks / len(w), ks * c.mean(), color="0.5", lw=1, ls="--", label="Random filter, mean")
    ax.plot(cv.keep_rate, cv.clv_usd, color="C0", marker=".", ms=3, lw=1.2, label="Balanced score, sweep of tau")
    k = (w[col] >= tau).sum()
    ax.scatter([k / len(w)], [c[w[col].to_numpy() >= tau].sum()], color="C0", s=70, zorder=5, edgecolor="k",
               label=f"Balanced, tau = {tau:.2f} tuned on Dec-Jan")
    base = w.orig_approve.to_numpy()
    ax.scatter([base.mean()], [c[base].sum()], color="C3", marker="s", s=60, zorder=5, edgecolor="k",
               label="Current sceptic")
    ax.scatter([(c > 0).mean()], [c[c > 0].sum()], color="C2", marker="^", s=60, zorder=5, edgecolor="k",
               label="Oracle (CLV > 0)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Keep rate (share of reviewed proposals approved)")
    ax.set_ylabel("CLV $ of kept trades")
    ax.set_title(f"Test window Feb - Apr 2026: {len(w)} reviewed proposals, $20 stake")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------- report ----------------

def write_report(df, table, taus, aucs, scorers, cal, args):
    f2 = lambda v: f"{v:+.2f}"
    md = ["# Approach A: balanced scored sceptic vs the current approve/reject sceptic", "",
          "**Setup.** The current sceptic (deepseek-reasoner, one call) argues only that the edge is priced in and "
          "answers approve/reject. The balanced sceptic (`sceptic_mode=\"balanced\"` in `agents/tool_agent.py`, "
          "opt-in; the default is unchanged) states the strongest case for and against the trade and returns "
          "`score` = its probability that the trade beats the closing line after fees; it approves iff score ≥ τ. "
          "Invalid output or banned wording scores 0 (fail closed). It saw exactly the user prompt the current "
          "sceptic saw (read back from the LLM cache); only the system prompt and the last line differ.", "",
          f"**Labelled set.** Every proposal the current sceptic reviewed: {len(df)} "
          f"({(df.window == 'prev').sum()} in Dec 2025 - Jan 2026 with walk-forward refit models, "
          f"{(df.window == 'test').sum()} in the Feb - Apr 2026 test window). Each gets a counterfactual fill at its "
          f"decision time from `Replay._fill`/`_settle`: ${STAKE:.0f} stake at the ask, CLV per contract = side's mid "
          "at tip − price paid (the fee is in P&L, not in CLV, as for real fills); CLV $ = CLV × contracts. "
          f"Recomputed CLV matches the real fills of the {int(df.orig_approve.sum())} approved trades.", "",
          "**Scores.** deepseek-reasoner (the model of the current sceptic runs), one pass per proposal, streamed "
          f"(attempt capped at {ATTEMPT_S:.0f} s, no token within {FIRST_TOKEN_S:.0f} s = failed attempt, "
          f"{RETRIES} retries). {int(df.get('truncated_at_8k', pd.Series(dtype=bool)).sum())} replies spent the whole "
          f"8,192-token budget reasoning and returned no answer; those were re-asked once with max_tokens "
          f"{LONG_MAX_TOKENS:,} (any still invalid score 0, fail closed). τ is tuned on one window to maximise "
          f"CLV $ of the kept set with at least {MIN_KEPT} kept (ties → higher τ), then fixed on the other window. "
          f"95% CIs: day-clustered bootstrap ({REPS} replicates, seed {SEED}); random filter: {RANDOM_DRAWS:,} random "
          "subsets of the same size; percentile = share of random subsets with lower CLV $. Wrong veto = vetoed with "
          "CLV > 0; wrong approve = kept with CLV ≤ 0.", ""]
    for d in ("prev", "test"):
        w = df[df.window == d]
        md.append(f"- {WINDOW_NAMES[d]}: {len(w)} proposals, keep-all CLV $ {w.clv_usd.sum():+.2f}, "
                  f"{int((w.clv > 0).sum())} with CLV > 0, **oracle ceiling CLV $ {w.clv_usd[w.clv > 0].sum():+.2f}**; "
                  f"current sceptic kept {int(w.orig_approve.sum())}.")
    md.append(f"- Both windows: oracle ceiling CLV $ {df.clv_usd[df.clv > 0].sum():+.2f} "
              f"({int((df.clv > 0).sum())} of {len(df)} proposals).")
    md += ["", "Chosen τ: " + "; ".join(f"{label} tuned on {WINDOW_NAMES[tr]} → τ = {taus[(col, tr)]:.3f}"
                                         for col, label in scorers for tr in ("prev", "test")), ""]
    md += ["AUC of the score vs CLV > 0 (95% day-clustered CI): " + "; ".join(
        f"{label} {WINDOW_NAMES.get(win, 'pooled')} {a:.2f} [{lo:.2f}, {hi:.2f}]"
        for (col, win), (a, lo, hi) in aucs.items() for c2, label in scorers if c2 == col), ""]
    for direction, sub in table.groupby("direction", sort=False):
        md += [f"## {direction}", "",
               "| Filter | τ | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] "
               "| Random same size: mean [95% band] | Pctile vs random | Wrong veto | Wrong approve | CLV $ after fee "
               "| P&L [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in sub.itertuples():
            diff = f"{f2(r.diff_vs_current)} [{f2(r.diff_lo)}, {f2(r.diff_hi)}]" if pd.notna(
                getattr(r, "diff_vs_current", np.nan)) else "–"
            md.append(f"| {r.filter} | {'' if pd.isna(r.tau) else f'{r.tau:.3f}'} | {r.kept} | {r.keep_rate:.0%} | "
                      f"{f2(r.clv_usd)} [{f2(r.clv_usd_lo)}, {f2(r.clv_usd_hi)}] | "
                      + (f"{r.mean_clv_c:+.2f} [{r.mean_clv_lo:+.2f}, {r.mean_clv_hi:+.2f}]" if r.kept else "–")
                      + f" | {diff} | {f2(r.random_mean)} [{f2(r.random_lo)}, {f2(r.random_hi)}] | "
                      f"{r.random_pctile:.0f} | " + (f"{r.wrong_veto:.0%}" if pd.notna(r.wrong_veto) else "–") +
                      " | " + (f"{r.wrong_approve:.0%}" if pd.notna(r.wrong_approve) else "–") +
                      f" | {f2(r.clv_usd_after_fee)} | {f2(r.pnl)} [{f2(r.pnl_lo)}, {f2(r.pnl_hi)}] |")
        md.append("")
    for (col, label), c in zip(scorers, cal):
        md += [f"Calibration, {label}, both windows (score vs realised share with positive CLV):", "",
               "| Score bucket | n | Mean score | Share CLV > 0 | Share CLV $ − fee > 0 | Mean CLV ¢ |",
               "| --- | --- | --- | --- | --- | --- |"]
        md += [f"| {r.bucket} | {r.n} | {r.mean_score:.2f} | {r.share_clv_pos:.0%} | {r.share_clv_pos_after_fee:.0%} "
               f"| {r.mean_clv_c:+.2f} |" for r in c.itertuples()]
        md.append("")
    for col, label in scorers:
        s = df[col]
        md.append(f"{label} score distribution: mean {s.mean():.2f}, sd {s.std():.2f}, min {s.min():.2f}, "
                  f"max {s.max():.2f}, {int((s >= 0.5).sum())} of {len(s)} at ≥ 0.5.")
    if "reasoner_s2" in df:
        r = df[["reasoner_s1", "reasoner_s2"]].corr().iloc[0, 1]
        md.append(f"Reasoner sample-to-sample correlation {r:.2f}; mean |difference| "
                  f"{(df.reasoner_s1 - df.reasoner_s2).abs().mean():.3f}.")
    pooled = table[(table.direction == "pooled leave-one-window-out") & table["filter"].str.startswith("Balanced")]
    if len(pooled):
        r = pooled.iloc[0]
        a, lo, hi = aucs[(scorers[0][0], "both")]
        md += ["", "## Reading", "",
               f"Out of sample (each window filtered with τ from the other), the balanced sceptic keeps {r.kept} "
               f"proposals for CLV $ {r.clv_usd:+.2f} (percentile {r.random_pctile:.0f} among random filters of "
               f"the same size), and {r.diff_vs_current:+.2f} [{r.diff_lo:+.2f}, {r.diff_hi:+.2f}] vs the current "
               f"sceptic. Pooled AUC of the score vs CLV > 0 is {a:.2f} [{lo:.2f}, {hi:.2f}]: "
               + ("the score does not rank proposals better than chance, so " if lo <= 0.5 else "")
               + ("it does not beat the current approve/reject sceptic." if r.diff_hi < 0 or r.diff_vs_current <= 0
                  else "it improves on the current sceptic.")]
    if args.balance_before is not None and args.balance_after is not None:
        md += ["", f"DeepSeek spend for this experiment (account balance before minus after): "
                   f"${args.balance_before - args.balance_after:.2f} (${args.balance_before:.2f} → "
                   f"${args.balance_after:.2f})."]
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="call DeepSeek for missing scores (cached)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wait-for-api", action="store_true", help="poll until DeepSeek answers before scoring")
    ap.add_argument("--balance-before", type=float)
    ap.add_argument("--balance-after", type=float)
    args = ap.parse_args()
    tables = load_frozen()
    df = labelled_set(tables)
    print(f"labelled set: {len(df)} ({df.window.value_counts().to_dict()})", flush=True)
    if args.score and args.wait_for_api:
        wait_for_api()
    df = score_all(df, SAMPLES, args.score, args.workers)
    WORK.mkdir(parents=True, exist_ok=True)
    df.drop(columns="prompt").to_csv(WORK / "labelled_scores.csv", index=False)
    scorers = [("score_reasoner", "reasoner")]
    if df[[c for c, _ in scorers]].isna().any().any():
        raise SystemExit("missing scores; run with --score")
    table, taus, aucs = evaluate(df, scorers)
    cal = [calibration(df, col) for col, _ in scorers]
    table.to_csv(RESULTS / f"{STEM}.csv", index=False)
    df.drop(columns=["prompt"]).to_csv(RESULTS / f"{STEM}_proposals.csv", index=False)
    chart(df, "score_reasoner", taus[("score_reasoner", "prev")], RESULTS / f"{STEM}.png")
    md = write_report(df, table, taus, aucs, scorers, cal, args)
    (RESULTS / f"{STEM}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
