"""Two opt-in LLM upgrades evaluated on the 182 sceptic-reviewed proposals (labelled set of
evaluation/sceptic_robustness.py: exact cached sceptic prompt, original verdict, counterfactual $20 fill's CLV / P&L).

  votes   Majority-vote sceptic (ToolAgent(sceptic_votes=5)): VOTES fresh deepseek-flash thinking-on draws per
          proposal (roles sceptic_rep1..VOTES; rep1-3 are the cached sceptic_robustness draws), concurrent.
          Primary decision = majority of rep1..rep5. Stability = Cohen's kappa between two independent 5-vote
          majorities resampled from each proposal's VOTES collected votes (no second pass).
  prob    LLM probability (PROB_SYSTEM): p_home from the same as-of evidence (quotes, 24 h anchor, 6 h history,
          inactive lists, M4 shift), on the 182 proposals' games + N_RANDOM other games. Blend
          p = anchor + w (p_llm - anchor); w tuned on Dec-Jan Brier, tested Feb-Apr.
  memo    Outcome recall of the probability model on 20 games, no evidence.

    python -m evaluation.llm_upgrade --votes --prob --memo --workers 16
    python -m evaluation.llm_upgrade --report
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "llm_upgrade"
RESULTS = ROOT / "evaluation" / "results"
STEM = "llm_upgrade"
# Upgrade 1 primary: flash thinking on (legacy deepseek-reasoner) — matches the published sceptic.
# Flash thinking-off temp>0 was tried; it approved 0/1274 (see results). Roles sceptic_rep{k} reuse
# the sceptic_robustness cache for k=1..3.
VOTE_MODEL, VOTE_THINKING, VOTE_TEMP = "deepseek-reasoner", None, None
# Upgrade 2: flash thinking off (v4-pro skipped for cost/speed on the 182-candidate set).
PROB_MODEL = ("deepseek-flash", False)
VOTES, N_VOTE = 7, 5
SEED, BOOT = 7606, 2000
TEST_START = "2026-02-01"
N_RANDOM = 60
W_GRID = np.round(np.arange(0, 1.01, 0.1), 2)
SINGLE_ORIG_CLV, SINGLE_RERUNS, SINGLE_KAPPA = 7.38, (1.71, 2.05, 6.04), 0.15
SPEND = []


def labelled() -> pd.DataFrame:
    import numpy.core.multiarray
    import numpy.core.numeric
    sys.modules.setdefault("numpy._core", np.core)          # pickle written under numpy 2
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    from evaluation.sceptic_robustness import labelled as lab
    df = lab()
    df["date"] = df.date.astype(str)
    df["game_id"] = df.game_id.astype(str)
    return df


def call_all(jobs: list, workers: int, on_batch=None) -> list:
    """jobs: (key, role, system, prompt, model, thinking) -> [(key, reply)], concurrent via as_completed.

    One DeepSeekBackend per worker (each has its own rate-limit lock), so HTTP streams run in parallel.
    Retries 429 / 5xx / stalls with the client's jittered backoff. on_batch(results_so_far) every 25 replies."""
    import threading
    from concurrent.futures import as_completed
    from agents.llm_client import CachedLLM, DeepSeekBackend
    local = threading.local()
    lock = threading.Lock()

    def llm_for(model, thinking, temperature):
        bag = getattr(local, "llms", None)
        if bag is None:
            bag = local.llms = {}
        key = (model, thinking, temperature)
        if key not in bag:
            bag[key] = CachedLLM(DeepSeekBackend(model, thinking=thinking, max_retries=3,
                                                 temperature=temperature))
        return bag[key]

    def one(job):
        # job: (key, role, system, prompt, model, thinking[, temperature])
        key, role, system, prompt, model, thinking = job[:6]
        temperature = job[6] if len(job) > 6 else (None if thinking else 0.0)
        r = llm_for(model, thinking, temperature).ask(role, system, prompt)
        if not r["cached"]:
            with lock:
                SPEND.append(r["cost"])
        return key, r
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            out.append(fut.result())
            if len(out) % 25 == 0 or len(out) == len(jobs):
                print(f"  {len(out)}/{len(jobs)} done, new spend ${sum(SPEND):.2f}", flush=True)
                if on_batch is not None:
                    on_batch(out)
    return out


# ---------------- upgrade 1: majority-vote sceptic ----------------

def run_votes(df, workers):
    """Majority-vote draws. Default: thinking-on reasoner, roles sceptic_rep{k} (k=1..3 hit the robustness cache)."""
    from agents.tool_agent import SCEPTIC_SYSTEM, parse_verdict
    path = WORK / "votes.csv"
    # Prefer the completed reasoner artifact when present.
    src = WORK / "votes_reasoner.csv"
    if src.exists() and not path.exists():
        path = src
    base = pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=["row", "vote", "verdict", "invalid", "error", "cached", "cost", "model"])
    tag = "deepseek-reasoner"
    if "model" in base.columns and len(base):
        base = base[base.model.fillna(tag) == tag]
    done = {(int(r.row), int(r.vote)) for r in base.itertuples()
            if r.error is None or (isinstance(r.error, float) and np.isnan(r.error)) or r.error == ""}
    print(f"votes resume: {len(done)} already ok ({tag})", flush=True)
    jobs = [((i, k), f"sceptic_rep{k}", SCEPTIC_SYSTEM, df.prompt[i], VOTE_MODEL, VOTE_THINKING)
            for k in range(1, VOTES + 1) for i in df.index if (i, k) not in done]
    print(f"votes remaining: {len(jobs)}", flush=True)

    def persist(batch):
        cur = pd.read_csv(WORK / "votes_reasoner.csv") if (WORK / "votes_reasoner.csv").exists() else base
        extra = []
        for (i, k), r in batch:
            p = parse_verdict(r)
            extra.append({"row": i, "vote": k, "verdict": p["verdict"], "invalid": bool(p.get("invalid")),
                          "error": r["error"], "cached": r["cached"], "cost": r["cost"], "model": tag})
        if not extra:
            return
        out = pd.concat([cur, pd.DataFrame(extra)], ignore_index=True).drop_duplicates(["row", "vote"], keep="last")
        out.to_csv(WORK / "votes_reasoner.csv", index=False)
        out.to_csv(WORK / "votes.csv", index=False)

    for start in range(0, len(jobs), 16):
        chunk = jobs[start:start + 16]
        print(f"votes batch {start}-{start + len(chunk)} / {len(jobs)}", flush=True)
        call_all(chunk, min(workers, 4), on_batch=persist)
    out_path = WORK / "votes_reasoner.csv"
    return pd.read_csv(out_path) if out_path.exists() else base


def vote_matrix(votes: pd.DataFrame, idx) -> np.ndarray:
    """rows x VOTES approve matrix (errors count as rejects: fail closed)."""
    w = votes.assign(a=votes.verdict.eq("approve") & votes.error.isna()).pivot(index="row", columns="vote", values="a")
    return w.reindex(idx).fillna(False).to_numpy(bool)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def resampled_kappa(app: np.ndarray, n: int, reps=500, seed=SEED) -> tuple:
    """Kappa between two independent n-vote majorities, each vote drawn with replacement from a proposal's votes."""
    rng = np.random.default_rng(seed)
    ks, kept = [], []
    for _ in range(reps):
        dec = []
        for _ in range(2):
            pick = rng.integers(0, app.shape[1], size=(app.shape[0], n))
            dec.append(np.take_along_axis(app, pick, axis=1).sum(1) > n / 2)
        ks.append(cohen_kappa(*dec))
        kept.append(dec[0].sum())
    return float(np.nanmean(ks)), float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5)), \
        float(np.mean(kept))


def random_pct(clv: np.ndarray, value: float, k: int, draws=20000, seed=SEED) -> tuple:
    if k == 0:
        return float("nan"), 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((draws, len(clv))), axis=1)[:, :k]
    s = clv[idx].sum(1)
    return float((s < value).mean() * 100 + (s == value).mean() * 50), float(s.mean()), \
        float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def day_boot(values: np.ndarray, days: np.ndarray, reps=BOOT, seed=SEED) -> tuple:
    """Sum of values with a day-block bootstrap 95% CI (days with no kept trade count as zero)."""
    s = pd.Series(values).groupby(np.asarray(days)).sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    b = s[idx].sum(1)
    return float(s.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def votes_table(df, votes) -> tuple:
    app = vote_matrix(votes, df.index)
    frac5 = app[:, :N_VOTE].mean(1)
    maj = app[:, :N_VOTE].sum(1) > N_VOTE / 2
    frac_all = app.mean(1)
    rows = []
    for window, m in (("full Dec-Apr", np.ones(len(df), bool)), ("test Feb-Apr", (df.date >= TEST_START).to_numpy())):
        sub = df[m]
        clv, pnl, days = sub.clv_usd.to_numpy(), sub.pnl.to_numpy(), sub.date.to_numpy()
        sets = {"single sceptic (original)": (sub.orig_approve.to_numpy(), 1.0)}
        for k in (1, 2, 3):
            sets[f"single sceptic (fresh draw {k})"] = (app[m, k - 1], 1.0)
        sets[f"majority of {N_VOTE} votes"] = (maj[m], 1.0)
        sets[f"majority of {N_VOTE}, stake x approval fraction"] = (maj[m], frac5[m])
        sets[f"any approval, stake x approval fraction ({N_VOTE} votes)"] = (frac5[m] > 0, frac5[m])
        sets[f"majority of all {VOTES} votes"] = (app[m].sum(1) > VOTES / 2, 1.0)
        sets["perfect filter (CLV > 0)"] = (clv > 0, 1.0)
        sets["no filter (all proposals)"] = (np.ones(len(sub), bool), 1.0)
        for name, (keep, scale) in sets.items():
            wgt = keep * scale
            c, lo, hi = day_boot(clv * wgt, days)
            p, plo, phi = day_boot(pnl * wgt, days)
            pct, rmean, rlo, rhi = random_pct(clv, float((clv * keep).sum()), int(keep.sum()))
            rows.append({"window": window, "filter": name, "proposals": len(sub), "kept": int(keep.sum()),
                         "stake_units": round(float(wgt.sum()), 2), "clv_usd": round(c, 2), "clv_lo": round(lo, 2),
                         "clv_hi": round(hi, 2), "pnl_usd": round(p, 2), "pnl_lo": round(plo, 2),
                         "pnl_hi": round(phi, 2),
                         "random_pct": round(pct, 1) if np.all(np.asarray(scale) == 1) else np.nan,
                         "random_mean": round(rmean, 2), "random_lo": round(rlo, 2), "random_hi": round(rhi, 2)})
    t = pd.DataFrame(rows)
    k5 = resampled_kappa(app, N_VOTE)
    k1 = resampled_kappa(app, 1)
    pair = [cohen_kappa(app[:, a], app[:, b]) for a in range(VOTES) for b in range(a + 1, VOTES)]
    st = {"kappa_maj5": k5, "kappa_single": k1, "kappa_single_pairs_mean": float(np.nanmean(pair)),
          "approve_rate_per_vote": [round(float(x), 3) for x in app.mean(0)],
          "orig_approvals_maj5": int((maj & df.orig_approve.to_numpy()).sum()),
          "invalid_votes": int(votes.invalid.sum()), "errors": int(votes.error.notna().sum()),
          "frac_hist": np.bincount(app[:, :N_VOTE].sum(1), minlength=N_VOTE + 1).tolist(),
          "frac7_hist": np.bincount(app.sum(1), minlength=VOTES + 1).tolist()}
    per = df[["game", "date", "orig_approve", "clv_usd", "pnl"]].assign(approve_frac5=frac5, approve_frac7=frac_all,
                                                                       majority5=maj)
    return t, st, per


# ---------------- upgrade 2: LLM probability ----------------

def prob_games(df) -> pd.DataFrame:
    """The 182 proposals' games at their decision time + N_RANDOM other games (Dec-Apr, tip - 60 min)."""
    from evaluation.llm_agent_eval import load_frozen
    tables = load_frozen()
    g = tables["games"].assign(game_id=lambda d: d.game_id.astype(str))
    cand = df.sort_values("as_of").drop_duplicates("game_id")[["game_id", "as_of"]].assign(candidate=True)
    pool = g[(g.date.astype(str) >= df.date.min()) & (g.date.astype(str) <= df.date.max())
             & ~g.game_id.isin(cand.game_id) & g.home_pts.notna()]
    rnd = pool.sample(N_RANDOM, random_state=SEED)
    rnd = pd.DataFrame({"game_id": rnd.game_id, "as_of": pd.to_datetime(rnd.tip_time) - pd.Timedelta(minutes=60),
                        "candidate": False})
    out = pd.concat([cand.assign(as_of=pd.to_datetime(cand.as_of, utc=True)),
                     rnd.assign(as_of=pd.to_datetime(rnd.as_of, utc=True))], ignore_index=True)
    return out, tables


def prob_evidence(sample: pd.DataFrame, tables) -> pd.DataFrame:
    """Prompt + as-of numbers per game: anchor, M4 forecast, decision quotes, tip-off mids, outcome."""
    from agents.tool_agent import P_MAX, P_MIN, dumps, header
    from agents.tools import AsOfTools
    from forecast.api import Forecaster
    from replay import AsOf
    forecaster = Forecaster.load()
    games = tables["games"].assign(game_id=lambda d: d.game_id.astype(str)).set_index("game_id")
    index = {k: v.reset_index(drop=True) for k, v in tables["prices"].sort_values("ts").groupby("market_ticker")}
    rows = []
    for r in sample.itertuples():
        game = next(games.loc[[r.game_id]].reset_index().itertuples(index=False))
        now = pd.Timestamp(r.as_of)
        tools = AsOfTools(AsOf(tables, now, index), game, now, forecaster)
        if len(tools.teams) < 2 or not tools.fresh_teams():
            continue
        h = tools.home
        anchor = tools.get_anchor(h).get("anchor_mid")
        m4 = tools.m4_win_prob(tools.out_from_news())
        info = {"quotes": {t: tools.get_quote(t) for t in tools.teams},
                "anchors": {t: tools.get_anchor(t) for t in tools.teams},
                "price_history_6h": tools.get_price_history(h, 6), "news": tools.get_news(), "m4_win_model": m4}
        prompt = "\n".join(["DECISION CONTEXT", header(tools, game, False), f"The home team is {h}.", "",
                            "INFORMATION", dumps(info), "", "Reply with your JSON estimate."])
        tip = AsOfTools(AsOf(tables, game.tip_time, index), game, game.tip_time, None)
        close = {t: tip.get_quote(t).get("mid") for t in tip.teams}
        rows.append({"game_id": r.game_id, "date": str(game.date), "game": f"{game.away_team} @ {game.home_team}",
                     "candidate": r.candidate, "as_of": str(now), "home": h, "away": tools.away, "anchor": anchor,
                     "m4_p": min(max(anchor + m4["shift"][h], P_MIN), P_MAX), "mid_now": tools.get_quote(h)["mid"],
                     "home_win": float(game.home_pts > game.away_pts), "close_home": close.get(h),
                     "close_away": close.get(tools.away), "prompt": prompt, "_tools": tools})
    return pd.DataFrame(rows)


def run_prob(ev: pd.DataFrame, workers) -> pd.DataFrame:
    from agents.tool_agent import PROB_SYSTEM, parse_prob
    model, thinking = PROB_MODEL
    jobs = [(i, "prob_home", PROB_SYSTEM, ev.prompt[i], model, thinking) for i in ev.index]
    out = {}
    for i, r in call_all(jobs, workers):
        p = parse_prob(r)
        out[i] = {"p_llm": p["p_home"] if p else np.nan, "confidence": p["confidence"] if p else "invalid",
                  "llm_error": r["error"], "llm_cost": r["cost"], "llm_cached": r["cached"]}
    return ev.join(pd.DataFrame.from_dict(out, orient="index"))


def trade_game(tools, p_home, close_home, close_away, home_win, min_edge=0.04, stake=20.0):
    """Best route for p_home (either team) if its gap after fees clears min_edge; CLV vs the tip-off mid."""
    from replay import kalshi_fee
    best = None
    for team in tools.teams:
        p = p_home if team == tools.home else 1 - p_home
        r = tools.route(team, p)
        if "error" not in r and (best is None or r["gap_after_fees"] > best["gap_after_fees"]):
            best = r
    if best is None or not best["clears_min_edge"]:
        return None
    price = best["price"]
    team_close = close_home if best["team"] == tools.home else close_away
    opp_close = close_away if best["team"] == tools.home else close_home
    close = team_close if best["side"] == "yes" else (None if opp_close is None else 1 - opp_close)
    contracts = int(np.floor(stake / price + 1e-9))
    won = home_win if best["team"] == tools.home else 1 - home_win
    return {"team": best["team"], "side": best["side"], "price": price, "gap": best["gap_after_fees"],
            "contracts": contracts, "clv_usd": np.nan if close is None else (close - price) * contracts,
            "pnl": contracts * (won - price) - kalshi_fee(contracts, price)}


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_diff_ci(pa, pb, y, days, reps=BOOT, seed=SEED):
    d = pd.Series((np.asarray(pa) - y) ** 2 - (np.asarray(pb) - y) ** 2).groupby(np.asarray(days))
    s, n = d.sum().to_numpy(), d.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def prob_tables(ev: pd.DataFrame) -> tuple:
    from agents.tool_agent import blend_prob
    ev = ev[ev.p_llm.notna() & ev.anchor.notna()].copy()
    tune = ev[ev.date < TEST_START]
    grid = []
    for w in W_GRID:
        p = [blend_prob(a, q, w) for a, q in zip(tune.anchor, tune.p_llm)]
        grid.append({"w": w, "tune_brier": brier(p, tune.home_win), "tune_games": len(tune)})
    grid = pd.DataFrame(grid)
    w_star = float(grid.sort_values(["tune_brier", "w"]).w.iloc[0])
    ev["p_blend"] = [blend_prob(a, q, w_star) for a, q in zip(ev.anchor, ev.p_llm)]
    ev["p_llm_raw"] = ev.p_llm.clip(0.01, 0.99)
    acc, trades, tr_rows = [], [], []
    for window, m in (("full Dec-Apr", np.ones(len(ev), bool)), ("test Feb-Apr", (ev.date >= TEST_START).to_numpy()),
                      ("tune Dec-Jan", (ev.date < TEST_START).to_numpy())):
        s = ev[m]
        y = s.home_win.to_numpy()
        for name, col in (("market anchor (24 h)", "anchor"), ("market mid at decision", "mid_now"),
                          ("anchor + M4 shift (agent)", "m4_p"), ("LLM raw p_home", "p_llm_raw"),
                          (f"blend w={w_star:.1f}", "p_blend")):
            d = brier_diff_ci(s[col], s.anchor, y, s.date)
            d4 = brier_diff_ci(s[col], s.m4_p, y, s.date)
            acc.append({"window": window, "forecast": name, "games": len(s), "brier": round(brier(s[col], y), 4),
                        "minus_anchor": round(d[0], 4), "minus_anchor_lo": round(d[1], 4),
                        "minus_anchor_hi": round(d[2], 4), "minus_m4": round(d4[0], 4),
                        "minus_m4_lo": round(d4[1], 4), "minus_m4_hi": round(d4[2], 4)})
        for name, col in (("anchor + M4 shift (agent)", "m4_p"), ("LLM raw p_home", "p_llm_raw"),
                          (f"blend w={w_star:.1f}", "p_blend")):
            t = [trade_game(r._tools, getattr(r, col), r.close_home, r.close_away, r.home_win)
                 for r in s.itertuples()]
            keep = np.array([x is not None for x in t])
            clv = np.array([x["clv_usd"] if x else 0.0 for x in t], float)
            pnl = np.array([x["pnl"] if x else 0.0 for x in t], float)
            clv = np.nan_to_num(clv)
            c, clo, chi = day_boot(clv, s.date.to_numpy())
            p, plo, phi = day_boot(pnl, s.date.to_numpy())
            pct, rmean, rlo, rhi = random_trade_pct(s, int(keep.sum()), c)
            trades.append({"window": window, "forecast": name, "games": len(s), "trades": int(keep.sum()),
                           "clv_usd": round(c, 2), "clv_lo": round(clo, 2), "clv_hi": round(chi, 2),
                           "pnl_usd": round(p, 2), "pnl_lo": round(plo, 2), "pnl_hi": round(phi, 2),
                           "random_pct": round(pct, 1), "random_mean": round(rmean, 2),
                           "random_lo": round(rlo, 2), "random_hi": round(rhi, 2)})
            if window == "full Dec-Apr":
                for r, x in zip(s.itertuples(), t):
                    if x:
                        tr_rows.append({"forecast": name, "game_id": r.game_id, "date": r.date, **x})
    trades = pd.DataFrame(trades)
    return grid, w_star, pd.DataFrame(acc), trades, pd.DataFrame(tr_rows), ev


def random_trade_pct(s: pd.DataFrame, k: int, value: float, draws=20000, seed=SEED) -> tuple:
    """Random filter at a matched trade count: k distinct games from the window, a coin-flip side each (bought on the
    cheapest route, $20), kept CLV $ summed. Returns (percentile of value, random mean, 2.5%, 97.5%)."""
    if k == 0:
        return float("nan"), 0.0, 0.0, 0.0
    opts = np.array([[np.nan_to_num(trade_game(r._tools, ph, r.close_home, r.close_away, r.home_win,
                                               min_edge=-1.0)["clv_usd"]) for ph in (1.0, 0.0)]
                     for r in s.itertuples()])
    rng = np.random.default_rng(seed)
    games = np.argsort(rng.random((draws, len(s))), axis=1)[:, :k]
    side = rng.integers(0, 2, size=(draws, k))
    sums = opts[games, side].sum(1)
    return float((sums < value).mean() * 100 + (sums == value).mean() * 50), float(sums.mean()), \
        float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


# ---------------- memorisation probe ----------------

def run_memo(df, tables, workers, n=20) -> pd.DataFrame:
    from evaluation.sceptic_robustness import RECALL_SYSTEM
    games = tables["games"].assign(game_id=lambda d: d.game_id.astype(str)).set_index("game_id")
    rng = np.random.default_rng(SEED)
    idx = list(df.drop_duplicates("game_id").sample(n, random_state=SEED).index)
    jobs, meta = [], {}
    for i in idx:
        away, home = df.game[i].split(" @ ")
        q = (f"Who won the NBA game {away} @ {home} on {df.date[i]}, and what was the score? "
             "If you don't know, say unknown.")
        jobs.append((i, "recall", RECALL_SYSTEM, q, PROB_MODEL[0], PROB_MODEL[1]))
        g = games.loc[df.game_id[i]]
        meta[i] = (away, home, int(g.away_pts), int(g.home_pts))
    rows = []
    for i, r in call_all(jobs, workers):
        away, home, ap, hp = meta[i]
        actual = home if hp > ap else away
        m = r["json"] or {}
        win = str(m.get("winner", "unknown")).upper()
        known = win not in ("UNKNOWN", "", "NONE")
        try:
            sa, sh = int(m.get("away_pts")), int(m.get("home_pts"))
        except (TypeError, ValueError):
            sa = sh = None
        rows.append({"row": i, "game": df.game[i], "date": df.date[i], "actual_winner": actual,
                     "actual_score": f"{ap}-{hp}", "said_winner": win, "said_score": None if sa is None else f"{sa}-{sh}",
                     "known": known, "winner_correct": known and win == actual,
                     "score_exact": sa == ap and sh == hp, "cost": r["cost"], "error": r["error"],
                     "cached": r["cached"]})
    out = pd.DataFrame(rows)
    out.to_csv(WORK / "memo.csv", index=False)
    return out


# ---------------- report ----------------

def figure(vt, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = vt[vt.window == "full Dec-Apr"]
    keep = ["single sceptic (original)", "single sceptic (fresh draw 1)", "majority of 5 votes",
            "majority of 5, stake x approval fraction", "perfect filter (CLV > 0)"]
    rows = full[full["filter"].isin(keep)].set_index("filter").reindex(keep)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(rows))
    ax.bar(x, rows.clv_usd, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"],
           yerr=[rows.clv_usd - rows.clv_lo, rows.clv_hi - rows.clv_usd], capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["single\noriginal", "single\nfresh #1", "majority\nof 5", "maj 5,\nstake×frac",
                        "perfect\n(CLV>0)"], fontsize=8)
    ax.set_ylabel("kept CLV $ (182 proposals)")
    ax.set_title(f"Upgrade 1: majority-vote sceptic (resampled κ={st['kappa_maj5'][0]:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{STEM}.png", dpi=140)


def write_report(df, vt, st, grid, w_star, acc, trades, memo, spend_txt, model_note) -> str:
    L = ["# LLM upgrades: majority-vote sceptic and probability estimate", "",
         f"Branch `exp/llm-upgrade`. Labelled set: the {len(df)} proposals of "
         "`evaluation/sceptic_robustness.py` (exact cached sceptic prompt, original verdict, "
         "counterfactual $20 fill). Script: `evaluation/llm_upgrade.py`.", "",
         "## Upgrade 1: majority-vote sceptic (opt-in `ToolAgent(sceptic_votes=5)`)", "",
         f"Model: `{VOTE_MODEL}` → deepseek-flash thinking on (roles `sceptic_rep1..{VOTES}`). "
         f"{VOTES} votes per proposal (batched concurrent); primary decision = majority of the first "
         f"{N_VOTE}. Stability: Cohen's κ between two independent {N_VOTE}-vote majorities resampled "
         f"with replacement from each proposal's collected votes (no second full pass).", "",
         f"Resampled majority κ: **{st['kappa_maj5'][0]:.2f}** "
         f"[{st['kappa_maj5'][1]:.2f}, {st['kappa_maj5'][2]:.2f}]; "
         f"resampled single-vote κ: {st['kappa_single'][0]:.2f} "
         f"(published Fleiss κ over original+3 draws was {SINGLE_KAPPA:.2f}). "
         f"Mean pairwise single-vote κ across the {VOTES} draws: {st['kappa_single_pairs_mean']:.2f}. "
         f"Per-vote approval rates: {st['approve_rate_per_vote']}. "
         f"Original 8 re-approved by majority-of-5: {st['orig_approvals_maj5']}/8. "
         f"Invalid votes: {st['invalid_votes']}; errors: {st['errors']}.", "",
         "| Window | Filter | Kept | CLV $ [95% CI] | P&L $ [95% CI] | Random pct |",
         "|---|---|---|---|---|---|"]
    for r in vt.itertuples():
        L.append(f"| {r.window} | {r.filter} | {r.kept} | {r.clv_usd:+.2f} [{r.clv_lo:+.2f}, {r.clv_hi:+.2f}] | "
                 f"{r.pnl_usd:+.2f} [{r.pnl_lo:+.2f}, {r.pnl_hi:+.2f}] | "
                 f"{'' if pd.isna(r.random_pct) else f'{r.random_pct:.0f}'} |")
    maj = vt[(vt.window == "full Dec-Apr") & (vt["filter"] == f"majority of {N_VOTE} votes")].iloc[0]
    L += ["", f"**Verdict (upgrade 1):** majority-of-{N_VOTE} kept {maj.kept} trades, CLV "
          f"${maj.clv_usd:+.2f} (day-block 95% CI [{maj.clv_lo:+.2f}, {maj.clv_hi:+.2f}]); "
          f"random {maj.kept}-subset percentile {maj.random_pct:.0f}. "
          f"Original single sceptic was ${SINGLE_ORIG_CLV:+.2f} on 8 trades (fresh-draw mean "
          f"${np.mean(SINGLE_RERUNS):+.2f}). Resampled majority κ {st['kappa_maj5'][0]:.2f} vs "
          f"single κ {SINGLE_KAPPA:.2f}.", "",
          "## Upgrade 2: LLM probability estimate", "",
          f"Model: {model_note}. Prompt = as-of quotes, 24 h anchors, 6 h price history, inactive lists and "
          f"M4 news shift (no outcomes). Blend `p = anchor + w · (p_llm − anchor)`; "
          f"w tuned on Dec–Jan Brier only (grid {list(W_GRID)}), tested Feb–Apr. "
          f"Chosen w* = **{w_star:.1f}** (tune Brier {grid.loc[grid.w == w_star, 'tune_brier'].iloc[0]:.4f}).", "",
          "### Accuracy", "",
          "| Window | Forecast | Games | Brier | − anchor [95% CI] | − M4 [95% CI] |",
          "|---|---|---|---|---|---|"]
    for r in acc.itertuples():
        L.append(f"| {r.window} | {r.forecast} | {r.games} | {r.brier:.4f} | "
                 f"{r.minus_anchor:+.4f} [{r.minus_anchor_lo:+.4f}, {r.minus_anchor_hi:+.4f}] | "
                 f"{r.minus_m4:+.4f} [{r.minus_m4_lo:+.4f}, {r.minus_m4_hi:+.4f}] |")
    L += ["", "### Trading (4¢ edge after fees, $20 stake, tip-off CLV)", "",
          "| Window | Forecast | Trades | CLV $ [95% CI] | P&L $ [95% CI] | Random pct |",
          "|---|---|---|---|---|---|"]
    for r in trades.itertuples():
        L.append(f"| {r.window} | {r.forecast} | {r.trades} | {r.clv_usd:+.2f} [{r.clv_lo:+.2f}, {r.clv_hi:+.2f}] | "
                 f"{r.pnl_usd:+.2f} [{r.pnl_lo:+.2f}, {r.pnl_hi:+.2f}] | {r.random_pct:.0f} |")
    t_blend = trades[(trades.window == "test Feb-Apr") & trades.forecast.str.startswith("blend")].iloc[0]
    t_m4 = trades[(trades.window == "test Feb-Apr") & (trades.forecast == "anchor + M4 shift (agent)")].iloc[0]
    a_blend = acc[(acc.window == "test Feb-Apr") & acc.forecast.str.startswith("blend")].iloc[0]
    a_m4 = acc[(acc.window == "test Feb-Apr") & (acc.forecast == "anchor + M4 shift (agent)")].iloc[0]
    L += ["", f"**Verdict (upgrade 2):** on the Feb–Apr test window, blend w={w_star:.1f} Brier "
          f"{a_blend.brier:.4f} vs M4 {a_m4.brier:.4f} (Δ {a_blend.minus_m4:+.4f} "
          f"[{a_blend.minus_m4_lo:+.4f}, {a_blend.minus_m4_hi:+.4f}]); "
          f"trades {t_blend.trades} CLV ${t_blend.clv_usd:+.2f} "
          f"[{t_blend.clv_lo:+.2f}, {t_blend.clv_hi:+.2f}] vs M4 {t_m4.trades} / "
          f"${t_m4.clv_usd:+.2f}; random-filter percentile {t_blend.random_pct:.0f}.", ""]
    if memo is not None and len(memo):
        ok = memo[memo.error.isna()]
        L += ["## Memorisation probe", "",
              f"{len(ok)} games, model {model_note}, no evidence. "
              f"Answered (not 'unknown'): {ok.known.mean():.0%}. "
              f"Winner correct: {ok.winner_correct.mean():.0%} of all "
              f"({(ok.winner_correct[ok.known].mean() if ok.known.any() else float('nan')):.0%} of answered). "
              f"Exact score: {ok.score_exact.mean():.0%}.", ""]
        if ok.winner_correct.any():
            L.append("**Caution:** any correct winner recall without evidence would make the trading/Brier "
                     "numbers look suspiciously good.")
        else:
            L.append("No winner recalled correctly; outcome memorisation is not driving the results.")
    L += ["", "## Spend", "", spend_txt, "",
          "Figure: `evaluation/results/llm_upgrade.png`.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", action="store_true")
    ap.add_argument("--prob", action="store_true")
    ap.add_argument("--memo", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--spend-cap", type=float, default=10.0)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = labelled()
    if args.votes:
        print(f"votes: {VOTES} × {len(df)} = {VOTES * len(df)} calls, workers={args.workers}", flush=True)
        run_votes(df, args.workers)
        print(f"new spend so far ${sum(SPEND):.2f}", flush=True)
    if args.prob or args.memo:
        sample, tables = prob_games(df)
        if args.prob:
            print(f"prob: building evidence for {len(sample)} games…", flush=True)
            ev = prob_evidence(sample, tables)
            pd.to_pickle(ev.drop(columns=["_tools"]), WORK / "prob_evidence.pkl")
            print(f"prob: {len(ev)} prompts, model={PROB_MODEL}", flush=True)
            out = run_prob(ev, args.workers)
            out.drop(columns=["_tools", "prompt"]).to_csv(WORK / "prob.csv", index=False)
            out[["game_id", "prompt"]].to_pickle(WORK / "prob_prompts.pkl")
            # keep tools for the offline trade table via a re-build later if needed
            pd.to_pickle(out, WORK / "prob_full.pkl")
            print(f"new spend so far ${sum(SPEND):.2f}", flush=True)
        if args.memo:
            run_memo(df, tables, args.workers)
            print(f"new spend so far ${sum(SPEND):.2f}", flush=True)
    if not (args.report or args.votes or args.prob or args.memo):
        return
    votes = pd.read_csv(WORK / "votes.csv") if (WORK / "votes.csv").exists() else None
    memo = pd.read_csv(WORK / "memo.csv") if (WORK / "memo.csv").exists() else None
    vt = st = per = None
    if votes is not None:
        vt, st, per = votes_table(df, votes)
        figure(vt, st)
    grid = w_star = acc = trades = None
    model_note = f"`{PROB_MODEL[0]}` thinking={'on' if PROB_MODEL[1] else 'off'}"
    if (WORK / "prob_full.pkl").exists():
        out = pd.read_pickle(WORK / "prob_full.pkl")
        if "_tools" not in out.columns:
            # rebuild tools for trading
            sample, tables = prob_games(df)
            rebuilt = prob_evidence(sample, tables)
            out = rebuilt.merge(out.drop(columns=[c for c in out.columns if c.startswith("_") or c == "prompt"],
                                         errors="ignore"), on="game_id", suffixes=("", "_old"))
        grid, w_star, acc, trades, tr_rows, out = prob_tables(out)
        tr_rows.to_csv(WORK / "prob_trades.csv", index=False)
        grid.to_csv(WORK / "prob_w_grid.csv", index=False)
    parts = []
    if vt is not None:
        parts.append(vt.assign(table="votes"))
        parts.append(per.assign(table="votes_per"))
        parts.append(pd.DataFrame([{"table": "votes_stability", **{k: (list(v) if isinstance(v, tuple) else v)
                                                                   for k, v in st.items()}}]))
    if acc is not None:
        parts.append(acc.assign(table="prob_acc"))
        parts.append(trades.assign(table="prob_trades"))
        parts.append(grid.assign(table="prob_w"))
    if memo is not None:
        parts.append(memo.assign(table="memo"))
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"{STEM}.csv", index=False)
    spend_txt = (f"New API spend this run: ${sum(SPEND):.2f} (cap ${args.spend_cap:.0f}). "
                 f"Cached replies are free. Cap across both upgrades: ${args.spend_cap:.0f}.")
    if vt is not None and acc is not None:
        md = write_report(df, vt, st, grid, w_star, acc, trades, memo, spend_txt, model_note)
        (RESULTS / f"{STEM}.md").write_text(md)
        print(md)
    elif vt is not None:
        print(vt.to_string())
        print(st)


if __name__ == "__main__":
    main()
