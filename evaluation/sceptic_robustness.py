"""Robustness of the original approve/reject sceptic (SCEPTIC_SYSTEM + ToolAgent._sceptic) on its 182 reviewed proposals.

Labelled set: evaluation.sceptic_balanced.labelled_set (the exact cached sceptic prompt, the original verdict and the
counterfactual $20 fill's CLV). Three tests:

  rules      $0. The sceptic's 8 approvals vs one-line rules at a matched keep count (top 8 by the rule's score),
             with the percentile against random 8-subsets.
  stability  The original sceptic re-asked REPS more times on a subset (all approvals + rejects stratified by CLV
             sign, fixed seed); roles sceptic_rep{k}, so these are fresh draws and the original entries are untouched.
  memo       (a) recall: deepseek-flash, thinking off, temperature 0, asked who won each game and the score.
             (b) blinded sceptic: same prompts with team codes / names, player names / ids, dates and news ids
             replaced by neutral tokens (numbers kept), role sceptic_blind.

    python -m evaluation.sceptic_robustness --rules
    python -m evaluation.sceptic_robustness --stability --memo --workers 12
    python -m evaluation.sceptic_robustness --report
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "sceptic_robustness"
RESULTS = ROOT / "evaluation" / "results"
STEM = "sceptic_robustness"
REASONER = "deepseek-reasoner"          # flash, thinking on (agents.llm_client.LEGACY_DEEPSEEK)
REPS = 3
N_SUBSET, SEED = 60, 7606
RANDOM_DRAWS = 20000
K = 8
ALL = False
READING = ("**Not memorised, and a weak real signal; the +$7.38 is a lucky draw.** Fresh draws of the same prompt "
           "re-approve only 1-2 of the original 8 (Fleiss' kappa 0.15), keep 5 trades each and earn +$1.71, +$2.05 "
           "and +$6.04 (93rd-100th percentile vs random 5-subsets; mean +$3.27, less than half of +$7.38). "
           "Approvals still concentrate on the original approvals (17% vs 2.1% per draw), so the sceptic ranks "
           "proposals better than random but picks which of them it approves almost at random. No one-line rule "
           "matches it at 8 kept (best +$0.20). The model cannot name the winner of any of the 20 games, and "
           "blinding teams, players and dates changes verdicts no more than resampling does (92% vs 93% agreement), "
           "so team or outcome knowledge is not driving the approvals.")

NICKNAMES = {"ATL": ["Atlanta", "Hawks"], "BOS": ["Boston", "Celtics"], "BKN": ["Brooklyn", "Nets"],
             "CHA": ["Charlotte", "Hornets"], "CHI": ["Chicago", "Bulls"], "CLE": ["Cleveland", "Cavaliers", "Cavs"],
             "DAL": ["Dallas", "Mavericks", "Mavs"], "DEN": ["Denver", "Nuggets"], "DET": ["Detroit", "Pistons"],
             "GSW": ["Golden State", "Warriors"], "GS": ["Golden State", "Warriors"], "HOU": ["Houston", "Rockets"],
             "IND": ["Indiana", "Pacers"], "LAC": ["Clippers"], "LAL": ["Lakers"], "MEM": ["Memphis", "Grizzlies"],
             "MIA": ["Miami", "Heat"], "MIL": ["Milwaukee", "Bucks"], "MIN": ["Minnesota", "Timberwolves", "Wolves"],
             "NOP": ["New Orleans", "Pelicans"], "NO": ["New Orleans", "Pelicans"], "NYK": ["New York", "Knicks"],
             "NY": ["New York", "Knicks"], "OKC": ["Oklahoma City", "Thunder"], "ORL": ["Orlando", "Magic"],
             "PHI": ["Philadelphia", "76ers", "Sixers"], "PHX": ["Phoenix", "Suns"], "POR": ["Portland", "Trail Blazers",
             "Blazers"], "SAC": ["Sacramento", "Kings"], "SAS": ["San Antonio", "Spurs"], "SA": ["San Antonio", "Spurs"],
             "TOR": ["Toronto", "Raptors"], "UTA": ["Utah", "Jazz"], "UTAH": ["Utah", "Jazz"],
             "WAS": ["Washington", "Wizards"], "WSH": ["Washington", "Wizards"]}


def labelled() -> pd.DataFrame:
    p = WORK / "labelled.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from evaluation.llm_agent_eval import load_frozen
    from evaluation.sceptic_balanced import labelled_set
    df = labelled_set(load_frozen())
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return df


def evidence(prompt: str) -> dict:
    lines = prompt.split("\n")
    return json.loads(lines[lines.index("EVIDENCE") + 1])


def subset(df: pd.DataFrame) -> pd.Index:
    """All approvals plus rejects sampled stratified by CLV sign (fixed seed); with ALL, every proposal with the
    subset first (so a partial extension still contains the subset)."""
    rej = df[~df.orig_approve]
    if ALL:
        return df.index
    n = N_SUBSET - int(df.orig_approve.sum())
    pos = rej[rej.clv_usd > 0]
    k_pos = int(round(n * len(pos) / len(rej)))
    rng = np.random.default_rng(SEED)
    pick = list(rng.choice(pos.index, k_pos, replace=False)) + \
        list(rng.choice(rej.index.difference(pos.index), n - k_pos, replace=False))
    return df.index[df.orig_approve].append(pd.Index(pick)).sort_values()


# ---------------- test 3: one-line rules ----------------

def rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous score per rule (higher = keep); the top K are kept."""
    out = pd.DataFrame(index=df.index)
    ev = df.prompt.map(evidence)
    move = ev.map(lambda e: e["anchor"].get("move_since_anchor", np.nan)).astype(float)
    news_age = ev.map(lambda e: min([i.get("minutes_ago", np.inf) for i in e["news"].get("items", [])] or [np.inf]))
    change6 = ev.map(lambda e: e["price_history_6h"].get("change", np.nan)).astype(float)
    out["move_since_anchor"] = move
    out["news_age_min"] = news_age.astype(float)
    out["rule: skip if moved against (keep most toward)"] = move
    out["rule: skip if moved toward >= 3c (keep least toward)"] = -move
    out["rule: skip stale news (keep freshest)"] = -news_age.astype(float)
    out["rule: skip if 6h move against (keep most 6h toward)"] = change6
    out["rule: keep largest gap after fees"] = df.gap.astype(float)
    return out


def random_pct(clv: np.ndarray, value: float, k=K, draws=RANDOM_DRAWS, seed=SEED) -> tuple:
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((draws, len(clv))), axis=1)[:, :k]
    sums = clv[idx].sum(axis=1)
    return float((sums < value).mean() * 100 + (sums == value).mean() * 50), float(sums.mean()), \
        float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


def rules_table(df: pd.DataFrame) -> pd.DataFrame:
    clv = df.clv_usd.to_numpy()
    sc = rule_scores(df)
    rows = []
    sets = {"LLM sceptic (original approvals)": df.orig_approve.to_numpy()}
    rng = np.random.default_rng(SEED)
    for col in [c for c in sc.columns if c.startswith("rule:")]:
        s = sc[col].to_numpy(dtype=float)
        s = np.where(np.isfinite(s), s, -1e9) + rng.random(len(s)) * 1e-9      # random tie-break, fixed seed
        keep = np.zeros(len(s), bool)
        keep[np.argsort(-s)[:K]] = True
        sets[col.replace("rule: ", "")] = keep
    for name, keep in sets.items():
        v = float(clv[keep].sum())
        pct, rmean, rlo, rhi = random_pct(clv, v)
        rows.append({"test": "rules", "filter": name, "kept": int(keep.sum()), "clv_usd": round(v, 2),
                     "mean_clv_c": round(float(df.clv[keep].mean()) * 100, 2),
                     "overlap_with_llm": int((keep & df.orig_approve.to_numpy()).sum()),
                     "random_pct": round(pct, 1), "random_mean": round(rmean, 2),
                     "random_lo": round(rlo, 2), "random_hi": round(rhi, 2)})
    return pd.DataFrame(rows)


# ---------------- LLM calls ----------------

def parse_sceptic(reply: dict) -> dict:
    """ToolAgent._sceptic's parse, verbatim (fail closed)."""
    from agents.graph import BANNED
    msg = reply["json"] or {}
    verdict = str(msg.get("verdict", "")).lower()
    if reply["error"] or verdict not in ("approve", "reject"):
        return {"verdict": "reject", "reason": "sceptic output invalid or failed (fail closed)", "invalid": True}
    reason = str(msg.get("reason", ""))[:300]
    if BANNED.search(reason):
        return {"verdict": "reject", "reason": "sceptic used banned wording (fail closed)", "invalid": True}
    return {"verdict": verdict, "reason": reason}


def blind(prompt: str, away: str, home: str) -> str:
    ev = evidence(prompt)
    names, ids = set(), set()

    def walk(x):
        if isinstance(x, dict):
            if x.get("player"):
                names.add(str(x["player"]))
            if x.get("player_id") is not None:
                ids.add(str(x["player_id"]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(ev)
    for v in re.findall(r'"out_used":\[([\d,]*)\]', prompt) + re.findall(r'"ignored":\[([\d,]*)\]', prompt):
        ids.update(i for i in v.split(",") if i)
    out = prompt
    out = re.sub(r"espn-\d+-\d+", lambda m: "news-item", out)
    out = re.sub(r", game date \d{4}-\d{2}-\d{2}", ", game date Day X", out)
    out = re.sub(r"\d{4}-\d{2}-\d{2}", "Day X", out)
    pmap = {n: f"Player {i + 1}" for i, n in enumerate(sorted(names, key=len, reverse=True))}
    for n, tok in pmap.items():
        out = out.replace(n, tok)
    for n, tok in pmap.items():
        for part in n.split():
            if len(part) >= 4 and part not in ("Team", "Player"):
                out = re.sub(rf"\b{re.escape(part)}\b", tok, out)
    for i, pid in enumerate(sorted(ids)):
        out = re.sub(rf"\b{pid}\b", str(9000 + i), out)
    for code, tok in ((away, "Team A"), (home, "Team B")):
        for nick in sorted(NICKNAMES.get(code, []), key=len, reverse=True):
            out = re.sub(rf"\b{re.escape(nick)}\b", tok, out)
        out = re.sub(rf"\b{code}\b", tok, out)
    return out


CACHE_ONLY = False


class CacheOnly:
    def __init__(self, model, thinking=None):
        from agents.llm_client import resolve_deepseek
        self.model = model
        self.resolved_model, self.thinking = resolve_deepseek(model, thinking)

    def generate(self, system, prompt, meta=None):
        raise RuntimeError("cache miss (cache-only)")


def call_all(jobs: list, workers: int) -> list:
    """jobs: (key, role, system, prompt, model, thinking). Returns (key, reply)."""
    from agents.llm_client import CachedLLM, DeepSeekBackend
    llms = {}

    def llm_for(model, thinking):
        if (model, thinking) not in llms:
            llms[(model, thinking)] = CachedLLM(CacheOnly(model, thinking) if CACHE_ONLY else
                                                DeepSeekBackend(model, thinking=thinking, max_retries=1))
        return llms[(model, thinking)]
    for _k, _r, _s, _p, m, t in jobs:
        llm_for(m, t)

    def one(job):
        key, role, system, prompt, model, thinking = job
        return key, llms[(model, thinking)].ask(role, system, prompt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def run_stability(df, idx, workers, blind_too: bool):
    from agents.tool_agent import SCEPTIC_SYSTEM
    jobs = []
    for i in sorted(idx, key=lambda i: (not df.orig_approve[i], i)):    # approvals first; all draws of a row together
        jobs += [((i, f"rep{k}"), f"sceptic_rep{k}", SCEPTIC_SYSTEM, df.prompt[i], REASONER, None)
                 for k in range(1, REPS + 1)]
        if blind_too:
            away, home = df.game[i].split(" @ ")
            jobs.append(((i, "blind"), "sceptic_blind", SCEPTIC_SYSTEM, blind(df.prompt[i], away, home), REASONER, None))
    rows = []
    for (i, draw), r in call_all(jobs, workers):
        p = parse_sceptic(r)
        rows.append({"row": i, "draw": draw, "verdict": p["verdict"], "invalid": bool(p.get("invalid")),
                     "error": r["error"], "cached": r["cached"], "cost": r["cost"], "tokens_out": r["tokens_out"],
                     "reason": p["reason"]})
    return pd.DataFrame(rows)


RECALL_SYSTEM = "You answer questions about past NBA games. Reply with exactly one JSON object: " \
                '{"winner": team abbreviation or "unknown", "away_pts": integer or null, "home_pts": integer or null}.'


def run_recall(df, games, workers, n_random=12):
    rng = np.random.default_rng(SEED)
    appr = list(df.index[df.orig_approve])
    others = df[~df.orig_approve].drop_duplicates("game_id")
    others = others[~others.game_id.isin(df.game_id[df.orig_approve])]
    idx = appr + list(rng.choice(others.index, n_random, replace=False))
    g = games.set_index("game_id")
    jobs = []
    for i in idx:
        away, home = df.game[i].split(" @ ")
        q = (f"Who won the NBA game {away} @ {home} on {df.date[i]}, and what was the score? "
             "If you don't know, say unknown.")
        jobs.append((i, "recall", RECALL_SYSTEM, q, "deepseek-flash", False))
    rows = []
    for i, r in call_all(jobs, workers):
        gid = df.game_id[i]
        away, home = df.game[i].split(" @ ")
        hp, ap = int(g.home_pts[gid]), int(g.away_pts[gid])
        actual = home if hp > ap else away
        m = r["json"] or {}
        win = str(m.get("winner", "unknown")).upper()
        known = win not in ("UNKNOWN", "", "NONE")
        try:
            sa, sh = int(m.get("away_pts")), int(m.get("home_pts"))
        except (TypeError, ValueError):
            sa = sh = None
        rows.append({"row": i, "game": df.game[i], "date": df.date[i], "approved": bool(df.orig_approve[i]),
                     "actual_winner": actual, "actual_score": f"{ap}-{hp}", "said_winner": win,
                     "said_score": None if sa is None else f"{sa}-{sh}", "known": known,
                     "winner_correct": known and win == actual,
                     "score_exact": sa == ap and sh == hp,
                     "score_abs_err": None if sa is None else abs(sa - ap) + abs(sh - hp),
                     "cost": r["cost"], "cached": r["cached"], "error": r["error"]})
    return pd.DataFrame(rows)


# ---------------- stats ----------------

def fleiss_kappa(mat: np.ndarray) -> float:
    """mat: items x categories counts, equal raters per item."""
    n = mat.sum(axis=1)[0]
    p_j = mat.sum(axis=0) / mat.sum()
    p_i = ((mat ** 2).sum(axis=1) - n) / (n * (n - 1))
    pbar, pe = p_i.mean(), (p_j ** 2).sum()
    return float((pbar - pe) / (1 - pe)) if pe < 1 else float("nan")


def stability_stats(df, idx, draws: pd.DataFrame) -> tuple:
    sub = df.loc[idx]
    rej_w = (~df.orig_approve).sum() / (~sub.orig_approve).sum()     # scale sampled rejects to all rejects
    w = np.where(sub.orig_approve, 1.0, rej_w)
    wide = draws[draws.draw.str.startswith("rep")].pivot(index="row", columns="draw", values="verdict")
    wide = wide.reindex(idx)
    wide.insert(0, "orig", np.where(sub.orig_approve, "approve", "reject"))
    appr = wide == "approve"
    rows = []
    for d in wide.columns:
        a = appr[d].to_numpy()
        rows.append({"test": "stability", "filter": f"draw {d}", "kept": int(a.sum()),
                     "orig8_reapproved": int((a & sub.orig_approve.to_numpy()).sum()),
                     "kept_scaled_full": round(float((a * w).sum()), 1),
                     "clv_usd": round(float(sub.clv_usd.to_numpy()[a].sum()), 2),
                     "clv_usd_scaled_full": round(float((sub.clv_usd.to_numpy() * a * w).sum()), 2),
                     "agree_with_orig": round(float((wide[d] == wide["orig"]).mean()), 3),
                     "invalid": int(draws[(draws.draw == d)].invalid.sum()) if d != "orig" else 0})
    mat = np.stack([appr.sum(axis=1).to_numpy(), (~appr).sum(axis=1).to_numpy()], axis=1)
    kappa = fleiss_kappa(mat)
    rate = appr.mean(axis=1)
    per = pd.DataFrame({"row": idx, "game": sub.game, "date": sub.date, "orig_approve": sub.orig_approve,
                        "clv_usd": sub.clv_usd, "approve_rate_4": rate.to_numpy()})
    pairs = [(a, b) for i, a in enumerate(wide.columns) for b in wide.columns[i + 1:]]
    pair_agree = float(np.mean([(wide[a] == wide[b]).mean() for a, b in pairs]))
    unanimous = float((rate.isin([0, 1])).mean())
    reps = [c for c in wide.columns if c != "orig"]
    fa = appr[reps].to_numpy()
    o = sub.orig_approve.to_numpy()
    pcts = []
    for c in reps:                     # random subsets of the same (unweighted) size from the evaluated rows
        a = appr[c].to_numpy()
        pcts.append(random_pct(sub.clv_usd.to_numpy(), float(sub.clv_usd.to_numpy()[a].sum()), k=int(a.sum()))[0]
                    if a.sum() else float("nan"))
    return pd.DataFrame(rows), per, {"kappa": kappa, "pair_agree": pair_agree, "unanimous": unanimous,
                                     "rej_weight": rej_w, "fresh_rate_orig": float(fa[o].mean()),
                                     "fresh_rate_rej": float(fa[~o].mean()), "fresh_pcts": pcts}


def write_report(df, rules, stab, per, st, blindres, recall, spend):
    L = ["# Sceptic robustness: rules, stability, memorisation", "",
         f"Original approve/reject sceptic (`SCEPTIC_SYSTEM` + `ToolAgent._sceptic` parse, fail closed) on its "
         f"{len(df)} reviewed proposals (labelled set from `evaluation/sceptic_balanced.py`; CLV $ of the "
         f"counterfactual $20 fill). Original: {int(df.orig_approve.sum())} approvals, kept CLV "
         f"${df.clv_usd[df.orig_approve].sum():+.2f}; all {len(df)} sum to ${df.clv_usd.sum():+.2f}. "
         "Script: `evaluation/sceptic_robustness.py`.", "",
         f"## Test 3: one-line rules at a matched keep count of {K}", "",
         "Each rule keeps the top 8 by its continuous score (fixed-seed random tie-break). Random percentile = share "
         f"of {RANDOM_DRAWS:,} random 8-subsets of the 182 with lower kept CLV $.", "",
         "| Filter | Kept CLV $ | Mean CLV (c) | Overlap with LLM | Random pct | Random mean [95%] |", "|---|---|---|---|---|---|"]
    for r in rules.itertuples():
        L.append(f"| {r.filter} | {r.clv_usd:+.2f} | {r.mean_clv_c:+.2f} | {r.overlap_with_llm} | {r.random_pct:.0f} | "
                 f"{r.random_mean:+.2f} [{r.random_lo:+.2f}, {r.random_hi:+.2f}] |")
    if stab is not None:
        L += ["", f"## Test 1: stability ({REPS} fresh draws, roles `sceptic_rep1..{REPS}`)", "",
              (f"All {len(per)} proposals (the 60-proposal stratified subset, seed {SEED}, was run first)."
               if len(per) == len(df) else
               f"Subset: all {int(df.orig_approve.sum())} approvals + {len(per) - int(df.orig_approve.sum())} rejects "
               f"stratified by CLV sign (seed {SEED}). 'Scaled' weights each sampled reject by "
               f"{st['rej_weight']:.2f} to stand for all {int((~df.orig_approve).sum())} rejects."), "",
              "| Draw | Approvals (subset) | Original 8 re-approved | Approvals scaled to 182 | Kept CLV $ (subset) | "
              "Kept CLV $ scaled to 182 | Agreement with original | Invalid |", "|---|---|---|---|---|---|---|---|"]
        for r in stab.itertuples():
            L.append(f"| {r.filter} | {r.kept} | {r.orig8_reapproved} | {r.kept_scaled_full} | {r.clv_usd:+.2f} | "
                     f"{r.clv_usd_scaled_full:+.2f} | {r.agree_with_orig:.0%} | {r.invalid} |")
        fresh = stab[stab["filter"] != "draw orig"]
        L += ["", f"Fleiss' kappa over 4 draws: **{st['kappa']:.2f}**; mean pairwise agreement {st['pair_agree']:.0%}; "
              f"unanimous on {st['unanimous']:.0%} of proposals. Fresh draws' kept CLV $ (scaled): mean "
              f"{fresh.clv_usd_scaled_full.mean():+.2f}, range {fresh.clv_usd_scaled_full.min():+.2f} to "
              f"{fresh.clv_usd_scaled_full.max():+.2f}. Fresh-draw approval rate: "
              f"{st['fresh_rate_orig']:.0%} on the original approvals vs {st['fresh_rate_rej']:.1%} on the original "
              f"rejects. Fresh-draw kept CLV $ vs random same-size subsets of the evaluated rows (unweighted, so "
              f"only meaningful on the full set): percentiles "
              f"{', '.join(f'{p:.0f}' for p in st['fresh_pcts'])}.", "",
              "Per-proposal approval rate across the 4 draws (original + 3), approvals and any proposal approved at "
              "least once:", "", "| Game | Date | Original | CLV $ | Approve rate |", "|---|---|---|---|---|"]
        for r in per[(per.orig_approve) | (per.approve_rate_4 > 0)].sort_values("approve_rate_4", ascending=False) \
                .itertuples():
            L.append(f"| {r.game} | {r.date} | {'approve' if r.orig_approve else 'reject'} | {r.clv_usd:+.2f} | "
                     f"{r.approve_rate_4:.2f} |")
    if recall is not None:
        ok = recall[recall.error.isna()]
        L += ["", "## Test 2a: outcome recall (deepseek-flash, thinking off, temperature 0)", "",
              f"{len(ok)} games ({int(ok.approved.sum())} approved + {int((~ok.approved).sum())} random reviewed). "
              f"Answered (not 'unknown'): {ok.known.mean():.0%}. Winner correct: {ok.winner_correct.mean():.0%} of all, "
              f"{(ok.winner_correct[ok.known].mean() if ok.known.any() else float('nan')):.0%} of answered "
              f"(chance 50%). Exact score: {ok.score_exact.mean():.0%}; median |score error| "
              f"{ok.score_abs_err.median() if ok.score_abs_err.notna().any() else float('nan')} points.", "",
              "| Game | Date | Approved | Actual | Said winner | Said score | Correct |", "|---|---|---|---|---|---|---|"]
        for r in ok.itertuples():
            L.append(f"| {r.game} | {r.date} | {'yes' if r.approved else ''} | {r.actual_winner} {r.actual_score} | "
                     f"{r.said_winner} | {r.said_score} | {'yes' if r.winner_correct else 'no'} |")
    if blindres is not None:
        b = blindres
        L += ["", "## Test 2b: blinded sceptic (role `sceptic_blind`)", "",
              "Team codes and names -> Team A / Team B, player names -> Player k, player ids -> neutral ids, dates -> "
              "Day X, news ids -> news-item; all other numbers kept.", "",
              f"- Blinded approvals on the {b['n']}-proposal subset: {b['blind_appr']} (original: {b['orig_appr']}; "
              f"fresh unblinded draws: {b['rep_appr']}).",
              f"- Original 8 approved when blinded: {b['orig8_blind']} of 8.",
              f"- Agreement blinded vs original: {b['agree_orig']:.0%}; blinded vs unblinded fresh draws (mean): "
              f"{b['agree_reps']:.0%}; unblinded draws with each other (mean): {b['agree_rep_rep']:.0%}.",
              f"- Blinded kept CLV $: {b['clv']:+.2f} on the subset ({b['clv_scaled']:+.2f} scaled to 182).",
              f"- Invalid blinded replies: {b['invalid']}."]
    if READING:
        L += ["", "## Reading", "", READING]
    L += ["", "## Spend", "", spend, ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--memo", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--all", action="store_true", help="stability on all 182 proposals, not the 60 subset")
    ap.add_argument("--cache-only", action="store_true", help="no API calls; missing draws are dropped")
    ap.add_argument("--spend", default="")
    args = ap.parse_args()
    global CACHE_ONLY, ALL
    CACHE_ONLY, ALL = args.cache_only, args.all
    df = labelled()
    idx = subset(df)
    rules = rules_table(df)
    print(rules.to_string(), flush=True)
    if args.stability or args.memo:
        jobs_df = run_stability(df, idx, args.workers, blind_too=args.memo)
        if (WORK / "draws.csv").exists():
            old = pd.read_csv(WORK / "draws.csv")
            jobs_df = pd.concat([old, jobs_df]).drop_duplicates(["row", "draw"], keep="last")
        jobs_df.to_csv(WORK / "draws.csv", index=False)
        print(f"sceptic calls: {len(jobs_df)}, new {int((~jobs_df.cached).sum())}, est. "
              f"${jobs_df.cost[~jobs_df.cached].sum():.2f}, errors {int(jobs_df.error.notna().sum())}", flush=True)
    if args.recall:
        from evaluation.llm_agent_eval import load_frozen
        rec = run_recall(df, load_frozen()["games"], args.workers)
        rec.to_csv(WORK / "recall.csv", index=False)
        print(rec[["game", "date", "approved", "actual_winner", "actual_score", "said_winner", "said_score"]]
              .to_string(), flush=True)
        return
    if not (args.report or args.stability or args.memo):
        return
    draws = pd.read_csv(WORK / "draws.csv") if (WORK / "draws.csv").exists() else None
    recall = pd.read_csv(WORK / "recall.csv") if (WORK / "recall.csv").exists() else None
    stab = per = st = blindres = None
    if draws is not None:
        draws = draws[draws.error.isna()]
        full = draws[draws.draw.str.startswith("rep")].groupby("row").size()
        idx = pd.Index(sorted(full.index[full == REPS]))
        print(f"proposals with all {REPS} fresh draws: {len(idx)} ({int(df.orig_approve[idx].sum())} approvals)")
    if draws is not None and len(idx):
        stab, per, st = stability_stats(df, idx, draws)
    if draws is not None and (draws.draw == "blind").any():
        idx = idx.intersection(draws.row[draws.draw == "blind"])
        sub = df.loc[idx]
        bl = draws[draws.draw == "blind"].set_index("row").reindex(idx)
        ba = (bl.verdict == "approve").to_numpy()
        reps = draws[draws.draw.str.startswith("rep")].pivot(index="row", columns="draw", values="verdict").reindex(idx)
        orig = np.where(sub.orig_approve, "approve", "reject")
        rep_cols = list(reps.columns)
        blindres = {"n": len(idx), "blind_appr": int(ba.sum()), "orig_appr": int(sub.orig_approve.sum()),
                    "rep_appr": ", ".join(str(int((reps[c] == "approve").sum())) for c in rep_cols),
                    "orig8_blind": int((ba & sub.orig_approve.to_numpy()).sum()),
                    "agree_orig": float((bl.verdict.to_numpy() == orig).mean()),
                    "agree_reps": float(np.mean([(bl.verdict.to_numpy() == reps[c].to_numpy()).mean()
                                                 for c in rep_cols])) if rep_cols else float("nan"),
                    "agree_rep_rep": float(np.mean([(reps[a] == reps[b]).mean() for i, a in enumerate(rep_cols)
                                                    for b in rep_cols[i + 1:]])) if len(rep_cols) > 1 else float("nan"),
                    "clv": float(sub.clv_usd.to_numpy()[ba].sum()),
                    "clv_scaled": float((sub.clv_usd.to_numpy() * ba * np.where(sub.orig_approve, 1.0,
                                       (~df.orig_approve).sum() / max((~sub.orig_approve).sum(), 1))).sum()),
                    "invalid": int(bl.invalid.sum())}
    parts = [rules]
    if stab is not None:
        parts.append(stab)
        parts.append(per.assign(test="per_proposal"))
    if blindres is not None:
        parts.append(pd.DataFrame([{"test": "blind", **blindres}]))
    if recall is not None:
        parts.append(recall.assign(test="recall"))
    pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"{STEM}.csv", index=False)
    md = write_report(df, rules, stab, per, st, blindres, recall, args.spend)
    (RESULTS / f"{STEM}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
