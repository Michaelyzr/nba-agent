"""Parse evaluation/results/* into demo/traces/results.json and copy the figures into demo/figures/.

    PYTHONPATH=. python demo/build_results.py          # or: python demo/build_traces.py --results

Needs only pandas and git (no data, models or API keys). Every number shown on the demo's Results and
Learning pages comes from here; nothing is typed into demo_app.py by hand.
"""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
OUT = ROOT / "demo" / "traces"
FIG = ROOT / "demo" / "figures"
RESULTS_REF = "origin/results/deepseek-llm-agent"
ON_REF = ("llm_agent.md", "llm_vs_original.md", "edge_gate.md", "edge_gate.png", "llm_vs_original.png")
FIGURES = ("edge_gate.png", "adaptive_edge.png", "forecast_blend.png", "llm_vs_original.png", "model_review.png",
           "model_review_loop.png", "pnl_compare.png")
SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|api[_-]?key)", re.I)


def git_show(name, binary=False):
    try:
        out = subprocess.run(["git", "show", f"{RESULTS_REF}:evaluation/results/{name}"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        return out if binary else out.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def read(name):
    """Text of a results file; the final DeepSeek-era files are read from the results branch when present."""
    text = git_show(name) if name in ON_REF else None
    src = f"{name} ({RESULTS_REF})" if text else name
    return (text if text else (RES / name).read_text()), src


def md_tables(text):
    tables, cur = [], None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = {"head": cells, "rows": []}
            elif not all(set(c) <= set("-: ") for c in cells):
                cur["rows"].append(dict(zip(cur["head"], cells)))
        elif cur is not None:
            tables.append(cur["rows"])
            cur = None
    if cur is not None:
        tables.append(cur["rows"])
    return tables


def section(text, heading):
    """Body of the markdown section whose heading starts with `heading` (any level), up to the next heading."""
    m = re.search(rf"^(#+) {re.escape(heading)}.*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(rf"^#{{1,{len(m.group(1))}}} ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def num(s):
    try:
        x = float(str(s).replace(",", "").replace("%", "").replace("$", "").replace("¢", ""))
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def row(rows, col, prefix):
    return next((r for r in rows if r.get(col, "").startswith(prefix)), {})


def clean_cell(v):
    v = "" if v is None else str(v)
    return "–" if v.lower() in ("nan", "+nan", "none", "") else v.replace("+nan", "–")


def tidy(rows):
    return [{k: clean_cell(v) for k, v in r.items()} for r in rows]


# ---------------- win-probability models vs the market ----------------

def win_models():
    rows, srcs = [], []

    def add(name, group, brier, ll, acc, n, src):
        rows.append({"predictor": name, "group": group, "brier": brier, "log_loss": ll, "accuracy": acc, "games": n,
                     "source": src})

    m = pd.read_csv(RES / "m4_vs_market.csv").set_index("predictor")
    pick = lambda k: m.loc[k]  # noqa: E731
    r = pick("Always 55% home")
    add("Always 55% home", "baseline", r.brier, r.log_loss, r.accuracy, int(r.games), "m4_vs_market.csv")
    r = pick("M4, absences known")
    add("M4 (logistic regression, absences known)", "our model", r.brier, r.log_loss, r.accuracy, int(r.games),
        "m4_vs_market.csv")
    cal = md_tables(read("m4_calibration.md")[0])[0]
    for key, name in (("M4 Platt", "M4 + Platt scaling"), ("M4 isotonic", "M4 + isotonic"),
                      ("M4 Feb–Apr Platt", "M4 + Feb–Apr Platt (exploratory)")):
        x = row(cal, "Predictor", key)
        add(name, "our model", num(x["Brier"]), num(x["Log loss"]), num(x["Accuracy"]) / 100, 501,
            "m4_calibration.md")
    nn = md_tables(read("win_nn.md")[0])[0]
    for key, name, grp in (("M4-NN MLP", "M4-NN MLP (3-seed mean)", "our model"),
                           ("M4-NN GRU", "M4-NN GRU (3-seed mean)", "our model")):
        x = row(nn, "Predictor", key)
        add(name, grp, num(x["Brier"]), num(x["Log loss"]), num(x["Acc."]) / 100, 501, "win_nn.md")
    x = row(cal, "Predictor", "Market 24 h")
    add("Anchor: market 24 h before tip", "market", num(x["Brier"]), num(x["Log loss"]), num(x["Accuracy"]) / 100,
        501, "m4_calibration.md")
    x = row(cal, "Predictor", "Anchor estimate, raw M4 shift")
    add("Anchor + M4 shift (the agent's estimate)", "anchor + model", num(x["Brier"]), num(x["Log loss"]),
        num(x["Accuracy"]) / 100, 501, "m4_calibration.md")
    for key, name in (("Anchor + MLP shift", "Anchor + MLP shift"), ("Anchor + GRU shift", "Anchor + GRU shift")):
        x = row(nn, "Predictor", key)
        add(name, "anchor + model", num(x["Brier"]), num(x["Log loss"]), num(x["Acc."]) / 100, 501, "win_nn.md")
    fb = md_tables(read("forecast_blend.md")[0])[0]
    x = row(fb, "Predictor", "Blend: blend (all inputs)")
    add("Learned blend (all inputs, walk-forward)", "anchor + model", num(x["Brier"]), num(x["Log loss"]), None, 501,
        "forecast_blend.md")
    r = pick("Market, 1 h before tip")
    add("Kalshi 1 h before tip", "market", r.brier, r.log_loss, r.accuracy, int(r.games), "m4_vs_market.csv")
    r = pick("Market at tip")
    add("Kalshi at tip", "market", r.brier, r.log_loss, r.accuracy, int(r.games), "m4_vs_market.csv")
    v = pd.read_csv(RES / "venue_compare.csv")
    v = v[(v.section == "sharpness")]
    for venue, name in (("kalshi", "Kalshi close (venue comparison)"), ("polymarket", "Polymarket close"),
                        ("draftkings", "DraftKings close (de-vigged)")):
        b = v[(v.venue == venue) & (v.metric == "brier")].iloc[0]
        ll = v[(v.venue == venue) & (v.metric == "log_loss")].iloc[0]
        add(name, "venue close", float(b.value), float(ll.value), None, int(b.games), "venue_compare.csv")
    return rows


# ---------------- player models ----------------

def player_models():
    m2 = pd.read_csv(RES / "m2_heldout.csv")
    names = {"rolling10": "M1a: 10-game rolling average", "gbm": "M1b: gradient boosting",
             "gru": "M2: GRU (deep learning)"}
    m2_rows = [{"model": names.get(r.model, r.model), "target": {"pts": "points", "min": "minutes"}[r.target],
                "pinball": float(r.pinball), "coverage_10_90": float(r.coverage_10_90),
                "median_abs_error": float(r.median_abs_error)} for r in m2.itertuples()]
    m3 = pd.read_csv(RES / "m3_heldout.csv")
    m3_names = {"model": "M3 model", "played_rate10": "Played-rate over last 10 games",
                "played_last_game": "Played last game"}
    m3_rows = [{"predictor": m3_names.get(r.predictor, r.predictor), "brier": float(r.brier), "rows": int(r.rows),
                "base_rate": float(r.base_rate)} for r in m3.itertuples()]
    m6 = pd.read_csv(RES / "m6_heldout.csv")
    t = m6[(m6.split == "test") & (m6.rows_used == "decision times")]
    m6_rows = [{"model": r.model, "rows": int(r.rows), "mae_c": float(r.mae_c),
                "mae_minus_zero_c": float(r.mae_minus_zero_c),
                "ci": f"[{r.mae_minus_zero_lo:+.3f}, {r.mae_minus_zero_hi:+.3f}]"} for r in t.itertuples()]
    abl = md_tables(read("m6_ablation.md")[0])[0]
    m6_trades = row([r for r in abl if r.get("period") == "test"], "setup", "M6 agent, no learning").get("trades")
    return {"m2": m2_rows, "m3": m3_rows, "m6": m6_rows, "m6_agent_trades_test": m6_trades,
            "sources": ["m2_heldout.csv", "m3_heldout.csv", "m6_heldout.csv", "m6_ablation.md"]}


# ---------------- agents ----------------

def agents():
    sig = read("significance.md")[0]
    s1, s2 = md_tables(sig)[:2]
    orig = [r for r in s2 if r["Period"].startswith("Original test")]
    raw_n = row([r for r in s1 if r["Period"].startswith("Original test")], "Setup", "Raw model").get("Trades")
    raw_clv = row([r for r in s1 if r["Period"].startswith("Original test")], "Setup", "Raw model")
    raw_c = next(r for r in orig if r["Comparison"].startswith("Raw model") and r["Metric"] == "CLV $")
    raw_p = next(r for r in orig if r["Comparison"].startswith("Raw model") and r["Metric"] == "P&L")
    ae = md_tables(read("adaptive_edge.md")[0])
    full = [{"setup": "Raw M4, no agent (no anchor)", "trades": raw_n,
             "mean_clv": raw_clv.get("Mean CLV [95% CI]"), "clv_dollars": raw_c["Difference [95% CI]"],
             "pnl": raw_p["Difference [95% CI]"], "source": "significance.md"}]
    names = {"(i)": "Anchor agent, fixed 4¢ edge, no learning", "(ii)": "Split-gate learning (published agent)",
             "(iii)": "Learned edge threshold (base 4¢)", "(iv)": "Learned edge threshold (base 1¢)"}
    for r in ae[0]:
        k = r["Setup"].split(" ")[0]
        full.append({"setup": names.get(k, r["Setup"]), "trades": r["Trades"],
                     "mean_clv": r["Mean CLV/contract [95% CI]"], "clv_dollars": r["CLV $ [95% CI]"],
                     "pnl": r["P&L after fees [95% CI]"], "source": "adaptive_edge.md"})
    full.append({"setup": "Never trade", "trades": "0", "mean_clv": "–", "clv_dollars": "+0", "pnl": "+0",
                 "source": "–"})
    llm, _ = read("llm_agent.md")
    sub = []
    rsn = section(llm, "test-deepseek-reasoner:")
    chat = section(llm, "test-deepseek:")
    rs, cs = md_tables(rsn)[0], md_tables(chat)[0]
    a = row(rs, "Setup", "A.")
    sub.append({"setup": "A. Deterministic anchor agent (4¢ edge)", "trades": a.get("Trades"),
                "mean_clv": a.get("Mean CLV [95% CI]"), "clv_dollars": a.get("CLV $ [95% CI]"),
                "pnl": a.get("P&L after fees [95% CI]"), "source": "llm_agent.md"})
    for rows_, lab in ((cs, "DeepSeek chat"), (rs, "DeepSeek reasoner")):
        for k, n in (("B.", "B. plain LLM"), ("C.", "C. LLM tool agent"), ("D.", "D. tool agent + sceptic")):
            x = row(rows_, "Setup", k)
            sub.append({"setup": f"{n} ({lab})", "trades": x.get("Trades"), "mean_clv": x.get("Mean CLV [95% CI]"),
                        "clv_dollars": x.get("CLV $ [95% CI]"), "pnl": x.get("P&L after fees [95% CI]"),
                        "source": "llm_agent.md"})
    eg = md_tables(read("edge_gate.md")[0])
    for r in eg[1]:
        if r["Min edge"] == "0¢" and not r["Arm"].startswith("D-chat"):
            sub.append({"setup": f"{r['Arm']}, no edge gate (0¢)", "trades": r["Trades"],
                        "mean_clv": r["Mean CLV [95% CI]"], "clv_dollars": r["CLV $ [95% CI]"],
                        "pnl": r["P&L after fees [95% CI]"], "source": "edge_gate.md"})
    sub.append({"setup": "Never trade", "trades": "0", "mean_clv": "–", "clv_dollars": "+0", "pnl": "+0",
                "source": "–"})
    return {"full": tidy(full), "subsample": tidy(sub),
            "full_label": "Full test period, 1 Feb – 12 Apr 2026 (every decision point, 501 games)",
            "subsample_label": "40% subsample: 304 test decision points (seed 7606), the LLM comparison window",
            "raw_pnl": raw_p["Difference [95% CI]"],
            "anchor_pnl": row(ae[0], "Setup", "(i)").get("P&L after fees [95% CI]")}


# ---------------- edge gate ----------------

def edge_gate():
    text, src = read("edge_gate.md")
    t = md_tables(text)
    sweep = [{"min_edge": r["Min edge"], "trades": int(num(r["Trades"])), "clv_dollars": num(r["CLV $ [95% CI]"].split()[0]),
              "clv_ci": r["CLV $ [95% CI]"], "mean_clv": r["Mean CLV [95% CI]"], "pnl": r["P&L after fees [95% CI]"],
              "win_rate": r["Win rate"]} for r in t[0]]
    paired = tidy([r for r in t[2] if not r["Arm"].startswith("D-chat")])
    split = tidy([r for r in t[3] if r["Trades"] in ("low-edge trades (gap ≤ 4¢), edge 0",
                                                    "normal trades (gap > 4¢), edge 0")])
    sceptic = tidy(t[4])
    summary = re.search(r"\*\*Summary\.\*\*\s*(.+)", text)
    d0 = row([r for r in sceptic if r["Min edge"] == "0¢"], "Arm", "D. Reasoner")
    d_arm = row([r for r in t[1] if r["Min edge"] == "0¢"], "Arm", "D. Reasoner")
    a_pair = next(r for r in paired if r["Arm"].startswith("A.") and r["Metric"] == "CLV $")
    return {"source": src, "summary": summary.group(1).strip() if summary else "", "sweep": sweep,
            "paired": paired, "split": split, "sceptic": sceptic,
            "a_paired_clv": a_pair["Difference [95% CI]"],
            "d0": {"calls": d0.get("Sceptic calls"), "vetoes": d0.get("Vetoes"),
                   "low_edge": d0.get("Low-edge (≤ 4¢) calls / vetoes (vetoed CLV)"), "trades": d_arm.get("Trades")},
            "figure": "edge_gate.png"}


# ---------------- learned edge threshold ----------------

def adaptive_edge():
    text, src = read("adaptive_edge.md")
    t = md_tables(text)
    setups, paired = tidy(t[0]), tidy(t[1])
    arm3 = section(text, "Threshold the agent ended up at")
    m = re.search(r"\*\*\(iii\)[^*]*\*\*\s*—\s*market-wide threshold:\s*(.+?)\.\s", arm3)
    timeline = [s.strip() for s in m.group(1).split("→")] if m else []
    rules3 = md_tables(arm3.split("**(iii)", 1)[1].split("**(iv)", 1)[0])[0] if "**(iii)" in arm3 else []
    rules2 = md_tables(arm3.split("**(ii)", 1)[1].split("**(iii)", 1)[0])[0] if "**(ii)" in arm3 else []
    story = []
    for r in rules3:
        before, after = [num(x) for x in r["CLV $ without → with"].split("→")]
        accepted = r["Status"] in ("active", "accepted", "superseded")
        story.append({"rule": r["Rule"], "decided": r["Gate decided"], "proposal": f"set the edge threshold to {r['Edge']}"
                      + ("" if r["Slice"] == "all games" else f" on {r['Slice']}"),
                      "gate_days": r["Gate days"], "clv_without": before, "clv_with": after,
                      "changed_trades": int(num(r["Changed trades"]) or 0), "accepted": accepted,
                      "status": r["Status"]})
    pick = lambda a, b, metric: next((r["Difference [95% CI]"] for r in paired  # noqa: E731
                                      if r["Comparison"].startswith(a) and f"− {b}" in r["Comparison"]
                                      and r["Metric"] == metric), "")
    design = section(text, "Design").strip()
    verdict = re.search(r"\*\*Verdict\.\*\*\s*(.+)", text)
    expiry = re.search(r"the 10¢ rule expired after its (\d+) days on ([^.]+)\.", text)
    return {"source": src, "setups": setups, "paired": paired, "timeline": timeline, "story": story,
            "rules_split_gate": tidy(rules2), "design": design,
            "vs_none": pick("(iii)", "(i)", "clv_dollars"), "vs_split": pick("(iii)", "(ii)", "clv_dollars"),
            "vs_none_pnl": pick("(iii)", "(i)", "pnl"), "vs_split_pnl": pick("(iii)", "(ii)", "pnl"),
            "expiry": {"days": expiry.group(1), "on": expiry.group(2)} if expiry else None,
            "verdict": verdict.group(1).strip() if verdict else "", "figure": "adaptive_edge.png"}


# ---------------- forecast blend ----------------

def forecast_blend():
    text, src = read("forecast_blend.md")
    t = md_tables(text)
    log = json.loads((RES / "forecast_blend_log.json").read_text())
    signals = log[0]["signals"]
    traj = []
    w = log[0]["current"]
    for e in log:
        w = e["proposed"] if e["result"] == "accepted" else e["current"]
        traj.append({"day": e["day"], **{s: round(x, 4) for s, x in zip(signals, w)}})
    accepted = [{"day": e["day"], "brier_before": round(e["brier_current"], 4),
                 "brier_after": round(e["brier_proposed"], 4), "gate_games": e["gate_games"],
                 "gate_days": " – ".join(e["gate_days"]),
                 "weights": ", ".join(f"{s} {x:.2f}" for s, x in zip(signals, e["proposed"]) if x >= 0.005)}
                for e in log if e["result"] == "accepted"]
    counts = {k: sum(e["result"] == k for e in log) for k in ("accepted", "rejected", "deferred")}
    keep = ("Market 24 h", "Market 1 h", "Market at tip", "Anchor + M4 shift", "Anchor + MLP shift",
            "Anchor + GRU shift", "M4 raw", "Blend: blend (all inputs)", "Blend: no market inputs",
            "Blend: market only")
    forecast = tidy([r for r in t[0] if r["Predictor"].startswith(keep)])
    trading, tpaired = tidy(t[1]), tidy(t[2])
    gp = lambda a, metric: next((r["Difference [95% CI]"] for r in tpaired  # noqa: E731
                                 if r["Paired comparison"].startswith(a) and r["Metric"] == metric), "")
    tr = lambda a: row(trading, "Forecaster", a)  # noqa: E731
    reading = re.search(r"^Reading: (.+)$", text, re.M)
    intro = text.split("\n\n", 2)[1].strip() if text.count("\n\n") >= 2 else ""
    return {"source": src, "intro": intro, "signals": signals, "trajectory": traj, "accepted": accepted,
            "counts": counts, "forecast": forecast, "trading": trading, "paired": tpaired,
            "blend_clv": tr("Blend (all inputs)").get("CLV $ [95% CI]", ""),
            "agent_clv": tr("Anchor + M4 shift").get("CLV $ [95% CI]", ""),
            "nomarket_clv": tr("Blend, no market inputs").get("CLV $ [95% CI]", ""),
            "nomarket_pnl": tr("Blend, no market inputs").get("P&L after fees [95% CI]", ""),
            "blend_vs_agent_clv": gp("Blend (all inputs)", "clv_dollars"),
            "blend_vs_agent_pnl": gp("Blend (all inputs)", "pnl"),
            "reading": reading.group(1).strip() if reading else "",
            "final_weights": (re.search(r"Final weights: ([^.]+(?:\.\d+[^.]*)*)\.\s*$", text, re.M) or [None, ""])[1],
            "figure": "forecast_blend.png"}


# ---------------- LLM vs original (decision level) ----------------

def llm_vs_original():
    text, src = read("llm_vs_original.md")
    verdict = section(text, "Verdict")
    improves = section(verdict, "What the LLM improves") if "**What" not in verdict else \
        verdict.split("**What the LLM improves**", 1)[1].split("**What it does not improve", 1)[0]
    worse = verdict.split("**What it does not improve", 1)[1].split("**", 1)[1] if "**What it does not" in verdict else ""
    bullets = lambda s: [b.strip()[2:].strip() for b in s.strip().splitlines() if b.strip().startswith("- ")]  # noqa
    vetoes = section(text, "3. Sceptic veto reasons")
    vt = md_tables(vetoes)
    over = [r for t in md_tables(text) for r in t if "Buy proposals" in r and "LLM − mid" in r]
    sel = [r for t in md_tables(text) for r in t if "p (random ≥ LLM)" in r and r.get("Model") == "reasoner"
           and r.get("Arm", "").startswith(("C", "D"))]
    pct = lambda arm: next((r["p (random ≥ LLM)"] for r in sel if r["Arm"].startswith(arm)), None)  # noqa: E731
    wrong = re.search(r"\*\*Vetoes the sceptic got wrong\*\*[^:]*:\s*(.+?)\s+Flag rates", vetoes, re.S)
    return {"source": src, "improves": bullets(improves), "worse": bullets(worse),
            "veto_reasons": tidy(vt[0]) if vt else [], "overconfidence": tidy(over), "selection": tidy(sel),
            "random_keep_pct_c": pct("C"), "random_keep_pct_d": pct("D"),
            "vetoes_wrong": wrong.group(1).strip() if wrong else "", "figure": "llm_vs_original.png"}


# ---------------- model review (forecast switches inside the full agent) ----------------

def model_review():
    text, src = read("model_review.md")
    t = md_tables(text)
    trading, paired = tidy(t[0]), tidy([r for r in t[1]])
    for r in paired:
        r["p (one-sided, a > b)"] = r.get("p (one-sided, a > b)", "–")
    sub = section(text, "Which blend was active when")
    part = sub.split("**Model review subloop + rule learning**", 1)[1].split("**Model review + adaptive", 1)[0]
    rules = md_tables(part)[0]
    story = []
    for r in rules:
        before, after = [num(x) for x in r["CLV $ without → with"].split("→")]
        b0, b1 = [num(x) for x in r["Brier used → blend (gate days)"].split("→")]
        story.append({"rule": r["Rule"], "blend": r["Blend"], "status": r["Status"], "decided": r["Decided"],
                      "selection": r["Selection Brier (best 3)"], "gate_days": r["Gate days"],
                      "clv_without": before, "clv_with": after, "brier_used": b0, "brier_blend": b1})
    row_ = lambda p: row(trading, "Setup", p)  # noqa: E731
    verdict = re.search(r"\*\*Verdict\.\*\*\s*(.+)", text)
    return {"source": src, "trading": trading, "paired": paired, "story": story,
            "proposed": len(story), "accepted": sum(s["status"] not in ("rejected",) for s in story),
            "review_rules": row_("Model review subloop"), "review_edge": row_("Model review + adaptive"),
            "current": row_("Fixed M4"), "verdict": verdict.group(1).strip() if verdict else "",
            "figure": "model_review.png", "loop_figure": "model_review_loop.png"}


# ---------------- cumulative P&L ----------------

def pnl_curves():
    df = pd.read_csv(RES / "pnl_compare.csv")
    order = list(dict.fromkeys(df.framework))
    trades = {f: int(df[df.framework == f].trades.iloc[0]) for f in order}
    wide = df.pivot_table(index="day", columns="framework", values="cumulative_pnl").reindex(columns=order)
    wide = wide.sort_index().ffill().fillna(0.0).round(2)
    final = {f: float(wide[f].iloc[-1]) for f in order}
    return {"source": "pnl_compare.csv", "frameworks": order, "trades": trades, "final": final,
            "days": list(wide.index), "series": {f: wide[f].tolist() for f in order}, "figure": "pnl_compare.png"}


# ---------------- model cards ----------------

def src_const(path, pattern, cast=str):
    m = re.search(pattern, (ROOT / path).read_text(), re.M)
    return cast(m.group(1)) if m else None


def auc(p, y):
    r = pd.Series(p).rank()
    pos = int((y == 1).sum())
    neg = len(y) - pos
    return float((r[y.values == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def models(win, pm, fb, lv_text):
    by = {r["predictor"]: r for r in win}
    coef = json.loads((RES / "m4_coefficients.json").read_text())
    g = pd.read_csv(RES / "m4_games.csv")
    m4_auc = auc(g.m4_after.values, g.home_win)
    half = src_const("forecast/win.py", r"^HALF_LIFE = ([\d.]+)", float)
    shrink = src_const("forecast/win.py", r"^SHRINK = ([\d.]+)", float)
    feats = src_const("forecast/win.py", r"^WIN_FEATURES = \[(.+)\]")
    seq_len = src_const("forecast/gru.py", r"^SEQ_LEN = (\d+)", int)
    seq_cols = src_const("forecast/history.py", r"^SEQ_COLS = \[(.+)\]")
    sigma = src_const("forecast/inplay.py", r"sigma=([\d.]+)", float)
    m2 = {(r["model"].split(":")[0], r["target"]): r for r in pm["m2"]}
    m3 = {r["predictor"]: r for r in pm["m3"]}
    m6 = {r["model"]: r for r in pm["m6"]}
    plain = [r for t in md_tables(lv_text) for r in t if r.get("Forecaster") in ("B_chat", "B_reasoner", "mid", "A")]
    pf = {r["Forecaster"]: r for r in plain}
    fmt4 = lambda x: f"{x:.4f}"  # noqa: E731
    cards = [
        {"name": "M1a rolling average / M1b gradient boosting", "code": "forecast/baselines.py",
         "predicts": "A player's points and minutes as quantiles (10/25/50/75/90%).",
         "inputs": "M1a: the player's last 10 games. M1b: recent form plus teammates-out, rest and opponent features.",
         "architecture": "M1a: rolling quantiles. M1b: gradient-boosted trees, one per quantile.",
         "training": "Player games before 1 Feb 2026; held-out test from 1 Feb.",
         "metric": f"Points pinball {m2[('M1a', 'points')]['pinball']:.3f} (M1a) / "
                   f"{m2[('M1b', 'points')]['pinball']:.3f} (M1b); 10–90% coverage "
                   f"{pct(m2[('M1a', 'points')]['coverage_10_90'])} / {pct(m2[('M1b', 'points')]['coverage_10_90'])}.",
         "trading": "Baselines for M2; not used by the trader."},
        {"name": "M2 GRU (deep learning)", "code": "forecast/gru.py",
         "predicts": "A player's points and minutes as quantiles (10/25/50/75/90%).",
         "inputs": f"Last {seq_len} games × {len(seq_cols.split(','))} columns ({seq_cols.replace(chr(34), '')}) "
                   "plus a mask, and 17 static features (rest, teammates out, opponent pace, ...).",
         "architecture": "GRU (64) over the sequence → concat static → MLP 64, ReLU, dropout 0.1 → 2 targets × 5 "
                         "quantiles (monotone via softplus increments); pinball loss.",
         "training": "Player games before 1 Feb 2026.",
         "metric": f"Points pinball {m2[('M2', 'points')]['pinball']:.3f} vs {m2[('M1b', 'points')]['pinball']:.3f} "
                   f"(M1b) / {m2[('M1a', 'points')]['pinball']:.3f} (M1a); coverage points "
                   f"{pct(m2[('M2', 'points')]['coverage_10_90'])}, minutes {pct(m2[('M2', 'minutes')]['coverage_10_90'])}.",
         "trading": "Not in the trader: M4's missing-minutes/points inputs already carry absent players' usual output."},
        {"name": "M3 play model", "code": "forecast/play.py",
         "predicts": "Probability a player plays in the next game.",
         "inputs": "Availability features for every player on the team's recent roster, as of tip.",
         "architecture": "Standardised logistic regression (C = 1).",
         "training": "Games before 1 Feb 2026.",
         "metric": f"Brier {fmt4(m3['M3 model']['brier'])} vs {fmt4(min(m3['Played last game']['brier'], m3['Played-rate over last 10 games']['brier']))} "
                   f"(best simple rule), {m3['M3 model']['rows']:,} rows.",
         "trading": "Not in the trader (different target)."},
        {"name": "M4 win model (logistic regression)", "code": "forecast/win.py",
         "predicts": "P(home team wins) given who is out.",
         "inputs": f"{feats.replace(chr(34), '')}. Rating: exponentially weighted margin (half-life {half:g} games, "
                   f"shrunk with {shrink:g} pseudo-games); absences counted for rotation players (≥ 15 min, played "
                   "within 14 days).",
         "architecture": "Logistic regression, C = 1. Coefficients: " + ", ".join(
             f"{k.replace('_diff', '').replace('home_', '')} {v:+.4g}" for k, v in coef.items()) + ".",
         "training": "Games before 1 Feb 2026.",
         "metric": f"Brier {fmt4(by['M4 (logistic regression, absences known)']['brier'])}, accuracy "
                   f"{pct(by['M4 (logistic regression, absences known)']['accuracy'])}, AUC {m4_auc:.3f} on "
                   f"{len(g)} test games (Kalshi at tip: {fmt4(by['Kalshi at tip']['brier'])}).",
         "trading": "Its news shift (after news − before news) is added to the 24 h market anchor: the published "
                    "agent's estimate. Known flaw: the shift's 'before' uses out=[], double-counting absences already "
                    "known at anchor time."},
        {"name": "M4-NN MLP / GRU", "code": "forecast/win_nn.py",
         "predicts": "P(home wins), like M4.",
         "inputs": "MLP: M4's six features. GRU: each team's last 10 games plus the six features.",
         "architecture": "MLP 6 → 16 → 1; GRU hidden 8; chosen on 1 Dec – 31 Jan validation; 3-seed mean.",
         "training": "Games before 1 Feb 2026 (early stopping on Dec–Jan, then refit).",
         "metric": f"Brier {fmt4(by['M4-NN MLP (3-seed mean)']['brier'])} (MLP) / "
                   f"{fmt4(by['M4-NN GRU (3-seed mean)']['brier'])} (GRU); anchor + MLP shift "
                   f"{fmt4(by['Anchor + MLP shift']['brier'])}, the best forecast we have.",
         "trading": "A signal in the forecast blend and a candidate in model review; never adopted by the full agent."},
        {"name": "M6 ImpactNet", "code": "forecast/impact.py",
         "predicts": "How much the home price moves between now and tip.",
         "inputs": "Recent price moves, spread, volume, anchor, move since anchor, hours to tip, news flags.",
         "architecture": "Small neural net over the recent price sequence plus static features.",
         "training": "Decision times before 1 Feb 2026.",
         "metric": f"Null result: MAE {m6['M6 ImpactNet']['mae_c']:.3f}¢ vs {m6['zero move']['mae_c']:.3f}¢ for "
                   f"predicting no move ({m6['M6 ImpactNet']['mae_minus_zero_c']:+.3f}¢ {m6['M6 ImpactNet']['ci']}).",
         "trading": f"M6 agent placed {pm['m6_agent_trades_test']} trades; both M6 proposals in model review were rejected."},
        {"name": "Forecast blend", "code": "agents/forecast_blend.py",
         "predicts": "P(home wins) as a weighted log-odds average of 8 signals.",
         "inputs": ", ".join(fb["signals"]) + ".",
         "architecture": "Simplex weights, refit daily on settled games (exponentiated gradient); a gate accepts a "
                         "refit only if held-out Brier improves by ≥ 0.0005.",
         "training": "Walk-forward over the test period, settled games only.",
         "metric": f"Brier {fmt4(by['Learned blend (all inputs, walk-forward)']['brier'])}; "
                   f"{fb['counts']['accepted']} updates accepted. Final weights: {fb['final_weights']}.",
         "trading": "Simplified offline trader only; not wired into the full agent."},
        {"name": "In-play model", "code": "forecast/inplay.py",
         "predicts": "P(home wins) during the game.",
         "inputs": "Pre-game probability, live score margin, time remaining, news margin.",
         "architecture": f"Brownian-motion (diffusion) model of the margin, σ = {sigma:g} points per game.",
         "training": "σ fixed; pre-game prior from M4.",
         "metric": "Demo only (synthetic in-play script).",
         "trading": "Not used in any published trading result."},
        {"name": "Plain LLM as a forecaster", "code": "evaluation/llm_vs_original.py",
         "predicts": "P(home wins) from one prompt with quotes, anchors, news and M4.",
         "inputs": "The same as-of information block as the agent.",
         "architecture": "DeepSeek chat / reasoner, one call, no tools.",
         "training": "None (prompted).",
         "metric": f"Brier {pf.get('B_chat', {}).get('Brier', '–')} (chat) vs {pf.get('mid', {}).get('Brier', '–')} for "
                   f"the current mid; mean distance to the mid {pf.get('B_chat', {}).get('Mean abs. distance to mid', '–')}: "
                   "it essentially copies the market.",
         "trading": "Arm B on the 40% subsample (3–5 trades)."},
    ]
    return {"cards": cards, "m4_auc": round(m4_auc, 3)}


def frameworks(ag, mr, fb, pnl):
    full = {r["setup"]: r for r in ag["full"]}
    sub = {r["setup"]: r for r in ag["subsample"]}
    d = sub.get("D. tool agent + sceptic (DeepSeek reasoner)", {})
    blend = row(fb["trading"], "Forecaster", "Blend (all inputs)")
    rows = [
        ("Raw M4, no agent", "M4 alone", "edge > 4¢ only", "none", full["Raw M4, no agent (no anchor)"], "full"),
        ("Anchor agent", "24 h anchor + M4 shift", "4¢ edge, checks, risk", "none",
         full["Anchor agent, fixed 4¢ edge, no learning"], "full"),
        ("Split-gate learning (published agent)", "24 h anchor + M4 shift", "+ learned skip rules", "rules via split gate",
         full["Split-gate learning (published agent)"], "full"),
        ("Learned edge threshold", "24 h anchor + M4 shift", "+ learned min edge", "rules + edge thresholds",
         full["Learned edge threshold (base 4¢)"], "full"),
        ("Model review + rules", "anchor + M4 (all 6 switches rejected)", "+ learned skip rules",
         "rules + forecast switches (CLV gate)",
         {"trades": mr["review_rules"].get("Trades"), "clv_dollars": mr["review_rules"].get("CLV $ [95% CI]"),
          "pnl": mr["review_rules"].get("P&L after fees [95% CI]")}, "full"),
        ("Forecast blend", "learned blend of 8 signals", "4¢ edge, one decision per game", "weights via Brier gate",
         {"trades": blend.get("Trades"), "clv_dollars": blend.get("CLV $ [95% CI]"),
          "pnl": blend.get("P&L after fees [95% CI]")}, "simplified"),
        ("LLM reasoner + sceptic", "LLM estimate from 10 as-of tools", "validation, 4¢ edge, sceptic veto", "none",
         d, "subsample"),
        ("Never trade", "–", "–", "–", {"trades": "0", "clv_dollars": "+0", "pnl": "+0"}, "any"),
    ]
    window = {"full": "full test, full agent replay", "simplified": "full test, simplified offline trader",
              "subsample": "40% subsample (304 decision points)", "any": "–"}
    return [{"framework": n, "probability": p, "filters": f, "learning": lrn, "trades": clean_cell(x.get("trades")),
             "clv_dollars": clean_cell(x.get("clv_dollars")), "pnl": clean_cell(x.get("pnl")),
             "window": window[w]} for n, p, f, lrn, x, w in rows]


def workflow_facts():
    tools = re.findall(r'^    "([a-z_0-9]+)": \(', (ROOT / "agents" / "tools.py").read_text(), re.M)
    g = (ROOT / "agents" / "graph.py").read_text()
    banned = re.search(r'^BANNED = re\.compile\(r"(.+?)", re\.I\)', g, re.M)
    max_order = src_const("agents/graph.py", r"^MAX_ORDER = ([\d.]+)", float)
    risks = re.findall(r'return False, "([a-z_]+)"', g[g.find("def pretrade_risk"):g.find("def pretrade_risk") + 1200])
    checks = sorted(set(re.findall(r'failures\.append\(\("([a-z_]+)"', g)))
    ground = src_const("agents/tool_agent.py", r"^GROUND_BAND = ([\d.]+)", float)
    caps = re.search(r"def basic_risk\(.*max_order=([\d.]+), max_game=([\d.]+), max_day=([\d.]+)\)",
                     (ROOT / "replay.py").read_text())
    quote = src_const("replay.py", r"^MAX_QUOTE_AGE = pd\.Timedelta\(minutes=(\d+)\)", int)
    return {"tools": tools, "banned_regex": banned.group(1) if banned else "", "max_order": max_order,
            "risk_reasons": risks, "checks": checks, "ground_band": ground,
            "caps": [float(c) for c in caps.groups()] if caps else [], "max_quote_age_min": quote}


# ---------------- takeaways ----------------

def takeaways(win, ag, fb, pm, mr):
    by = {r["predictor"]: r for r in win}
    best_model = min((r for r in win if r["group"] in ("our model", "anchor + model")), key=lambda r: r["brier"])
    tip = by["Kalshi at tip"]
    m4 = by["M4 (logistic regression, absences known)"]
    m3 = next(r for r in pm["m3"] if r["predictor"] == "M3 model")
    m3_base = min(r["brier"] for r in pm["m3"] if r["predictor"] != "M3 model")
    gru = next(r for r in pm["m2"] if r["model"].startswith("M2") and r["target"] == "points")
    m1 = min((r for r in pm["m2"] if r["model"].startswith("M1") and r["target"] == "points"),
             key=lambda r: r["pinball"])
    money = lambda s: ("−" if s.startswith("-") else "+") + "$" + s.lstrip("+-")  # noqa: E731
    ci = lambda s: " ".join([money(s.split(" ", 1)[0])] + s.split(" ", 1)[1:])  # noqa: E731
    raw, anc = money(ag["raw_pnl"].split()[0]), money(ag["anchor_pnl"].split()[0])
    return [
        f"**No model beats the market at forecasting.** M4 alone scores Brier {m4['brier']:.4f}; the best "
        f"model-plus-market estimate ({best_model['predictor']}, {best_model['brier']:.4f}) only ties Kalshi at tip "
        f"({tip['brier']:.4f}).",
        f"**Anchoring on the market cuts the losses**: P&L after fees goes from {raw} (raw M4, no agent) to "
        f"{anc} (anchor agent), full test period.",
        "**Every safety or learning layer helps the same way: by trading less.** Split-gate learning, the learned "
        "edge threshold, the 4¢ edge gate and the LLM sceptic all improve CLV by cutting trades, not by picking "
        "better ones.",
        f"**Better Brier ≠ better trading.** The learned blend gains {ci(fb['blend_vs_agent_clv'])} CLV only on a "
        f"simplified trader (P&L {ci(fb['blend_vs_agent_pnl'])}, no difference). Inside the full agent, model review "
        f"proposed {mr['proposed']} lower-Brier forecasts and the CLV gate rejected {mr['proposed'] - mr['accepted']}: "
        f"the agent kept anchor + M4 ({mr['review_rules'].get('Trades')} trades, CLV $ "
        f"{mr['review_rules'].get('CLV $ [95% CI]')}, same as the published agent).",
        f"**The best deep-learning results are the player models**: M2 GRU points pinball {gru['pinball']:.3f} vs "
        f"{m1['pinball']:.3f} for the best M1 baseline; M3 Brier {m3['brier']:.4f} vs {m3_base:.4f}. "
        "Next step: earlier information (official injury reports) rather than a better model.",
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        data = git_show(name, binary=True) if name in ON_REF else None
        (FIG / name).write_bytes(data if data else (RES / name).read_bytes())
    win = win_models()
    pm = player_models()
    ag = agents()
    fb = forecast_blend()
    mr = model_review()
    pnl = pnl_curves()
    r = mr["review_rules"]
    ag["full"].insert(-1, tidy([{"setup": "Model review subloop + rule learning", "trades": r.get("Trades"),
                                 "mean_clv": r.get("Mean CLV/contract [95% CI]"), "clv_dollars": r.get("CLV $ [95% CI]"),
                                 "pnl": r.get("P&L after fees [95% CI]"), "source": "model_review.md"}])[0])
    out = {"agent": "results", "win_models": win, "players": pm, "agents": ag, "edge_gate": edge_gate(),
           "adaptive_edge": adaptive_edge(), "forecast_blend": fb, "llm_vs_original": llm_vs_original(),
           "model_review": mr, "pnl": pnl, "models": models(win, pm, fb, read("llm_vs_original.md")[0]),
           "frameworks": frameworks(ag, mr, fb, pnl), "workflow": workflow_facts(),
           "takeaways": takeaways(win, ag, fb, pm, mr),
           "sources": sorted({r["source"] for r in win} | set(pm["sources"]) |
                             {"significance.md", "adaptive_edge.md", "llm_agent.md", "edge_gate.md",
                              "forecast_blend.md", "forecast_blend_log.json", "llm_vs_original.md",
                              "gate_audit.md", "model_review.md", "pnl_compare.csv", "m4_coefficients.json",
                              "m4_games.csv"}),
           "built_at": pd.Timestamp.now(tz="UTC").isoformat()}
    text = json.dumps(out, indent=1, default=str)
    if SECRET.search(text):
        raise ValueError("results.json: looks like a secret")
    if re.search(r'"nan"|: NaN', text):
        raise ValueError("results.json: contains NaN")
    (OUT / "results.json").write_text(text)
    print(f"results.json {len(text) / 1024:.0f} KB; figures: {', '.join(FIGURES)}")
    return out


if __name__ == "__main__":
    main()
