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
FIGURES = ("edge_gate.png", "adaptive_edge.png", "forecast_blend.png", "llm_vs_original.png")
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


# ---------------- takeaways ----------------

def takeaways(win, ag, fb, pm):
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
        f"**The learned forecast blend** improves CLV by {ci(fb['blend_vs_agent_clv'])} vs the agent on the "
        f"simplified trader, with no detectable P&L change ({ci(fb['blend_vs_agent_pnl'])}).",
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
    out = {"agent": "results", "win_models": win, "players": pm, "agents": ag, "edge_gate": edge_gate(),
           "adaptive_edge": adaptive_edge(), "forecast_blend": fb, "llm_vs_original": llm_vs_original(),
           "takeaways": takeaways(win, ag, fb, pm),
           "sources": sorted({r["source"] for r in win} | set(pm["sources"]) |
                             {"significance.md", "adaptive_edge.md", "llm_agent.md", "edge_gate.md",
                              "forecast_blend.md", "forecast_blend_log.json", "llm_vs_original.md",
                              "gate_audit.md"}),
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
