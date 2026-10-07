"""Build the 10-minute presentation deck from the figures in presentation/figures/.

    python presentation/build_deck.py            # writes presentation/nba_agent_deck.pptx
    python presentation/build_deck.py --check    # also prints a fit report for every text box

Narrative ("Market-Graded Learning"): the market's closing price grades every component of our
agent, and only the components that defer to the market survive. If a disciplined agent with our
data cannot beat the close, a consumer cannot either, so the product educates and protects.
Every number on the slides lives in K below, with its source, so it can be updated in one place.
New agentic results: fill K["agentic"] from evaluation/results/*.md and rebuild; the build copies
SYNC figures from evaluation/results/ and lists result files that exist but are still "pending".
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RESULTS = HERE.parent / "evaluation" / "results"
OUT = HERE / "nba_agent_deck.pptx"
SYNC = ["injury_timing.png", "m6_robustness.png", "orchestrator.png"]   # copied from RESULTS when present

W, H = 13.333, 7.5
MARGIN = 0.5
TOP = 1.2          # below the title bar
BOTTOM = 6.95      # above the footer
FONT = "Arial"

NAVY = RGBColor(0x1E, 0x29, 0x3B)
BLUE = RGBColor(0x25, 0x63, 0xEB)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
RED = RGBColor(0xDC, 0x26, 0x26)
GREY = RGBColor(0x64, 0x74, 0x8B)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
TEXT = RGBColor(0x0F, 0x17, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FOOTER = "DASC7606C Group 12 · Market-Graded Learning: the closing price grades every component"
PENDING = "pending"

# Key numbers. Sources: README "Results" (R); evaluation/results/consumer_fairness.md (F);
# m4_vs_market.csv (M); trade funnel / fairness_run_costs.csv (T); walkforward_summary.md,
# significance.md, holdout.md (W); m6_heldout.csv, m6_ablation.md, README M6 section (S);
# m4_calibration.md (C); m6_robustness.md (G); injury_timing.md (I).
K = {
    "games": "1,316", "price_rows": "~3.9M",                                   # frozen markets / prices
    "cost_c": "1.92", "cost_pct": "5.0%", "half_spread_c": "0.53", "fee_c": "1.39",  # F1
    "cost_cheap_pct": "9–14%",                                                 # F1, buckets under 20¢
    "full_mid": "+$36", "full_pnl": "−$32",                                    # F2, Feb–Apr test window
    "decisions": "771", "gap4": "153", "trades": "74",                         # T
    "beat_close": "27 (36%)", "pass_share": "90%",                             # T; 1 − 74/771
    "disc_games": "193", "disc_before": "1.80", "disc_after": "0.18", "disc_share": "91%",  # F3
    "m4_games": "501", "m4_brier": "0.186", "mkt_brier": "0.164",              # R/M: both 1 h before tip
    "m4_brier_tip": "0.183", "mkt_brier_tip": "0.163",                         # M: absences known / at tip
    "policy": "9 / 9",                                                         # R, policy_tests.csv
    # walk-forward, season Nov – 12 Apr, M4 retrained monthly, day-clustered bootstrap (W)
    "wf_full_n": "157", "wf_full_pnl": "−$25", "wf_full_ci": "−$569 to +$564", "wf_full_p": "0.53",
    "wf_nolearn_n": "276", "wf_nolearn_pnl": "−$407", "wf_nolearn_ci": "−$1,324 to +$553",
    "wf_raw_n": "711", "wf_raw_pnl": "−$2,475", "wf_raw_ci": "−$3,995 to −$845",
    "wf_anchor_gain": "+$2,450", "wf_anchor_p": "0.002", "wf_learn_p": "0.13",
    "wf_learn_clv": "+$39", "wf_learn_clv_p": "0.001",
    "ho_games": "87", "ho_nolearn": "6 trades, −$74", "ho_raw": "+$450",       # W, play-off holdout
    # M6 market-impact model (S) and its robustness grid (G)
    "m6_mae": "0.867", "zero_mae": "0.833", "m6_coef": "0.025", "m6_coef_pct": "2.5%",
    "m6_forced_n": "501", "m6_forced_pnl": "−$652",
    "m6_cfgs": "9", "m6_seeds": "3", "m6_best_gap": "+0.017¢", "m6_worst_gap": "+0.036¢", "m6_cis": "18 of 18",
    "m6_need": "5.5¢",
    # M4 recalibration on the 501 test games, absences known (C)
    "cal_raw": "0.183", "cal_platt": "0.185", "cal_iso": "0.184", "cal_mkt": "0.164",
    "cal_slope_early": "0.78", "cal_slope_late": "1.33",
    # M4-NN on the same 501 games, absences known, 3-seed mean (win_nn.md); M2 player GRU (m2_heldout.csv)
    "nn_mlp": "0.184", "nn_mlp_vs_m4": "+0.0015 [−0.006, +0.008]", "nn_gru": "0.186",
    "nn_anchor": "0.163", "nn_anchor_vs_agent": "−0.003 [−0.008, +0.001]", "nn_anchor_vs_mkt": "−0.001 [−0.006, +0.005]",
    "agent_est": "0.166", "m2_gru": "1.55", "m2_gbm": "1.59",
    # Injury-report timing, 193 test games, key absent player (I)
    "it_before": "67%", "it_ci": "47–85%", "it_nolag": "49%", "it_lead": "6", "it_mid": "24%", "it_after": "9%",
    # cumulative P&L after fees by settlement date (evaluation/results/cumulative_pnl.csv)
    "cum_nolearn_n": "129", "cum_nolearn_pnl": "−$247", "cum_raw_n": "366", "cum_raw_pnl": "−$2,087",
    "cum_full_low": "−$131",
    # League (agents/league.py), demo league league_data/friday-league.json
    "lg_bankroll": "$1,000", "lg_min_trades": "5", "lg_slate": "10", "lg_bot_pnl": "+$15", "lg_bot_clv": "−0.63¢",
    # Agentic upgrades (slide 11). Gate audit, split vs legacy gate, test window (gate_audit.md)
    "ga_split": "5 of 11", "ga_legacy": "0 of 24", "ga_trades": "129 → 53", "ga_clv_gain": "+$18 [+1, +34]",
    "ga_worst_day": "−$54",
    "pl_rate": "57%",                                                          # gate_placebo.md, split gate
    # coach_sim.md, compliance c = 1, Δ CLV $ with − without Coach [slate CI]
    "co_over": "+$88 [+65, +111]", "co_long": "+$72 [+53, +90]", "co_caut": "+$24 [+14, +36]",
    "co_chase": "+$17 [−4, +33]", "co_cut": "66–87%", "co_flag": "99%",
    # llm_agent.md, 40% subsample of test decision points
    "llm_points": "304", "llm_plain_n": "5", "llm_plain_clv": "−$1", "llm_det_n": "53", "llm_det_clv": "−$19",
    # Tool agent + sceptic (DeepSeek re-run): None shows "pending". Fill with one short string, e.g.
    # "8 trades, CLV $ −$2 [−5, +1] vs never; sceptic vetoed 3 of 9"
    "llm_tool": None,
    # Team contributions (slide 13): PRs on GitHub; names as the author appears on the PR.
    "team": [
        ("Pregame news and fair-odds loop", "#7", "codingMiiichael", "Multi-source news polling → fair odds"),
        ("Player-feature experiment", "#8", "Catharine Li", "Isolated player features + probability calibration"),
        ("In-play news and odds", "#9", "codingMiiichael", "Independent in-play updates (synthetic scenarios)"),
        ("Live Polymarket markets", "#11, #12", "shidafuyang-alt, hijojo", "Read-only live prices and page"),
        ("Replay, agent and evaluation", "branch", "Michael Ngan", "As-of replay, M4 / M6, LangGraph agent, gate, "
         "walk-forward, Coach, League"),
    ],
    # Worked decision for slide 4 (replay trace, test window)
    "example": {"game": "CLE at POR", "news": "Deni Avdija is out, 32.8 minutes and 26 points a game",
                "shift": "−8.7", "anchor": "42.5%", "est": "33.8%", "price": "55¢", "fee": "1.73¢",
                "gap": "9.5¢", "qty": "36", "close": "62.5¢", "clv": "+7.5¢", "pnl": "+$15.57"},
}

TRADE_LESS = ("Every agentic upgrade, graded by the market, converges on the same answer: trade less. The market "
              "is hard to beat after fees, which is exactly why the product is education, not a trading bot.")

# Agentic upgrades (slide 11): (component, what it is, result files, result). Result None = pending;
# a result starting "Not run" is shown muted. The tool-agent row is K["llm_tool"].
K["agentic"] = [
    ("Gate audit", "Rule gate judged on held-out days, by CLV $", ["gate_audit"],
     f"Kept {K['ga_split']} (legacy {K['ga_legacy']}); trades {K['ga_trades']}; CLV $ {K['ga_clv_gain']}, "
     f"per trade no better; never trading still wins"),
    ("Placebo rules", "Random rules through the same gate", ["gate_placebo"],
     f"Random rules pass {K['pl_rate']}, same as the reviewer's: learning works by trading less"),
    ("Kill switch", "$100 daily realised-loss stop", ["gate_audit"],
     f"Enforced, never tripped at $20 stakes (worst day {K['ga_worst_day']}); property-tested"),
    ("Fractional Kelly", "¼-Kelly with the fee vs flat $20", ["kelly"], "Not run (time)"),
    ("Coach, simulated users", "Biased personas follow nudges (compliance 1.0)", ["coach_sim"],
     f"CLV $ {K['co_chase'].split(' ')[0]} to {K['co_over'].split(' ')[0]} for all 4 personas; trades "
     f"−{K['co_cut']}; flags {K['co_flag']}, so the gain is trading less"),
    ("Plain LLM", "One LLM call per decision, no tools", ["llm_agent"],
     f"{K['llm_plain_n']} trades in {K['llm_points']} decisions, CLV $ {K['llm_plain_clv']} = never trading "
     f"(deterministic agent: {K['llm_det_n']} trades, {K['llm_det_clv']})"),
    ("LLM tool agent + sceptic", "Picks as-of tools; code checks numbers; sceptic can veto", ["llm_agent"],
     K["llm_tool"]),
    ("Orchestrator", "Pregame → trader → Coach → briefs, one as-of time", [],
     "Built and tested (diagram); no new trading evaluation"),
    ("Learned policy", "Trade / pass model, selected on Nov–Jan", ["learned_policy"],
     "Selection chose never trade; test not scored"),
    ("Rule DSL + memory", "LLM rules in a safe DSL; recall of similar past trades", ["rule_dsl", "memory"],
     "Built and unit-tested; not evaluated"),
]

# Data, measured from data/frozen/*.parquet by evaluation/data_overview.py (data_overview.csv).
D = {
    "price_rows": "3,938,368", "price_first": "20 Oct 2025", "price_last": "14 Jun 2026",
    "markets": "2,632", "settlements": "2,634", "priced_games": "1,316",
    "games": "3,962", "seasons": "2023-24 to 2025-26",
    "box_rows": "85,422", "players": "818",
    "news_rows": "3,371", "news_games": "1,772", "news_last": "7 May 2026",
    "test_games": "501", "holdout_games": "87",
}

# External published research (NOT our measurement); full references in presentation/README.md.
# L = Wardle et al., Lancet Public Health Commission on gambling, Lancet Public Health 2024; 9: e950–94.
# H = Hollenbeck, Larsen & Proserpio, "The Financial Consequences of Legalized Sports Gambling",
#     SSRN 4903302 (2024, rev. 2025). B = Baker et al., "Gambling Away Stability", NBER WP 33108 (2024).
EXT = {
    "states": "38",              # H: US states legal after the 2018 Supreme Court ruling
    "adults_pct": "46%",         # L: 46.2% of adults gambled in the past year (global estimate)
    "disorder": "80M",           # L: ~80 million adults with gambling disorder or problematic gambling
    "credit_online": "2.75",     # H: credit-score fall with online access (0.8 pts for any legal betting)
    "bankrupt": "10%",           # H: bankruptcies ~10% higher with online access
    "invest": "$0.99",           # B: $1 of online betting reduces net investment by $0.99
}
# Market growth and regulation (docs/market_regulation.md has full sources and dates).
MKT = {
    "pm_may": "$26B", "pm_jul": "$53B",          # Pew Research, 23 Sep 2026: Kalshi + Polymarket monthly
    "k_sep": "$61B", "k_yoy": "46×",             # NEXTPredict / TickerTracker: Sep 2026 notional; Aug daily avg YoY
    "k_val": "$22B", "k_val_talks": "~$40B",     # Kalshi release 7 May 2026; Reuters 29 Sep 2026 (talks)
    "p_val": "$21B",                             # Bloomberg 31 Aug 2026
    "aga_jul": "$11.6B", "k_sports_jul": "$31B", # AGA tracker, July 2026 handle; Pew, Kalshi July sports
    "hood_users": "~2M", "hood_q2": "13.6B",     # Robinhood Q2 2026 results, 29 Jul 2026
}

_fit_report: list[tuple[int, str, float, float]] = []


# ---------------------------------------------------------------- text fitting

def _measurer():
    """Return a function (text, pt, bold) -> width in inches, using Arial metrics if PIL has them."""
    try:
        from PIL import ImageFont
        paths = {False: "/System/Library/Fonts/Supplemental/Arial.ttf",
                 True: "/System/Library/Fonts/Supplemental/Arial Bold.ttf"}
        cache = {}

        def width(text, pt, bold=False):
            key = (bold, pt)
            if key not in cache:
                cache[key] = ImageFont.truetype(paths[bold], size=int(pt * 10))
            return cache[key].getlength(text) / 10 / 72
        width("x", 18)
        return width
    except Exception:
        return lambda text, pt, bold=False: len(text) * pt * 0.55 / 72


_width = _measurer()


def _lines(text, pt, box_w, bold=False):
    """Number of wrapped lines for one paragraph in a box of width box_w inches."""
    n, line = 1, ""
    for word in text.split(" "):
        trial = f"{line} {word}".strip()
        if _width(trial, pt, bold) * 1.04 > box_w and line:
            n, line = n + 1, word
        else:
            line = trial
    return n


# ---------------------------------------------------------------- primitives

def _run(p, text, pt, color=TEXT, bold=False, italic=False, font=FONT):
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(pt), bold, italic, font
    r.font.color.rgb = color
    return r


def textbox(slide, x, y, w, h, paras, pt=20, color=TEXT, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, bullets=False, space=8, slide_no=0, label=""):
    """paras: list of str, or (str, dict) with keys level/bold/color/pt/italic.

    A str may contain **bold** spans. Records the estimated height for --check.
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    inset = 0.08
    tf.margin_left = tf.margin_right = Inches(inset)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    used = 0.08
    for i, item in enumerate(paras):
        text, opt = (item, {}) if isinstance(item, str) else item
        level = opt.get("level", 0)
        size = opt.get("pt", pt if level == 0 else pt - 2)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        indent = 0.0
        if bullets and not opt.get("heading"):
            indent = 0.32 + 0.35 * level
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(indent))))
            pPr.set("indent", str(-Emu(Inches(0.28))))
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar",
                                 {"char": "•" if level == 0 else "–"})
            pPr.append(bu)
        parts = text.split("**")
        for j, part in enumerate(parts):
            if part:
                _run(p, part, size, opt.get("color", color),
                     bold=opt.get("bold", False) or j % 2 == 1, italic=opt.get("italic", False))
        plain = text.replace("**", "")
        n = _lines(plain, size, w - 2 * inset - indent, opt.get("bold", False))
        used += n * size * 1.2 / 72 + space / 72
    _fit_report.append((slide_no, label or (paras[0] if isinstance(paras[0], str) else paras[0][0])[:40],
                        round(used, 2), h))
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def picture(slide, name, x, y, w, h, align="center"):
    """Place figures/<name> as large as possible inside the box, keeping its aspect ratio."""
    from PIL import Image
    path = FIG / name
    with Image.open(path) as im:
        pw, ph = im.size
    scale = min(w / pw, h / ph)
    fw, fh = pw * scale, ph * scale
    fx = x + (w - fw) / 2 if align == "center" else x
    fy = y + (h - fh) / 2
    slide.shapes.add_picture(str(path), Inches(fx), Inches(fy), Inches(fw), Inches(fh))
    return fx, fy, fw, fh


def table(slide, rows, x, y, col_w, row_h=0.5, pt=16, header_fill=NAVY, emphasis_row=None,
          header_h=None, center_cols=(), muted=(), label="table"):
    """Rows grow in PowerPoint when a cell wraps, so --check records the estimated total height."""
    n = len(slide.part.package.presentation_part.presentation.slides)
    header_h = header_h or row_h
    need = 0.0
    for i, row in enumerate(rows):
        lines = max(_lines(str(v).replace("**", ""), pt, cw - 0.16, bold=i == 0) for v, cw in zip(row, col_w))
        need += max(lines * pt * 1.2 / 72 + 0.1, header_h if i == 0 else row_h)
    _fit_report.append((n, label, round(need, 2), header_h + row_h * (len(rows) - 1)))
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_w)), Inches(header_h + row_h * (len(rows) - 1)))
    t = shape.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(header_h if i == 0 else row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            elif i == emphasis_row:
                c.fill.fore_color.rgb = RGBColor(0xDC, 0xFC, 0xE7)
            else:
                c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j in center_cols else PP_ALIGN.LEFT
            parts = str(val).split("**")
            colour = WHITE if i == 0 else GREY if (i, j) in muted else TEXT
            for k, part in enumerate(parts):
                if part:
                    _run(p, part, pt, colour, bold=(i == 0 or k % 2 == 1), italic=(i, j) in muted)
    return shape


# ---------------------------------------------------------------- slide chrome

def new_slide(prs, title, notes, number):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, 0.95, NAVY)
    rect(s, 0, 0.95, W, 0.06, ORANGE)
    textbox(s, MARGIN, 0.12, W - 2 * MARGIN, 0.75, [(title, {"bold": True})], pt=30, color=WHITE,
            anchor=MSO_ANCHOR.MIDDLE, space=0, slide_no=number, label="title")
    textbox(s, MARGIN, 7.05, 10, 0.35, [FOOTER], pt=11, color=GREY, space=0, slide_no=number, label="footer")
    textbox(s, W - MARGIN - 1.0, 7.05, 1.0, 0.35, [str(number)], pt=11, color=GREY,
            align=PP_ALIGN.RIGHT, space=0, slide_no=number, label="page")
    s.notes_slide.notes_text_frame.text = notes
    return s


def callout(slide, x, y, w, h, text, pt=20, fill=RGBColor(0xFE, 0xF3, 0xC7), color=TEXT, n=0):
    rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.12
    textbox(slide, x + 0.1, y, w - 0.2, h, [text], pt=pt, color=color, anchor=MSO_ANCHOR.MIDDLE,
            space=0, slide_no=n, label="callout")


def card(slide, x, y, w, h, heading, lines, accent, pt=18, n=0):
    rect(slide, x, y, w, h, WHITE, line=accent, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.06
    rect(slide, x, y, 0.12, h, accent)
    textbox(slide, x + 0.2, y + 0.08, w - 0.3, 0.5, [(heading, {"bold": True, "color": accent})],
            pt=pt + 2, space=0, slide_no=n, label=heading)
    textbox(slide, x + 0.2, y + 0.58, w - 0.3, h - 0.66, lines, pt=pt, bullets=True, space=6,
            slide_no=n, label=heading + " body")


# ---------------------------------------------------------------- agentic results

def sync_figures():
    """Copy SYNC figures from evaluation/results/ into figures/ when they exist there."""
    for name in SYNC:
        src = RESULTS / name
        if src.exists() and (not (FIG / name).exists() or src.stat().st_mtime > (FIG / name).stat().st_mtime):
            shutil.copy2(src, FIG / name)


def agentic_status():
    """(filled, pending, result files that exist but whose row is still pending)."""
    filled, pending, waiting = [], [], []
    for name, _, files, result in K["agentic"]:
        (filled if result else pending).append(name)
        if not result:
            waiting += [f"{f}.md" for f in files if (RESULTS / f"{f}.md").exists()]
    return filled, pending, sorted(set(waiting))


# ---------------------------------------------------------------- slides

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    CW = W - 2 * MARGIN

    # 1 Title ---------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 4.55, W, 0.08, ORANGE)
    textbox(s, 0.9, 0.95, W - 1.8, 1.2, [("Market-Graded Learning", {"bold": True})],
            pt=48, color=WHITE, slide_no=1, label="title")
    textbox(s, 0.9, 2.15, W - 1.8, 2.2,
            ["The market's closing price grades every part of our NBA trading agent. Only the parts that "
             "defer to the market survive, so a consumer betting against it is at a structural disadvantage."],
            pt=24, color=RGBColor(0xCB, 0xD5, 0xE1), slide_no=1, label="subtitle")
    textbox(s, 0.9, 4.85, W - 1.8, 1.9,
            ["Market-Graded Learning for NBA Game-Impact Intelligence",
             "DASC7606C (COMP7606) · Group 12 · Track 2: Agentic Framework Design",
             "Presenter: Ngan Tsz Sui · Friday 9 October 2026"],
            pt=20, color=WHITE, slide_no=1, label="meta")
    s.notes_slide.notes_text_frame.text = (
        "Good morning, we are Group 12. Our title is Market-Graded Learning, and it is also our thesis. We built an "
        "AI agent that trades NBA prediction markets, and we let the market's closing price grade every part of "
        "it: the models, the learned rules and the LLM. The finding is that only the parts that defer to the "
        "market survive. If a disciplined agent cannot beat the close, a consumer cannot either, so our product "
        "educates and protects instead of giving tips.")

    # 2 The problem -----------------------------------------------------------
    n = 2
    s = new_slide(prs, "The problem: surging, under-protected betting markets", (
        f"Three parts. First, volume is surging. Pew Research, using The Block's data, found that monthly volume "
        f"on Kalshi and Polymarket more than doubled from {MKT['pm_may']} in May to {MKT['pm_jul']} in July 2026, "
        f"mostly sports. Kalshi alone traded a record {MKT['k_sep']} notional in September 2026, about "
        f"{MKT['k_yoy']} its daily average a year earlier (TickerTracker, via NEXTPredict). Notional counts "
        f"each contract at one dollar, so cash paid is lower, but in July Kalshi's sports notional of "
        f"{MKT['k_sports_jul']} compared with {MKT['aga_jul']} of handle at all US sportsbooks (AGA). It reaches "
        f"retail users through brokers: Robinhood reported {MKT['hood_q2']} event contracts in Q2 2026 and nearly "
        f"2 million prediction-market customers; Coinbase routes to Kalshi since January 2026. Investors value "
        f"Kalshi at {MKT['k_val']} (May 2026, talks at about $40 billion reported by Reuters on 29 September) "
        f"and Polymarket at {MKT['p_val']} (Bloomberg, 31 August). "
        f"Second, price protection is thin. An exchange has no house edge, but every taker pays a fee, "
        f"0.07 times p times one minus p per contract on Kalshi, plus the spread, and trades against professional "
        f"market makers, including Kalshi's own affiliate, Kalshi Trading, which the CFTC has proposed to "
        f"restrict. Kalshi has been CFTC-regulated since 2020 and self-certified sports contracts in January 2025; "
        f"the CFTC did not block them. CFTC core principles protect market integrity, surveillance and "
        f"manipulation, but nobody checks that the price a retail user pays is fair, and there is no federal "
        f"responsible-gambling duty; sports contracts are open at 18, not 21. States are fighting back: the Third "
        f"Circuit said New Jersey cannot regulate Kalshi, the Ninth Circuit in August said Nevada can. Our own "
        f"measurement shows what this gap costs: about {K['cost_pct']} per trade, and {K['disc_share']} of the "
        f"price move before the public news. Third, published research shows harm: about {EXT['disorder']} adults "
        f"with gambling disorder (Lancet Public Health 2024); online betting lowers credit scores by "
        f"{EXT['credit_online']} points and raises bankruptcies by about {EXT['bankrupt']} (Hollenbeck, Larsen and "
        f"Proserpio, SSRN 4903302). Full sources and dates: docs/market_regulation.md."), n)
    cw3 = (CW - 0.5) / 3
    pillars = [
        ("1  Surging volume", BLUE, [
            f"Kalshi + Polymarket: **{MKT['pm_may']} → {MKT['pm_jul']}** a month, May → Jul 2026 ¹",
            f"Kalshi Sep 2026: **{MKT['k_sep']}** notional, ~{MKT['k_yoy']} a year earlier ²",
            f"Sold in Robinhood ({MKT['hood_users']} users) and Coinbase apps ³",
            f"Valued at {MKT['k_val']} (Kalshi), {MKT['p_val']} (Polymarket) ⁴",
        ]),
        ("2  Thin price protection", ORANGE, [
            "No house edge, but fees, spreads and pro market makers, incl. Kalshi's own ⁵",
            "CFTC checks integrity, not fair prices or gambling harm; 18+ ⁶",
            "Courts split on state power (3rd vs 9th Cir.) ⁷",
            f"**Ours: ~{K['cost_pct']} cost per trade; {K['disc_share']} of the move before the news**",
        ]),
        ("3  Betting causes harm", RED, [
            f"~{EXT['disorder']} adults with gambling disorder or problem gambling ⁸",
            f"Online betting: credit scores −{EXT['credit_online']} pts, bankruptcies +~{EXT['bankrupt']} ⁹",
        ]),
    ]
    for i, (head, col, lines) in enumerate(pillars):
        card(s, MARGIN + i * (cw3 + 0.25), TOP + 0.08, cw3, 4.07, head, lines, col, pt=16, n=n)
    callout(s, MARGIN, TOP + 4.24, CW, 0.55,
            "Consumers need education and protection, not betting tips", pt=20, n=n)
    textbox(s, MARGIN, 6.03, CW, 0.92, [
        "¹ Pew Research, 23 Sep 2026 (The Block data)   ² NEXTPredict / TickerTracker, Sep–Oct 2026   "
        "³ Robinhood Q2 2026 results; Coinbase, 28 Jan 2026   ⁴ Kalshi, 7 May 2026; Bloomberg, 31 Aug 2026   "
        "⁵ Kalshi fee schedule & Rule 2.12; CFTC affiliate proposal 2026   ⁶ DCM core principles; NBA letter "
        "to CFTC, 30 Apr 2026   ⁷ KalshiEX v. Flaherty (3d Cir., 6 Apr 2026); KalshiEX v. Assad (9th Cir., "
        "28 Aug 2026)   ⁸ Lancet Public Health 2024   ⁹ Hollenbeck, Larsen & Proserpio 2024 (SSRN)   "
        "Full list: docs/market_regulation.md"], pt=10, color=GREY, space=0, slide_no=n, label="refs")

    # 3 Research question and what we built ----------------------------------------
    n = 3
    s = new_slide(prs, "Research question, and what we built to test it", (
        "Our research question: can a consumer, even one armed with an AI agent and public news, beat the "
        "market? We built the strongest consumer we could: public news, trained models including a neural net "
        "trained on the market's own moves, a LangGraph agent with code risk limits and gated rule learning, and "
        "an LLM tool agent. A replay walks the season day by day, shows the agent only what was public, and fills "
        "every order at the recorded ask plus the Kalshi fee. The answer, tested with confidence intervals, is "
        "no: never trading wins."), n)
    callout(s, MARGIN, TOP + 0.1, CW, 0.95,
            "Can a consumer, even one armed with an AI agent and public news, beat the market?", pt=26,
            fill=RGBColor(0xDB, 0xEA, 0xFE), color=NAVY, n=n)
    textbox(s, MARGIN, TOP + 1.2, 5.7, 3.3, [
        ("We built the strongest consumer we could:", {"bold": True, "color": NAVY, "heading": True}),
        "Public news: ESPN inactive lists and box scores, 3 seasons",
        "Trained models, incl. a neural net trained on the market's own moves",
        "LangGraph agent; code risk limits; gated rule learning; LLM tools",
        f"Graded on {K['games']} Kalshi games ({K['price_rows']} price rows); every order pays the recorded "
        f"ask + fee",
    ], pt=17, bullets=True, space=6, slide_no=n)
    picture(s, "product_workflow.png", 6.35, TOP + 1.2, W - MARGIN - 6.35, 3.3)
    rect(s, MARGIN, TOP + 4.65, CW, 0.95, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.12
    textbox(s, MARGIN + 0.3, TOP + 4.65, CW - 0.6, 0.95, [
        "**Answer: no.** Tested with confidence intervals, never trading wins, so the product educates and "
        "protects instead."], pt=21, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, space=0, slide_no=n, label="answer")

    # 4 How the agent works ---------------------------------------------------------
    n = 4
    ex = K["example"]
    s = new_slide(prs, "How the agent works: decide, learn, graded by the close", (
        f"This is the agent. The top row is the decide loop, a LangGraph graph. A trigger, news or the clock an "
        f"hour before tip, wakes it; it then chooses which as-of tools to call, prices, news, rotations, and our "
        f"models M4 and M6, which only see what was public. It estimates a probability, the market anchor plus "
        f"the model's news shift, or an LLM analyst; trades only if the gap after spread and fee is at least 4 "
        f"cents; an LLM sceptic asks whether the news is already priced in and can veto; code re-checks every "
        f"number; risk caps and a kill switch the LLM cannot override; then a confirm step, and an order or, "
        f"about {K['pass_share']} of the time, a pass with the reason. One example: {ex['game']}. "
        f"{ex['news']}. M4's shift is {ex['shift']} points, so the anchor of {ex['anchor']} becomes "
        f"{ex['est']}. The agent bought POR “no” at {ex['price']}, fee {ex['fee']}, a gap of {ex['gap']} after "
        f"costs, {ex['qty']} contracts. The close was {ex['close']}: closing-line value {ex['clv']}, profit "
        f"{ex['pnl']}. The bottom row is the learn loop: at the close every trade is settled and graded by CLV, "
        f"a reviewer proposes one rule, and the gate keeps it only if it helps on held-out days; kept rules feed "
        f"back into the decision. The Coach and League turn the same analysis into education. The closing price "
        f"grades every box."), n)
    picture(s, "agent_architecture.png", MARGIN, TOP - 0.05, CW, 4.6)
    cw4 = (CW - 0.45) / 4
    for i, txt in enumerate([
            "**Autonomy:** it decides when to act, what to look at, and whether to trade",
            f"**Safety:** code limits and a kill switch; **{K['policy']}** planted violations blocked",
            "**Learning:** rules graded by the market on held-out days, placebo-audited",
            "**People in the loop:** humans confirm; Coach and League teach with its analysis"]):
        callout(s, MARGIN + i * (cw4 + 0.15), 5.7, cw4, 1.15, txt, pt=14, n=n)

    # 5 Market-Graded Learning: the thesis ------------------------------------------
    n = 5
    s = new_slide(prs, "Market-Graded Learning: the closing price is the grader", (
        f"This is what the title means. Every component is a hypothesis that must pass an evidence gate on recorded "
        f"prices, and the grader is the market's closing price. The raw win model fails: {K['wf_raw_pnl']} over "
        f"the season, and recalibrating it does not help. The neural impact model fails: all {K['m6_cfgs']} "
        f"configurations are worse than predicting no move. What passes defers to the market: the market anchor, "
        f"{K['wf_anchor_gain']} with p {K['wf_anchor_p']}, and gated learning, which passes only by trading less. "
        f"The newer agentic parts face the same gate; slide 11. So the finding is the product: if a disciplined "
        f"agent cannot beat the close, a consumer cannot either. We say “we pass” as often as “we trade”."), n)
    rows = [["Component", "Market's test", "Verdict"],
            ["Raw win model (M4) trading", "Beat never trading", f"**Failed:** {K['wf_raw_pnl']}, CI below 0"],
            ["Recalibrated M4", "Beat the market's Brier score",
             f"**Failed:** {K['cal_platt']} vs market {K['cal_mkt']}"],
            ["M6 neural impact model", "Beat “the price won't move”",
             f"**Failed:** {K['m6_cfgs']} of {K['m6_cfgs']} configurations worse"],
            ["Market anchor", "Beat the raw model", f"**Passed:** {K['wf_anchor_gain']}, p = {K['wf_anchor_p']}"],
            ["Gated rule learning", "Beat no learning on CLV $",
             f"**Passed (trades less):** {K['wf_learn_clv']} CLV, p = {K['wf_learn_clv_p']}"],
            ["LLM agent, Coach, gate audit …", "Same gates (slide 11)", "**Help only by trading less**"],
            ["A consumer", f"Beat ~{K['cost_c']}¢ cost; news {K['disc_share']} priced", "**Structurally hard**"]]
    table(s, rows, MARGIN, TOP + 0.1, [3.55, 4.05, CW - 7.6], row_h=0.55, pt=16, header_h=0.48, emphasis_row=4)
    callout(s, MARGIN, 5.95, CW, 0.85,
            "Only the components that defer to the market survive, so the product teaches, lets people practise "
            "and enforces limits", pt=19, n=n)

    # 6 Data -----------------------------------------------------------------
    n = 6
    s = new_slide(prs, "Data: five public sources, split in time, no look-ahead", (
        f"The data is all public. From Kalshi: every NBA game-winner market for 2025-26, {D['markets']} markets on "
        f"{D['priced_games']} games, {D['price_rows']} one-minute rows of bid, ask and volume, and settlements. From "
        f"ESPN: three seasons of games, {D['box_rows']} box-score rows and {D['news_rows']} inactive-list entries, "
        f"which are our news. Models train on games before 1 February; the agent develops on November to January; "
        f"the test period is 1 February to 12 April, {D['test_games']} games; the walk-forward retrains monthly; "
        f"and the {D['holdout_games']} play-off games are a holdout scored once. The replay shows the agent only "
        f"what was public, counts each inactive list as news 30 minutes before tip, and fills at the recorded ask "
        f"plus the Kalshi fee."), n)
    rows = [["Source", "What it contains", "Rows", "Used for"],
            ["Kalshi prices (KXNBAGAME)", "1-minute bid, ask, volume", D["price_rows"], "Fills, costs, CLV, anchor, M6"],
            ["Kalshi markets, settlements", "Market → game; yes / no result",
             f"{D['markets']} / {D['settlements']}", "Settling P&L"],
            ["ESPN games", "Schedule, tip time, score, 3 seasons", D["games"], "Splits, M4 labels"],
            ["ESPN box scores", "Minutes, points, shots per player", D["box_rows"], "M1–M4 features"],
            ["ESPN inactive lists", "Who is out and why (our news)", D["news_rows"], "News trigger; M4 after news"]]
    table(s, rows, MARGIN, TOP + 0.1, [3.1, 3.9, 1.8, CW - 8.8], row_h=0.5, pt=15, header_h=0.45, center_cols=(2,))
    textbox(s, MARGIN, 4.45, CW / 2 - 0.15, 2.45, [
        ("Time splits", {"bold": True, "color": NAVY, "heading": True}),
        "Models train before 1 Feb 2026; agent develops Nov–Jan",
        f"Test 1 Feb – 12 Apr ({D['test_games']} games); walk-forward retrains monthly",
        f"Play-offs ({D['holdout_games']} games): holdout scored once",
    ], pt=16, bullets=True, space=3, slide_no=n, label="splits")
    textbox(s, MARGIN + CW / 2 + 0.15, 4.45, CW / 2 - 0.15, 2.45, [
        ("No look-ahead, real costs", {"bold": True, "color": RED, "heading": True}),
        "Agent sees only what was published by then",
        "Inactive list = news at tip − 30 min (ends 7 May)",
        "Fill at ask + fee; quote ≤ 5 min old; ≤ 10% of volume",
    ], pt=16, bullets=True, space=3, slide_no=n, label="rules")

    # 7 Evidence: costs and information speed ---------------------------------
    n = 7
    s = new_slide(prs, "Evidence 1: costs, and a price that moves before the news", (
        f"First, costs. One hour before tip a $20 order pays half the bid-ask spread plus the Kalshi fee: "
        f"{K['cost_c']} cents per contract on average, {K['cost_pct']} of the price, and {K['cost_cheap_pct']} on "
        f"cheap contracts. Our best agent would have made {K['full_mid']} at the mid but lost {K['full_pnl']} after "
        f"costs; only {K['beat_close']} of its {K['trades']} trades beat or matched the close. Second, speed. On the "
        f"{K['disc_games']} test games where the inactive list moved our win model by at least a point, "
        f"{K['disc_share']} of the price move in the news direction happened before the inactive list, when a "
        f"retail user could act. Against the official NBA injury reports, about two-thirds, {K['it_before']} with "
        f"a confidence interval of {K['it_ci']}, came before the first report listing the player out, which comes "
        f"about {K['it_lead']} hours before the inactive list; {K['it_nolag']} assuming no publication lag. These "
        f"two facts are why the agent passes by default and why its trigger matters."), n)
    hw = (CW - 0.3) / 2
    picture(s, "cost_burden.png", MARGIN, TOP, hw, 2.5)
    picture(s, "price_discovery.png", MARGIN + hw + 0.3, TOP, hw, 2.5)
    textbox(s, MARGIN, TOP + 2.65, hw, 0.5, [("Costs", {"bold": True, "color": ORANGE, "heading": True})],
            pt=20, space=0, slide_no=n, label="costs head")
    textbox(s, MARGIN, TOP + 3.1, hw, 2.6, [
        f"**{K['cost_c']}¢ per contract** = {K['cost_pct']} of price; **{K['cost_cheap_pct']}** under 20¢",
        f"Best agent, Feb–Apr: {K['full_mid']} at mid → **{K['full_pnl']}** after costs",
        f"{K['decisions']} decision points → **{K['trades']} trades**; only **{K['beat_close']}** beat the close",
    ], pt=17, bullets=True, space=6, slide_no=n, label="costs")
    textbox(s, MARGIN + hw + 0.3, TOP + 2.65, hw, 0.5, [("Speed", {"bold": True, "color": BLUE, "heading": True})],
            pt=20, space=0, slide_no=n, label="speed head")
    textbox(s, MARGIN + hw + 0.3, TOP + 3.1, hw, 2.6, [
        f"**{K['disc_share']}** of the move before the inactive list (+{K['disc_before']} vs +{K['disc_after']} "
        f"pts, {K['disc_games']} games)",
        f"**~⅔** ({K['it_before']}, CI {K['it_ci']}) before the first official injury report, ~{K['it_lead']} h "
        f"earlier",
        (f"{K['it_nolag']} with no publication lag; official reports only", {"pt": 14, "color": GREY}),
    ], pt=17, bullets=True, space=6, slide_no=n, label="speed")

    # 8 Evidence: market beats models ---------------------------------------
    n = 8
    s = new_slide(prs, "Evidence 2: the market beats our model", (
        f"Third, expertise. On {K['m4_games']} test games our win model's Brier score one hour before tip is "
        f"{K['m4_brier']}, against {K['mkt_brier']} for the market at the same moment; lower is better. When they "
        f"disagree by more than five points, the market is right in both directions."), n)
    picture(s, "m4_vs_market.png", MARGIN, TOP + 0.05, 7.6, 5.7, align="left")
    textbox(s, 8.3, TOP + 0.2, W - MARGIN - 8.3, 5.6, [
        f"Brier, {K['m4_games']} test games, 1 h before tip: **market {K['mkt_brier']}**, our model {K['m4_brier']}",
        "Disagree by more than 5 pts, and the market is right:",
        ("model higher on home: model 48%, market 35%, actual 34%", {"level": 1}),
        ("model lower on home: model 61%, market 73%, actual 77%", {"level": 1}),
        (f"With the inactive list: model {K['m4_brier_tip']} vs market at tip {K['mkt_brier_tip']}",
         {"pt": 16, "color": GREY}),
    ], pt=19, bullets=True, space=10, slide_no=n)

    # 9 Deep learning: M6 -----------------------------------------------------
    n = 9
    s = new_slide(prs, "Deep learning: four neural parts, graded by the market", (
        f"Deep learning sits in four places. M2 is a GRU over each player's last 20 games giving points and minutes "
        f"quantiles; it beats gradient boosting, pinball {K['m2_gru']} against {K['m2_gbm']}. M4-NN is a neural win "
        f"model on M4's features and split: alone it ties logistic regression, Brier {K['nn_mlp']} against "
        f"{K['cal_raw']}, because 3,400 games is too few for capacity to pay, but plugged into the agent's anchor "
        f"its news shift gives {K['nn_anchor']}, level with the market, not significantly better. The LLM agent is "
        f"a transformer used as a reasoning policy with tools and code guardrails. "
        f"M6 is the fourth. Its label is the market itself: the move of the home price from the "
        f"decision time to tip-off, so every decision time is a training row. A 16-unit GRU reads the last six "
        f"hours of the price in 15-minute steps; it is joined with 18 static features such as the anchor, the "
        f"clock, M4's news shift and who is out, and an MLP outputs the 10th, 50th and 90th percentiles, trained "
        f"with pinball loss. It cannot beat “the price won't move”: {K['m6_mae']} against {K['zero_mae']} cents. "
        f"To check that this is not a tuning accident, we ran a pre-declared grid of {K['m6_cfgs']} configurations, "
        f"GRU sizes, history lengths, an MLP and a CNN, with {K['m6_seeds']} seeds each: every one is worse than "
        f"zero move, all {K['m6_cis']} confidence intervals above zero, and none ever predicts the {K['m6_need']} "
        f"move needed to pay for a trade. Recalibrating the win model does not help either: Brier {K['cal_raw']} "
        f"raw, {K['cal_platt']} Platt, {K['cal_iso']} isotonic, against {K['cal_mkt']} for the market, because its "
        f"miscalibration flips between early and late season."), n)
    rows = [["Neural part", "Design", "Market-graded result"],
            ["M2 player GRU", "GRU over last 20 games → pts / min quantiles",
             f"**Beats GBM:** pinball {K['m2_gru']} vs {K['m2_gbm']}"],
            ["M4-NN win MLP", "M4's 6 features, BCE, early stop, 3 seeds",
             f"Brier {K['nn_mlp']} vs M4 {K['cal_raw']}: tie"],
            ["Anchor + NN shift", "24 h mid + MLP news shift",
             f"Brier **{K['nn_anchor']}** vs market {K['cal_mkt']}: level"],
            ["M6 impact GRU", "6 h prices + 18 features → quantiles",
             f"**Fails:** {K['m6_cfgs']} of {K['m6_cfgs']} configs worse than zero move"],
            ["LLM tool agent", "Transformer as policy; code guardrails", "Slide 11: helps only by trading less"]]
    table(s, rows, MARGIN, TOP + 0.05, [2.0, 3.0, 2.9], row_h=0.62, pt=13, header_h=0.4, label="dl table")
    picture(s, "m6_robustness.png", MARGIN + 8.1, TOP + 0.05, CW - 8.1, 3.6)
    textbox(s, MARGIN, 5.55, CW, 0.7, [
        "**Why small and quantile:** ~3.4k training games, 3.9M price rows but only ~600 training games of them; "
        "GRU-8/16 with early stopping; pinball loss gives a 10–90% band, not a point bet"], pt=15, space=0,
        slide_no=n, label="why")
    textbox(s, MARGIN, 6.28, CW, 0.65, [
        "**Null results are the finding:** in an efficient market, “no move” and the market's own price are "
        "hard baselines; deep learning helps only next to the anchor"], pt=15, space=0, slide_no=n, label="null")

    # 10 Money over time, tested with CIs ------------------------------------
    n = 10
    s = new_slide(prs, "Evidence 3: every line drifts down; never trading wins", (
        f"Money over time, after fees; the dashed line at zero is never trading. The red raw model falls steadily, "
        f"{K['cum_raw_pnl']} in the test period and {K['wf_raw_pnl']} over the season: that slope is costs. The "
        f"agent without learning falls less; the full agent learned rules that mostly say do not trade, so it "
        f"flattens near zero. Then the honest test: walk-forward over the whole season with the win model retrained "
        f"monthly and day-clustered bootstrap intervals. Mean closing-line value is below zero for every setup. "
        f"The full agent's {K['wf_full_pnl']} is indistinguishable from never trading, p {K['wf_full_p']}. The "
        f"market anchor is real, {K['wf_anchor_gain']}, p {K['wf_anchor_p']}; learning only means trading less. On "
        f"the play-off holdout the full agent made no trades."), n)
    picture(s, "cumulative_pnl.png", MARGIN, TOP, CW, 3.5)
    rows = [["Walk-forward, Nov – 12 Apr", "Trades", "P&L after fees [95% CI]"],
            ["**Never trade**", "0", "**$0**"],
            ["Full agent: anchor + learning", K["wf_full_n"], f"{K['wf_full_pnl']} [{K['wf_full_ci']}]"],
            ["Anchor, no learning", K["wf_nolearn_n"], f"{K['wf_nolearn_pnl']} [{K['wf_nolearn_ci']}]"],
            ["Raw model", K["wf_raw_n"], f"{K['wf_raw_pnl']} [{K['wf_raw_ci']}]"]]
    table(s, rows, MARGIN, 4.8, [3.2, 0.95, 2.85], row_h=0.38, pt=14, emphasis_row=1, header_h=0.4,
          center_cols=(1, 2))
    textbox(s, MARGIN + 7.2, 4.75, CW - 7.2, 2.2, [
        "Mean CLV < 0 for every setup; **every CI below zero**",
        f"Anchor is real: {K['wf_anchor_gain']} (p = {K['wf_anchor_p']}); learning = trading less "
        f"(p = {K['wf_learn_p']})",
        f"Play-off holdout: full agent 0 trades",
    ], pt=16, bullets=True, space=4, slide_no=n, label="wf bullets")

    # 11 Agentic upgrades ------------------------------------------------------
    n = 11
    orch = (FIG / "orchestrator.png").exists()
    llm_tool = (f"The tool agent with the sceptic scored {K['llm_tool']}." if K["llm_tool"] else
                "The tool agent and the sceptic are pending: the Gemini quota ran out and they are being re-run "
                "on DeepSeek.")
    s = new_slide(prs, "Agentic upgrades: the market's answer is “trade less”", (
        f"We then made the agent more agentic and graded each upgrade the same way, pre-registered before "
        f"scoring. Gate audit: the split gate, which judges a rule on held-out days by CLV dollars, kept "
        f"{K['ga_split']} proposed rules where the legacy gate kept {K['ga_legacy']}. Trades fell from "
        f"{K['ga_trades'].replace(' → ', ' to ')} and CLV dollars improved by {K['ga_clv_gain']}, but mean CLV per "
        f"trade did not improve and it still does not beat never trading. The placebo test says why: random rules "
        f"pass the split gate {K['pl_rate']} of the time, the same as the reviewer's picks, so learning works by "
        f"trading less, not by finding better rules. The kill switch is enforced but never tripped at $20 stakes; "
        f"property tests show it blocks orders after a trip. Kelly sizing was not run. The Coach on simulated "
        f"users with full compliance raises CLV dollars for every persona: overtrader {K['co_over']}, long-shot "
        f"lover {K['co_long']}, cautious {K['co_caut']}, price chaser {K['co_chase']}, whose interval includes "
        f"zero, and cuts trades by "
        f"{K['co_cut']}. But it flags {K['co_flag']} of trades, so again the gain is trading less. The plain LLM "
        f"made {K['llm_plain_n']} trades in {K['llm_points']} decisions, CLV {K['llm_plain_clv']}, the same as "
        f"never trading; the deterministic agent made {K['llm_det_n']} trades, CLV {K['llm_det_clv']}. {llm_tool} "
        f"The orchestrator is built, shown on the right. The learned policy's selection step chose never trade, "
        f"and the test was not scored. The rule language and memory are built and tested but not evaluated. "
        f"The message: {TRADE_LESS}"), n)
    tw = 8.9 if orch else CW
    rows = [["Upgrade", "What it is", "Result"]]
    muted = set()
    for i, (name, what, _, res) in enumerate(K["agentic"], 1):
        rows.append([f"**{name}**", what, res or PENDING])
        if not res or res.startswith("Not run"):
            muted.add((i, 2))
    if orch:
        table(s, rows, MARGIN, TOP + 0.05, [1.95, 2.55, tw - 4.5], row_h=0.47, pt=11, header_h=0.34,
              muted=muted, label="agentic table")
        picture(s, "orchestrator.png", MARGIN + tw + 0.15, TOP + 0.05, CW - tw - 0.15, 4.2)
        textbox(s, MARGIN + tw + 0.15, TOP + 4.35, CW - tw - 0.15, 0.55, [
            "Pre-registered before scoring; same replay, costs and day-clustered bootstrap as above"],
            pt=11, color=GREY, space=0, slide_no=n, label="prereg")
    else:
        table(s, rows, MARGIN, TOP + 0.05, [2.4, 4.4, tw - 6.8], row_h=0.47, pt=11, header_h=0.34,
              muted=muted, label="agentic table")
    callout(s, MARGIN, 6.36, CW, 0.56, TRADE_LESS, pt=14, n=n)

    # 12 Product: protect, teach, practise ---------------------------------
    n = 12
    s = new_slide(prs, "Our product: protect, teach, practise", (
        f"So the product educates and protects instead of encouraging bets. Protect: the agent passes by default, "
        f"about {K['pass_share']} of decision points, and says why; risk limits and banned words are code the LLM "
        f"cannot override, and all {K['policy']} planted violations were blocked. Teach: the Coach explains a game "
        f"with only what was public and shows the break-even after spread and fee. Practise: the League gives "
        f"everyone the same {K['lg_slate']} real past decisions and {K['lg_bankroll']} of play money, and ranks on "
        f"closing-line value, not profit. In our example league the raw-model bot is up {K['lg_bot_pnl']} with "
        f"negative CLV, and the badge says costs, not skill."), n)
    cards = [("Protect: the agent", NAVY, [
                 f"**Passes by default:** {K['pass_share']} of decision points",
                 "Says why there is no order",
                 "Code limits: $50 / order, $100 / game, $300 / day",
                 f"Bans “lock”, “guaranteed”; **{K['policy']}** planted violations blocked"]),
             ("Teach: the Coach", ORANGE, [
                 "Only what was public at that moment",
                 "Break-even after spread + fee",
                 "Lessons: priced in, long shots, closing line",
                 "Paper call, then the close and the agent's choice"]),
             ("Practise: the League", GREEN, [
                 f"Same {K['lg_slate']} real decisions, {K['lg_bankroll']} play money",
                 "Fills at ask + fee, like the agent",
                 f"**Ranked by CLV**, not profit (≥ {K['lg_min_trades']} trades)",
                 f"Raw-model bot {K['lg_bot_pnl']} with CLV {K['lg_bot_clv']}: “costs”"])]
    for i, (head, col, lines) in enumerate(cards):
        card(s, MARGIN + i * (cw3 + 0.25), TOP + 0.15, cw3, 3.6, head, lines, col, pt=16, n=n)
    picture(s, "league_reveal.png", MARGIN, TOP + 3.9, CW, 0.95)
    callout(s, MARGIN, 6.2, CW, 0.65,
            "Educational, paper money only · only information public at that moment · no promise words", pt=18, n=n)

    # 13 Team contributions -------------------------------------------------------
    n = 13
    s = new_slide(prs, "Why an agentic framework, and who built which part", (
        f"Why an agent and not one classifier? Each box of the slide 4 diagram answers a feature of the problem. "
        f"The problem is event-driven: news arrives at random times and {K['disc_share']} of the move comes before "
        f"the inactive list, about two-thirds before the first official report, so the hard decision is when to "
        f"act; that is the trigger. What to look at is the as-of tools, which enforce no look-ahead. Whether to "
        f"trade is the edge test and the LLM sceptic, and the code checks, risk caps and kill switch make sure "
        f"the LLM can never override a limit, which matters for gambling harm. The learn loop is graded by the "
        f"market through the gate, and the placebo shows honestly that it works by trading less. The Coach and "
        f"League are the people in the loop. The counter-argument, a single classifier would do, is answered by "
        f"our own data: the model alone loses {K['wf_raw_pnl']} over {K['wf_raw_n']} trades; the decision process "
        f"around it, the diagram's anchor, edge threshold and pass, adds {K['wf_anchor_gain']}, p "
        f"{K['wf_anchor_p']}. "
        "The work is split across the team. The pregame news loop polls several news sources before a game and "
        "turns them into fair odds, pull request 7. The player-feature experiment tested isolated player features "
        "and probability calibration, pull request 8. In-play news and odds updates are pull request 9. Live, "
        "read-only Polymarket prices are pull requests 11 and 12. The replay, the agent, the models M4 and M6, "
        "the evaluation, the Coach and the League are on our main branch. Each part had to face the same market "
        "grade."), n)
    textbox(s, MARGIN, TOP + 0.05, 6.3, 4.6, [
        ("Why agentic: each box of slide 4 answers the problem", {"bold": True, "color": NAVY, "heading": True}),
        f"**When to act = trigger:** news is asynchronous; {K['disc_share']} of the move before the inactive "
        f"list, ~⅔ before the first report",
        "**What to look at = as-of tools:** prices, news, rotations, M4 / M6; no look-ahead",
        "**Whether to trade = edge + sceptic:** code checks, risk caps, kill switch; LLM can't override",
        "**Learn loop:** CLV → reviewer → held-out gate → rules; placebo-audited",
        "**People:** Coach and League teach with the same analysis",
    ], pt=15, bullets=True, space=4, slide_no=n, label="why agentic")
    rows = [["Work", "PR", "Who"]] + [[r[0], r[1], r[2]] for r in K["team"]]
    table(s, rows, MARGIN + 6.5, TOP + 0.05, [3.0, 0.95, CW - 6.5 - 3.95], row_h=0.62, pt=12, header_h=0.4,
          center_cols=(1,), label="team table")
    callout(s, MARGIN, 5.75, CW, 1.05,
            f"“A single classifier would do”? The model alone loses {K['wf_raw_pnl']} over {K['wf_raw_n']} trades; "
            f"the decision process around it adds {K['wf_anchor_gain']} [+$854, +$3,984], p = {K['wf_anchor_p']}",
            pt=17, n=n)

    # 14 Demo -----------------------------------------------------------------
    n = 14
    s = new_slide(prs, "Live demo: streamlit run app.py", (
        "Now the demo, focused on protection. On the replayed night, watch the agent see the news, compare its "
        "estimate with the bid and ask, and usually decline with a reason. In the Coach, I'll make a paper call "
        "and compare it with the close. In the League tab I'll make one call on the shared slate, watch the "
        "closing line and result being revealed, and show the leaderboard ranked by closing-line value next to "
        "the house bots. Then a quick look at the learned rules and planted violations being blocked. If anything "
        "fails, we have a backup recording."), n)
    rows = [["Tab", "What to watch"],
            ["Replayed night", "News → estimate vs bid-ask → usually no order, with the reason"],
            ["Coach", "Break-even after spread + fee; paper call vs the close and the agent"],
            ["League", "One call on the shared slate → close revealed → ranked by CLV next to the house bots"],
            ["Learning", "Gate keeps protective rules (skip long shots, skip chasing)"],
            ["Safety", f"Planted violations blocked ({K['policy']})"]]
    table(s, rows, MARGIN, TOP + 0.15, [2.6, CW - 2.6], row_h=0.7, pt=18)
    callout(s, MARGIN, 6.2, CW, 0.6, "Backup: recorded run of the same demo", pt=18, n=n)

    # 15 Recommendations, scope, Q&A ---------------------------------------------
    n = 15
    s = new_slide(prs, "Recommendations, honest scope and Q&A", (
        "Our recommendations follow from the evidence. Before any order, show the all-in cost and the break-even "
        "probability. Warn by default when news is likely already priced in. Put education and play-money practice "
        "in front of the first trade, and treat betting harm as a public-health issue, as the Lancet Commission "
        "argues. Honest scope: unfair here means structural disadvantages, costs, speed and expertise, not "
        "manipulation or fraud, and we make no accusation against Kalshi. We designed the market anchor and the "
        "reviewer after seeing the first February-to-April results, so the walk-forward season and the play-off "
        "holdout are our headline tests. Thank you, we are happy to take questions."), n)
    card(s, MARGIN, TOP + 0.15, 6.4, 3.9, "For products and policy", [
        "Show the all-in cost and break-even before every order",
        "Default warning: “this news is likely priced in”",
        "Education and play-money practice before the first trade",
        "Treat betting harm as a public-health issue (Lancet)",
    ], NAVY, pt=17, n=n)
    card(s, MARGIN + 6.7, TOP + 0.15, CW - 6.7, 3.9, "Honest scope", [
        "“Unfair” = costs, speed, expertise; **not** fraud; no accusation against Kalshi",
        "Anchor and reviewer designed after seeing Feb–Apr; walk-forward and play-offs are the honest tests",
        "One season, inactive lists as our news; paper money",
    ], RED, pt=16, n=n)
    textbox(s, MARGIN, 5.45, CW, 1.3, [("Thank you. Questions?", {"bold": True})], pt=40, color=NAVY,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0, slide_no=n)

    return prs


def check(prs):
    """Fail loudly if a picture leaves the slide or a text box is estimated to overflow."""
    problems = []
    sw, sh = prs.slide_width, prs.slide_height
    for i, slide in enumerate(prs.slides, 1):
        for shp in slide.shapes:
            if shp.left < 0 or shp.top < 0 or shp.left + shp.width > sw + 10 or shp.top + shp.height > sh + 10:
                problems.append(f"slide {i}: {shp.shape_type} '{shp.name}' leaves the slide")
        if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
            problems.append(f"slide {i}: no speaker notes")
    for slide_no, label, used, h in _fit_report:
        if used > h + 0.02:
            problems.append(f"slide {slide_no}: text '{label}' needs ~{used:.2f} in, box is {h:.2f} in")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the fit report for every text box")
    args = ap.parse_args()
    sync_figures()
    prs = build()
    prs.save(OUT)
    if args.check:
        for row in _fit_report:
            print("slide %2d  %-42s  %.2f / %.2f in" % row)
    problems = check(prs)
    filled, pending, waiting = agentic_status()
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(prs.slides)} slides)")
    print(f"agentic results filled: {', '.join(filled) or 'none'}; pending: {', '.join(pending) or 'none'}")
    if waiting:
        print(f"NOTE: result files exist for pending rows; check them and fill K: {', '.join(waiting)}")
    for p in problems:
        print("WARNING:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
