"""Build the 10-minute presentation deck from the figures in presentation/figures/.

    python presentation/build_deck.py            # writes presentation/nba_agent_deck.pptx
    python presentation/build_deck.py --check    # also prints a fit report for every text box

Narrative: sports prediction markets are structurally unfair to consumers; our agent is the
evidence and the product protects and educates. Every number on the slides lives in K below,
with its source, so it can be updated in one place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "nba_agent_deck.pptx"

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

FOOTER = "DASC7606C Group 12 · Betting markets are tilted against consumers"

# Key numbers. Sources: README "Results" (R); evaluation/results/consumer_fairness.md (F);
# m4_vs_market.csv (M); trade funnel / fairness_run_costs.csv (T); walkforward_summary.md,
# significance.md, holdout.md (W); m6_heldout.csv, m6_ablation.md, README M6 section (S).
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
    "ls_n": "66", "ls_implied": "7.4%", "ls_z": "−1.8",                        # F4, 0–10¢ bucket
    "ls_roi": "−13.4%", "fav_roi": "−2.2%",                                    # F5
    "policy": "9 / 9",                                                         # R, policy_tests.csv
    # walk-forward, season Nov – 12 Apr, M4 retrained monthly, day-clustered bootstrap (W)
    "wf_full_n": "157", "wf_full_pnl": "−$25", "wf_full_ci": "−$569 to +$564", "wf_full_p": "0.53",
    "wf_nolearn_n": "276", "wf_nolearn_pnl": "−$407", "wf_nolearn_ci": "−$1,324 to +$553",
    "wf_raw_n": "711", "wf_raw_pnl": "−$2,475", "wf_raw_ci": "−$3,995 to −$845",
    "wf_anchor_gain": "+$2,450", "wf_anchor_p": "0.002", "wf_learn_p": "0.13",
    "ho_games": "87", "ho_nolearn": "6 trades, −$74", "ho_raw": "+$450",       # W, play-off holdout
    # M6 market-impact model (S)
    "m6_mae": "0.867", "zero_mae": "0.833", "m6_coef": "0.025", "m6_coef_pct": "2.5%",
    "m6_forced_n": "501", "m6_forced_pnl": "−$652",
    # cumulative P&L after fees by settlement date (evaluation/results/cumulative_pnl.csv; ends match
    # ablations.md and walkforward_summary.md)
    "cum_nolearn_n": "129", "cum_nolearn_pnl": "−$247", "cum_raw_n": "366", "cum_raw_pnl": "−$2,087",
    "cum_full_low": "−$131",
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
          header_h=None, center_cols=()):
    header_h = header_h or row_h
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
            for k, part in enumerate(parts):
                if part:
                    _run(p, part, pt, WHITE if i == 0 else TEXT, bold=(i == 0 or k % 2 == 1))
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


# ---------------------------------------------------------------- slides

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    CW = W - 2 * MARGIN

    # 1 Title ---------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 4.55, W, 0.08, ORANGE)
    textbox(s, 0.9, 1.1, W - 1.8, 2.0,
            [("Betting markets are tilted against consumers", {"bold": True})],
            pt=44, color=WHITE, slide_no=1, label="title")
    textbox(s, 0.9, 3.1, W - 1.8, 1.3,
            ["An AI agent that tests it on a full NBA season, and a product that teaches it"],
            pt=24, color=RGBColor(0xCB, 0xD5, 0xE1), slide_no=1, label="subtitle")
    textbox(s, 0.9, 4.85, W - 1.8, 1.9,
            ["Market-Graded Learning for NBA Game-Impact Intelligence",
             "DASC7606C (COMP7606) · Group 12 · Track 2: Agentic Framework Design",
             "Presenter: Ngan Tsz Sui · Friday 9 October 2026"],
            pt=20, color=WHITE, slide_no=1, label="meta")
    s.notes_slide.notes_text_frame.text = (
        "Good morning, we are Group 12. Our project starts from a consumer problem: sports betting and prediction "
        "markets are now everywhere, people want to understand them, the markets are structurally tilted against "
        "them, and betting harms well-being. We built an AI agent to test how tilted the game is, and turned what "
        "we learned into a product that educates and protects.")

    # 2 The problem -----------------------------------------------------------
    n = 2
    s = new_slide(prs, "The problem: three things consumers face", (
        f"Three parts. First, consumers want to learn: since a 2018 US Supreme Court ruling, {EXT['states']} US "
        f"states have legalised sports betting, and the Lancet Public Health Commission estimates {EXT['adults_pct']} "
        f"of adults worldwide gambled in the past year, yet few users understand how prices, fees and spreads work. "
        f"Second, the markets are structurally unfair, which is what we measured: about {K['cost_pct']} cost per trade, "
        f"{K['disc_share']} of the price move before the public news, the market beats our model, and never trading "
        f"wins. Third, external research shows betting harms well-being: the Commission estimates "
        f"{EXT['disorder']} adults have gambling disorder or problem gambling; Hollenbeck, Larsen and Proserpio find "
        f"legal online betting lowers credit scores by {EXT['credit_online']} points and raises bankruptcies by about "
        f"{EXT['bankrupt']}; Baker and co-authors find each dollar of online betting cuts investment by about "
        f"{EXT['invest']}. References: Wardle H. et al., The Lancet Public Health Commission on gambling, Lancet "
        f"Public Health 2024; 9: e950–94. Hollenbeck B., Larsen P., Proserpio D., The Financial Consequences of "
        f"Legalized Sports Gambling, SSRN 4903302, 2024, revised 2025. Baker S.R., Balthrop J., Johnson M.J., "
        f"Kotter J.D., Pisciotta K., Gambling Away Stability: Sports Betting's Impact on Vulnerable Households, "
        f"NBER Working Paper 33108, 2024."), n)
    cw3 = (CW - 0.5) / 3
    pillars = [
        ("1  People want to learn", BLUE, [
            f"Mainstream: legal in {EXT['states']} US states since 2018 ¹",
            f"{EXT['adults_pct']} of adults worldwide gambled last year ²",
            "Few understand prices, fees and spreads",
        ]),
        ("2  Markets are tilted", ORANGE, [
            f"~{K['cost_pct']} cost on every trade",
            f"{K['disc_share']} of the price move comes before the public news",
            "The market beats our model; never trading wins",
        ]),
        ("3  Betting causes harm", RED, [
            f"~{EXT['disorder']} adults with gambling disorder or problem gambling ²",
            f"Online betting: credit scores −{EXT['credit_online']} pts, bankruptcies +~{EXT['bankrupt']} ¹",
            f"$1 of online betting cuts investment ~{EXT['invest']} ³",
        ]),
    ]
    for i, (head, col, lines) in enumerate(pillars):
        card(s, MARGIN + i * (cw3 + 0.25), TOP + 0.15, cw3, 3.55, head, lines, col, pt=18, n=n)
    textbox(s, MARGIN + cw3 + 0.25, TOP + 3.75, cw3, 0.45, [("Our measurements", {"italic": True})],
            pt=14, color=ORANGE, align=PP_ALIGN.CENTER, slide_no=n, label="ours")
    textbox(s, MARGIN + 2 * (cw3 + 0.25), TOP + 3.75, cw3, 0.45, [("External research", {"italic": True})],
            pt=14, color=RED, align=PP_ALIGN.CENTER, slide_no=n, label="external")
    callout(s, MARGIN, TOP + 4.35, CW, 0.7,
            "Consumers need education and protection, not betting tips", pt=22, n=n)
    textbox(s, MARGIN, 6.45, CW, 0.5, [
        "¹ Hollenbeck, Larsen & Proserpio 2024 (SSRN)   ² Lancet Public Health Commission on gambling 2024   "
        "³ Baker et al. 2024 (NBER WP 33108)"], pt=13, color=GREY, slide_no=n, label="refs")

    # 3 Research question --------------------------------------------------------
    n = 3
    s = new_slide(prs, "Research question", (
        "So our research question: can a consumer, even one armed with an AI agent and public news, beat the "
        "market? We built the strongest consumer we could: public news, trained models including a neural net "
        "trained on the market's own moves, an LLM-capable agent loop, risk limits and rules learned through a "
        "backtest gate, graded on a full season of real Kalshi prices. The answer, tested with confidence "
        "intervals, is no: never trading wins. So the product's job is to educate and protect."), n)
    callout(s, MARGIN, TOP + 0.15, CW, 1.15,
            "Can a consumer, even one armed with an AI agent and public news, beat the market?", pt=28,
            fill=RGBColor(0xDB, 0xEA, 0xFE), color=NAVY, n=n)
    textbox(s, MARGIN, TOP + 1.55, 6.9, 4.2, [
        ("We built the strongest consumer we could:", {"bold": True, "color": NAVY, "heading": True}),
        "Public news: ESPN inactive lists and box scores, 3 seasons",
        "Trained models, incl. a neural net trained on the market's own moves",
        "LLM-capable LangGraph agent; code risk limits; gated rule learning",
        f"Graded on {K['games']} Kalshi games ({K['price_rows']} price rows), 2025-26",
    ], pt=19, bullets=True, space=8, slide_no=n)
    x0 = 7.75
    rect(s, x0, TOP + 1.6, W - MARGIN - x0, 3.0, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.05
    textbox(s, x0 + 0.3, TOP + 1.75, W - MARGIN - x0 - 0.6, 2.75, [
        ("Answer: no", {"bold": True, "color": RGBColor(0xFD, 0xBA, 0x74)}),
        "Tested with confidence intervals, never trading wins.",
        "So the product educates and protects instead.",
    ], pt=22, color=WHITE, space=14, slide_no=n, label="answer")

    # 4 What we built -----------------------------------------------------
    n = 4
    s = new_slide(prs, "What we built: a market-graded agent", (
        "Briefly, the system. Public data on the left; models trained once; a replay that walks the season day "
        "by day and only shows the agent what was public at that moment; then the agent and its outputs. "
        "Because the replay uses recorded prices, every trade pays the real ask and the real Kalshi fee, which "
        "is what lets us measure what a consumer would actually face."), n)
    picture(s, "product_workflow.png", MARGIN, TOP, CW, 4.9)
    textbox(s, MARGIN, 6.15, CW, 0.8, [
        "Every simulated order pays the recorded ask plus the Kalshi fee, so we measure what a consumer would face"],
        pt=18, align=PP_ALIGN.CENTER, slide_no=n)

    # 5 Evidence: costs and selectivity --------------------------------------
    n = 5
    s = new_slide(prs, "Evidence 1: costs set a hurdle before you start", (
        f"First, costs. One hour before tip a $20 order pays half the bid-ask spread plus the Kalshi fee. On "
        f"average that is {K['cost_c']} cents per contract, {K['cost_pct']} of the price, and {K['cost_cheap_pct']} "
        f"on cheap contracts. You must beat the price by that much just to break even. On the right, our best agent "
        f"would have made {K['full_mid']} at the mid but lost {K['full_pnl']} after costs. Even being very selective, "
        f"{K['trades']} trades out of {K['decisions']} decision points, only {K['beat_close']} beat or matched the "
        f"closing price."), n)
    picture(s, "cost_burden.png", MARGIN, TOP, CW, 4.15)
    textbox(s, MARGIN, 5.45, CW / 2 - 0.1, 1.5, [
        f"**{K['cost_c']}¢ per contract** = {K['cost_pct']} of price; **{K['cost_cheap_pct']}** under 20¢",
        f"Best agent, Feb–Apr: {K['full_mid']} at mid → **{K['full_pnl']}** after costs",
    ], pt=18, bullets=True, space=4, slide_no=n)
    textbox(s, MARGIN + CW / 2 + 0.1, 5.45, CW / 2 - 0.1, 1.5, [
        f"{K['decisions']} decision points → {K['gap4']} with gap > 4 pts → **{K['trades']} trades**",
        f"Only **{K['beat_close']}** beat or matched the close",
    ], pt=18, bullets=True, space=4, slide_no=n)

    # 6 Evidence: information speed ----------------------------------------
    n = 6
    s = new_slide(prs, "Evidence 2: the price moves before the news reaches you", (
        f"Second, speed. On the {K['disc_games']} test games where the inactive list changed our win model by at "
        f"least a point, the price had already moved {K['disc_before']} points in the news direction before the "
        f"list was public, and only {K['disc_after']} after. About {K['disc_share']} of the move happened before a "
        f"retail user reading the list could act."), n)
    picture(s, "price_discovery.png", MARGIN, TOP, CW, 4.55)
    textbox(s, MARGIN, 5.85, CW, 1.1, [
        f"**{K['disc_share']}** of the news-direction move was already in the price when the list was public "
        f"(+{K['disc_before']} vs +{K['disc_after']} pts, {K['disc_games']} test games)",
        "List stamped tip − 30 min: our proxy for when a retail user sees it",
    ], pt=18, bullets=True, space=4, slide_no=n)

    # 7 Evidence: market beats models ---------------------------------------
    n = 7
    s = new_slide(prs, "Evidence 3: the market beats our model", (
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

    # 8 Evidence: M6 ---------------------------------------------------------
    n = 8
    s = new_slide(prs, "Evidence 4: even a market-trained neural net finds nothing", (
        f"Fourth, we trained M6, a neural network whose only job is to predict how the price moves between our "
        f"decision time and tip-off, using the market's own past reactions. It cannot beat the simplest guess, "
        f"that the price will not move: {K['m6_mae']} cents average error against {K['zero_mae']}, and R squared "
        f"about zero on both test and holdout. After the inactive list the price moves only about "
        f"{K['m6_coef_pct']} of what our win model implies. The M6 agent therefore makes no trades; forced to "
        f"trade, it loses {K['m6_forced_pnl']}. The price already contains the public information; what is left "
        f"for the consumer is the cost."), n)
    picture(s, "m6_vs_baselines.png", MARGIN, TOP, CW, 3.7)
    textbox(s, MARGIN, 4.95, CW, 1.42, [
        f"Average error on {K['decisions']} test decision times: **M6 {K['m6_mae']}¢** vs “price won't move” "
        f"**{K['zero_mae']}¢**; R² ≈ 0 on test and holdout",
        f"After the list, the price moves ~{K['m6_coef_pct']} of the model's implied shift (coefficient {K['m6_coef']}); "
        f"M6 agent: 0 trades. Forced: {K['m6_forced_n']} trades, {K['m6_forced_pnl']}",
    ], pt=18, bullets=True, space=4, slide_no=n)
    callout(s, MARGIN, 6.42, CW, 0.5,
            "The price already contains the public information; the consumer is left paying costs", pt=18, n=n)

    # 9 Evidence: money over time -------------------------------------------
    n = 9
    s = new_slide(prs, "Evidence 5: money over time, every line drifts down", (
        f"Here is the money over time, after fees, one line per strategy; the dashed black line at zero is never "
        f"trading. Left is the February-to-April test period, right the walk-forward season. Read it like this. "
        f"The red raw model, which trades whenever its estimate differs from the price, falls steadily: "
        f"{K['cum_raw_pnl']} on {K['cum_raw_n']} trades in the test period, {K['wf_raw_pnl']} over the season. "
        f"That steady slope is costs: every trade pays the spread and the fee, and there is no edge to pay them "
        f"back. Orange, the agent without learning, trades less and falls less, {K['cum_nolearn_pnl']}. Blue, "
        f"the full agent, learned rules that mostly say do not trade, so its line flattens near zero, "
        f"{K['full_pnl']}, after dipping to {K['cum_full_low']}. The M6 agent made no trades, so it sits on zero. "
        f"Learning does not find profit; it flattens the line by trading less. The dotted lines are closing-line "
        f"value: below zero for everyone, so the agent paid worse than the closing price. No strategy ends "
        f"above never trading; the next slide tests whether the gaps are more than luck."), n)
    picture(s, "cumulative_pnl.png", MARGIN, TOP, CW, 4.6)
    textbox(s, MARGIN, 5.9, CW, 1.05, [
        f"Raw model falls steadily ({K['cum_raw_pnl']} test, {K['wf_raw_pnl']} season): the slope is costs",
        f"Learning flattens the line by trading less: full agent {K['full_pnl']} test, {K['wf_full_pnl']} season; "
        f"M6: 0 trades",
    ], pt=18, bullets=True, space=4, slide_no=n)

    # 10 Evidence: walk-forward ----------------------------------------------
    n = 10
    s = new_slide(prs, "Evidence 6: tested with CIs, never trading wins", (
        f"Finally, the honest test. We re-ran the whole season walk-forward, retraining the win model every month, "
        f"with day-clustered bootstrap confidence intervals. Mean closing-line value is below zero for every setup, "
        f"and every interval lies entirely below zero. The full agent's {K['wf_full_pnl']} is indistinguishable from "
        f"never trading, p {K['wf_full_p']}; the raw model loses {K['wf_raw_pnl']}. The market anchor is a real "
        f"improvement, {K['wf_anchor_gain']} with p {K['wf_anchor_p']}, but learning only means trading less. On the "
        f"play-off holdout the full agent made no trades at all."), n)
    picture(s, "walkforward.png", MARGIN, TOP + 0.05, 6.5, 3.85, align="left")
    rows = [["Nov – 12 Apr", "Trades", "P&L [95% CI]"],
            ["**Never trade**", "0", "**$0**"],
            ["Full agent", K["wf_full_n"], f"{K['wf_full_pnl']} [{K['wf_full_ci']}]"],
            ["No learning", K["wf_nolearn_n"], f"{K['wf_nolearn_pnl']} [{K['wf_nolearn_ci']}]"],
            ["Raw model", K["wf_raw_n"], f"{K['wf_raw_pnl']} [{K['wf_raw_ci']}]"]]
    table(s, rows, 7.15, TOP + 0.1, [2.05, 0.95, 2.68], row_h=0.66, pt=16, emphasis_row=1,
          header_h=0.5, center_cols=(1, 2))
    textbox(s, 7.15, TOP + 3.3, W - MARGIN - 7.15, 0.7, ["Day-clustered bootstrap; M4 retrained monthly. Chart totals include play-offs, the table does not"],
            pt=14, color=GREY, slide_no=n, label="wf caption")
    textbox(s, MARGIN, 5.1, CW, 1.85, [
        "Mean CLV < 0 for every setup; **every 95% CI entirely below zero**",
        f"Market anchor is real: {K['wf_anchor_gain']} vs raw model (p = {K['wf_anchor_p']}); learning = trading "
        f"less (p = {K['wf_learn_p']})",
        f"Play-off holdout ({K['ho_games']} games): full agent 0 trades; no learning {K['ho_nolearn']}; raw model "
        f"{K['ho_raw']} with negative CLV (luck)",
    ], pt=18, bullets=True, space=4, slide_no=n)

    # 11 Product as consumer protection -------------------------------------
    n = 11
    s = new_slide(prs, "Our product: educate and protect the consumer", (
        f"So, back to the problem: the product educates and protects instead of encouraging bets. The agent passes by default: it declined about "
        f"{K['pass_share']} of decision points, and on the play-off holdout it did not trade at all. Retail briefs "
        f"show the bid and ask next to our estimate, and when there is no order they say why. The Coach shows the "
        f"full break-even after spread and fee. Risk limits and banned words are code the LLM cannot override; all "
        f"{K['policy']} planted violations were blocked. Even the learned rules ended up protective."), n)
    picture(s, "agent_workflow.png", MARGIN, TOP, 7.4, 5.75, align="left")
    textbox(s, 8.1, TOP + 0.15, W - MARGIN - 8.1, 5.6, [
        f"**Pass by default:** {K['pass_share']} of decision points → no trade",
        "**No-order reason:** “gap after fees too small or the price already moved”",
        "**Costs shown:** bid-ask in briefs; Coach shows break-even after spread + fee",
        "**Code limits:** $50 / order, $100 / game, $300 / day; bans “lock”, “guaranteed”, “risk-free”",
        f"**{K['policy']}** planted violations blocked",
        "**Learned rules:** skip ≤ 35¢; skip after a 2¢ move against",
    ], pt=18, bullets=True, space=8, slide_no=n)

    # 12 Coach ---------------------------------------------------------------
    n = 12
    s = new_slide(prs, "Coach: teach why most bets don't clear costs", (
        "The Coach turns this evidence into lessons. At a chosen moment it explains the game using only what was "
        "public and shows the break-even after spread and fee. Lesson cards appear when relevant: price is a "
        "probability, costs move your break-even, news may be priced in, long shots look cheap. Users make a "
        "paper call and see the close and what the agent did. Paper money only."), n)
    steps = [("1  What's going on", ["News so far; model before / after the news",
                                     "Market now vs 24 h ago; break-even after spread + fee"], BLUE),
             ("2  Lessons, when relevant", ["Price = probability · costs move break-even",
                                            "Priced in / chasing · long shots · closing line · passing"], ORANGE),
             ("3  Your paper call", ["Back a team or pass",
                                     "Then: close, CLV, result, and what the agent did"], GREEN),
             ("4  Habit feedback", ["Paying above the close, long shots, chasing",
                                    "Small samples are mostly luck"], NAVY)]
    cw2 = (CW - 0.3) / 2
    for i, (head, lines, col) in enumerate(steps):
        x = MARGIN + (i % 2) * (cw2 + 0.3)
        y = TOP + 0.15 + (i // 2) * 2.45
        card(s, x, y, cw2, 2.25, head, lines, col, pt=18, n=n)
    callout(s, MARGIN, 6.3, CW, 0.55,
            "Educational, paper money only · only information public at that moment · no promise words",
            pt=18, n=n)

    # 13 Honest scope ---------------------------------------------------------
    n = 13
    s = new_slide(prs, "Honest scope: what “unfair” means here", (
        "To be precise: unfair here means structural disadvantages, namely costs, speed and expertise. It does not "
        "mean manipulation or fraud, and we make no accusation against Kalshi. One disclosure: we designed the "
        "market anchor and the reviewer after seeing the first February-to-April results, so that window is not a "
        "clean holdout. That is why the walk-forward season and the untouched play-off holdout are our headline "
        "tests. Other caveats: one season, game-winner markets only, a proxy news timestamp, and long-shot bias "
        "was not clearly supported in our data. The well-being numbers on the problem slide are published "
        "research, not our measurement."), n)
    card(s, MARGIN, TOP + 0.15, 5.6, 5.55, "What we claim", [
        "Structural disadvantages: costs, speed, expertise",
        "**Not** manipulation or fraud",
        "**No** accusation against Kalshi or any venue",
        "Paper money; public data only",
    ], NAVY, pt=18, n=n)
    card(s, MARGIN + 5.9, TOP + 0.15, CW - 5.9, 5.55, "Caveats and disclosure", [
        "**Disclosure:** anchor and reviewer designed after seeing Feb–Apr results, so that window is not a clean "
        "holdout; walk-forward and play-off holdout are the honest checks",
        f"One season ({K['games']} games), game-winner markets only",
        "News stamped tip − 30 min (proxy; real reports come earlier)",
        "Costs for a $20 order at tip − 60 min; long-shot bias not clearly supported",
        "Well-being figures are published research, not ours",
    ], RED, pt=18, n=n)

    # 14 Demo -----------------------------------------------------------------
    n = 14
    s = new_slide(prs, "Live demo: streamlit run app.py", (
        "Now the demo, focused on protection. On the replayed night, watch the agent see the news, compare its "
        "estimate with the bid and ask, and usually decline with a reason. In the Coach, I'll make a paper call "
        "and compare it with the close. Then a quick look at the learned rules and planted violations being "
        "blocked. If anything fails, we have a backup recording."), n)
    rows = [["Tab", "What to watch"],
            ["Replayed night", "News → estimate vs bid-ask → usually no order, with the reason (or a guarded paper order)"],
            ["Coach", "Break-even after spread + fee; paper call vs the close and the agent"],
            ["Channel briefs", "Retail: model vs market and no-order reason; no market language for media / team"],
            ["Learning", "Gate keeps protective rules (skip long shots, skip chasing)"],
            ["Safety", f"Planted violations blocked ({K['policy']})"],
            ["Models", "Held-out results and the market beating our model"]]
    table(s, rows, MARGIN, TOP + 0.15, [2.6, CW - 2.6], row_h=0.66, pt=18)
    callout(s, MARGIN, 6.3, CW, 0.55, "Backup: recorded run of the same demo", pt=18, n=n)

    # 15 Recommendations ------------------------------------------------------
    n = 15
    s = new_slide(prs, "Recommendations, next steps and Q&A", (
        "Our recommendations follow from the evidence. Before any order, show the all-in cost and the break-even "
        "probability, not just the price. Warn by default when news is likely already priced in. Put education, "
        "like our Coach, in front of the first trade, with sensible default limits, and, as the Lancet "
        "Commission argues, treat betting harm as a public-health issue. For our own work: enforce a "
        "daily-loss stop, use timestamped injury reports, and extend to player-prop markets. Thank you, we are "
        "happy to take questions."), n)
    card(s, MARGIN, TOP + 0.15, 6.4, 4.0, "For products and policy", [
        "Show the all-in cost and break-even before every order",
        "Default warning: “this news is likely priced in”",
        "Education before the first trade; default stake limits",
        "Treat betting harm as a public-health issue (Lancet)",
    ], NAVY, pt=18, n=n)
    card(s, MARGIN + 6.7, TOP + 0.15, CW - 6.7, 4.0, "Our next steps", [
        "Show the full break-even in every retail brief",
        "Enforce a daily-loss stop",
        "Timestamped injury reports; player-prop markets",
    ], ORANGE, pt=18, n=n)
    textbox(s, MARGIN, 5.55, CW, 1.2, [("Thank you. Questions?", {"bold": True})], pt=40, color=NAVY,
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
    prs = build()
    prs.save(OUT)
    if args.check:
        for row in _fit_report:
            print("slide %2d  %-42s  %.2f / %.2f in" % row)
    problems = check(prs)
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(prs.slides)} slides)")
    for p in problems:
        print("WARNING:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
