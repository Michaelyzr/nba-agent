"""Build the 10-minute presentation deck from the figures in presentation/figures/.

    python presentation/build_deck.py            # writes presentation/nba_agent_deck.pptx
    python presentation/build_deck.py --check    # also prints a fit report for every text box

Every number on the slides comes from README.md "Results" and evaluation/results/.
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

FOOTER = "DASC7606C Group 12 · Market-Graded Learning for NBA Game-Impact Intelligence"

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
    textbox(s, 0.9, 1.3, W - 1.8, 1.9,
            [("Market-Graded Learning for NBA Game-Impact Intelligence", {"bold": True})],
            pt=44, color=WHITE, slide_no=1, label="title")
    textbox(s, 0.9, 3.2, W - 1.8, 1.2,
            ["An agent that reads late injury news, grades itself against prediction-market "
             "prices, and learns rules only when they pass a backtest"],
            pt=22, color=RGBColor(0xCB, 0xD5, 0xE1), slide_no=1, label="subtitle")
    textbox(s, 0.9, 4.9, W - 1.8, 1.6,
            ["DASC7606C (COMP7606) · Group 12 · Track 2: Agentic Framework Design",
             "Presenter: Ngan Tsz Sui · Friday 9 October 2026"],
            pt=20, color=WHITE, slide_no=1, label="meta")
    s.notes_slide.notes_text_frame.text = (
        "Good morning. We are Group 12, Track 2, and our project is market-graded learning for NBA "
        "game-impact intelligence. In one sentence: we built an agent that reads late NBA injury news, "
        "turns it into forecasts and briefs, and then grades itself against real prediction-market prices. "
        "It only keeps the lessons it learns if they pass a backtest on earlier days.")

    # 2 Problem -----------------------------------------------------------
    n = 2
    s = new_slide(prs, "The problem: late news moves win probabilities", (
        "Late news, such as a starter ruled out shortly before tip-off, changes who is likely to win. "
        "Prediction markets react to that news too, so the real question is whether an agent can act on "
        "public news before prices fully adjust. We also want it to score itself honestly against the "
        "market and learn from its mistakes. One forecast core serves several channels; they differ in the "
        "risk they may take, not in what they see."), n)
    textbox(s, MARGIN, TOP + 0.2, 6.6, 5.4, [
        "A starter ruled out shortly before tip-off changes a game's win probability",
        "Can an agent act on **public** news before prediction-market prices fully adjust?",
        "Grade every decision against the market price, not just win/loss",
        "Learn from mistakes, but keep a rule only if it passes a backtest on earlier days",
    ], pt=22, bullets=True, space=16, slide_no=n)
    x0 = 7.5
    rect(s, x0, TOP + 0.2, W - MARGIN - x0, 5.45, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.04
    textbox(s, x0 + 0.2, TOP + 0.3, 4.9, 0.6, [("Who uses it (one forecast core)", {"bold": True})],
            pt=20, color=NAVY, slide_no=n)
    users = [("Platforms (fantasy, media sites)", "audited projection refresh", BLUE),
             ("Media", "“who gains, who loses”, no bets", BLUE),
             ("Teams", "internal brief, no market signal", BLUE),
             ("Retail", "model vs market, guarded trade", ORANGE),
             ("Coach (learners)", "how markets and news work", GREEN)]
    for i, (who, what, col) in enumerate(users):
        y = TOP + 0.95 + i * 0.93
        rect(s, x0 + 0.2, y, 0.1, 0.75, col)
        textbox(s, x0 + 0.4, y - 0.05, 4.6, 0.85, [(who, {"bold": True, "color": col}), (what, {"pt": 18})],
                pt=19, space=0, slide_no=n, label=who)

    # 3 Whole product -------------------------------------------------------
    n = 3
    s = new_slide(prs, "The whole product, end to end", (
        "Here is the whole system. On the left is public data only: ESPN box scores and inactive lists for "
        "three seasons, and Kalshi game-winner prices for 2025-26, which is 1,316 games and about 3.9 million "
        "one-minute price rows. The models are trained once, then a replay walks through the season day by "
        "day and gives the agent an as-of view, so it can never see future information. The agent decides, "
        "reviews nightly, and its outputs go to channel briefs, paper orders with an audit log, the Coach, "
        "and our evaluation."), n)
    picture(s, "product_workflow.png", MARGIN, TOP, CW, 4.85)
    textbox(s, MARGIN, 6.1, CW, 0.8, [
        "**Data:** ESPN box scores + inactive lists, 2023-24 to 2025-26 · Kalshi game-winner prices, "
        "2025-26: 1,316 games, ~3.9M one-minute price rows"],
        pt=18, align=PP_ALIGN.CENTER, slide_no=n)

    # 4 Models -------------------------------------------------------------
    n = 4
    s = new_slide(prs, "Models M1–M5: only the M4 win model drives trades", (
        "We trained four models plus a wrapper. The GRU beats both baselines on pinball loss for points and "
        "minutes, and its 10-to-90 percent intervals cover about 80 percent of outcomes, which is what a "
        "calibrated interval should do. The play classifier beats simple rules on Brier score. Importantly, "
        "only M4, the win model, drives trades; M2 and M3 feed the briefs and player tables. M5 is the single "
        "as-of interface that the agent, replay and demo all call."), n)
    rows = [
        ["Model", "What it predicts", "Held-out result", "Used for"],
        ["M1 Baselines", "10-game rolling average; gradient boosting (GBM) with teammates-out",
         "Pinball: rolling 1.641 pts / 1.966 min; GBM 1.590 / 1.724", "Reference for M2"],
        ["M2 GRU", "Points and minutes quantiles from each player's last 20 games",
         "Pinball **1.553 pts / 1.680 min**; 10–90% coverage 0.81 / 0.80", "Briefs, player tables"],
        ["M3 Play classifier", "P(player plays)", "Brier **0.143** vs rules 0.169–0.172", "Briefs, player tables"],
        ["M4 Win model", "P(home wins), team ratings adjusted for who is out",
         "Brier **0.183** with absences, accuracy 74% (501 test games)", "**News shift → trades**"],
        ["M5 Forecaster", "One as-of interface over M1–M4, with “player X out” overrides",
         "No future data (leakage tests)", "Agent, replay, Coach, demo"],
    ]
    table(s, rows, MARGIN, TOP + 0.1, [2.3, 3.9, 4.0, 2.13], row_h=0.78, pt=16, emphasis_row=4,
          header_h=0.5)
    callout(s, MARGIN, 6.3, CW, 0.58,
            "Only M4 drives trades. M2 and M3 feed briefs and player tables.", pt=20, n=n)

    # 5 M4 vs market -------------------------------------------------------
    n = 5
    s = new_slide(prs, "Our win model vs the market: the market is better", (
        "We compared M4 to the Kalshi price on 501 test games. The market at tip has a Brier score of 0.163; "
        "M4 with absences known is 0.183. When the two disagree by more than five points, the market is the "
        "one that is right, in both directions. So the lesson was: do not trade the raw model. That shaped "
        "the whole trading design you will see next."), n)
    picture(s, "m4_vs_market.png", MARGIN, TOP + 0.05, 7.6, 5.7, align="left")
    textbox(s, 8.3, TOP + 0.2, W - MARGIN - 8.3, 5.5, [
        "Brier, 501 test games: **market at tip 0.163**, M4 0.183",
        "When they disagree by more than 5 points, the market is right:",
        ("M4 higher on home: M4 48%, market 35%, actual 34%", {"level": 1}),
        ("M4 lower on home: M4 61%, market 73%, actual 77%", {"level": 1}),
        ("Lesson: don't trade the raw model; use only its news shift", {"bold": True, "color": ORANGE}),
    ], pt=20, bullets=True, space=12, slide_no=n)

    # 6 Agent loop ---------------------------------------------------------
    n = 6
    s = new_slide(prs, "The agent loop: Decide, then nightly Review", (
        "The agent is one LangGraph with two phases. Decide runs at every public news item in the six hours "
        "before tip and once at tip minus 60 minutes: investigate, forecast, analyse, then propose or take "
        "no action, then code checks with one retry, risk limits, and confirm or block. Review runs once a "
        "night: settle, the reviewer proposes one rule, and a separate gate decides whether it goes into the "
        "notebook. The LLM, Gemini, is optional and only used in investigate, analyse and review; by default "
        "everything runs offline and deterministically."), n)
    picture(s, "agent_workflow.png", MARGIN, TOP, 9.2, 5.75, align="left")
    textbox(s, 9.85, TOP + 0.3, W - MARGIN - 9.85, 5.5, [
        "**Decide:** each news item ≤ 6 h before tip, and tip − 60 min",
        "**Checks:** one retry, then blocked",
        "**Review:** nightly; one rule proposed, the gate decides",
        "**LLM (Gemini):** optional, only in investigate / analyse / review; default offline and deterministic",
    ], pt=18, bullets=True, space=12, slide_no=n)

    # 7 Trade flow ---------------------------------------------------------
    n = 7
    s = new_slide(prs, "How one trade is decided", (
        "Because the market beats our model, we use the market as the base rate. The anchor is the market "
        "mid 24 hours before tip, and our estimate adds only M4's news shift. The gap is our probability for a "
        "side minus the price minus the Kalshi fee. We trade only if the gap is over 4 points, no notebook "
        "rule says skip, and we do not already hold the game. Risk caps are code: $50 per order, $100 per game, "
        "$300 per day, with a $20 stake."), n)
    picture(s, "trade_flow.png", MARGIN, TOP, 9.0, 5.75, align="left")
    textbox(s, 9.6, TOP + 0.2, W - MARGIN - 9.6, 5.6, [
        "**Anchor** = market mid 24 h before tip",
        "**Estimate** = anchor + M4 news shift",
        "**Gap** = P(side) − price − fee, fee = 7%·p·(1−p)",
        "**Trade if** gap > 4 pts, no rule skip, one position per game",
        "**$20 stake;** caps $50 / order, $100 / game, $300 / day",
    ], pt=18, bullets=True, space=10, slide_no=n)

    # 8 Worked example -----------------------------------------------------
    n = 8
    s = new_slide(prs, "Worked example: CLE @ POR, 1 February 2026", (
        "Here is one real trade from the test period. At 01:30 UTC the inactive list showed four Portland "
        "players out: Avdija, Henderson, Murray and Thybulle. The market had Portland at 42 percent 24 hours "
        "earlier; M4's news shift was minus 8.7 points, so our estimate was 33.8 percent. We bought Portland NO "
        "at 55 cents plus a 1.7 cent fee, a gap of 9.5 points. The price closed at 62.5 percent, so we beat "
        "the close by 7.5 cents, and Cleveland won 130 to 111 for a profit of $15.57."), n)
    picture(s, "trade_example.png", MARGIN, TOP, CW, 4.4)
    textbox(s, MARGIN, 5.7, CW / 2 - 0.1, 1.25, [
        "**News 01:30 UTC:** Avdija, Henderson, Murray, Thybulle out",
        "**POR:** anchor 42%, shift −8.7 → 33.8%",
    ], pt=18, bullets=True, space=4, slide_no=n)
    textbox(s, MARGIN + CW / 2 + 0.1, 5.7, CW / 2 - 0.1, 1.25, [
        "**Bought NO at 55¢**, break-even 56.7%; gap +9.5",
        "**Close 62.5%**, CLV +7.5¢; **P&L +$15.57**",
    ], pt=18, bullets=True, space=4, slide_no=n)

    # 9 Learning -----------------------------------------------------------
    n = 9
    s = new_slide(prs, "Learning: propose one rule a day, keep it only if it helps", (
        "Every night the reviewer looks at the worst closing-line-value slice of settled trades and proposes "
        "one machine-checkable rule. The agent that writes a rule never approves it: the gate backtests it on "
        "earlier days only, and keeps it if mean CLV improves. Over the test period it proposed 37 rules and "
        "kept only 5, for example skip sides priced at or below 35 cents, and skip when the market has already "
        "moved 2 cents or more against us. In development it proposed 21 and kept 2."), n)
    steps = [("1  Reviewer", ["Looks at the worst-CLV slice of settled trades", "Proposes one rule per day"], BLUE),
             ("2  Gate", ["Backtests on up to 14 earlier days only", "Keeps it if mean CLV improves by 0.005 over ≥ 3 changed trades"], ORANGE),
             ("3  Notebook", ["Active from the next day", "Rules expire after 45 days"], GREEN)]
    bw, gap = 3.85, 0.39
    for i, (head, lines, col) in enumerate(steps):
        x = MARGIN + i * (bw + gap)
        card(s, x, TOP + 0.15, bw, 2.55, head, lines, col, pt=18, n=n)
        if i < 2:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.04), Inches(TOP + 1.2),
                                   Inches(gap - 0.08), Inches(0.4))
            a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()
    y = TOP + 2.95
    for i, (big, small, col) in enumerate([("37 → 5", "rules proposed → kept, test period", NAVY),
                                           ("21 → 2", "rules proposed → kept, development", GREY)]):
        x = MARGIN + i * 3.0
        textbox(s, x, y, 2.9, 0.9, [(big, {"bold": True})], pt=40, color=col, space=0, slide_no=n, label=big)
        textbox(s, x, y + 0.9, 2.9, 0.8, [small], pt=18, color=GREY, space=0, slide_no=n, label=small)
    textbox(s, 6.5, y, W - MARGIN - 6.5, 2.8, [
        ("Examples of kept rules", {"bold": True, "color": NAVY, "heading": True}),
        "Skip when buying a side priced at or below 35¢",
        "Skip when the market has already moved 2¢ or more against us",
        "The agent that writes a rule never approves it",
    ], pt=19, bullets=True, space=8, slide_no=n)

    # 10 Results -----------------------------------------------------------
    n = 10
    s = new_slide(prs, "Results on the test period (1 Feb – 12 Apr 2026)", (
        "These are test-period results on real Kalshi prices, the same prices for every setup, with fills at "
        "the ask plus fees. Trading the raw model loses $2,087 on 366 trades and has 10 days that lost over "
        "$100. Anchoring to the market cuts that to minus $247, and adding learning brings it to minus $32 on "
        "74 trades, with the maximum drawdown down to $195. We are still slightly negative, and we say so."), n)
    picture(s, "headline_clv.png", MARGIN, TOP + 0.05, 6.3, 3.6, align="left")
    rows = [["Setup", "Trades", "P&L", "ROI", "Max DD"],
            ["**Full agent** (anchor + learning)", "**74**", "**−$32**", "**−2.2%**", "**$195**"],
            ["Anchor, no learning", "129", "−$247", "−9.7%", "$439"],
            ["Win-rate placeholder model", "246", "−$429", "−8.9%", "$599"],
            ["Raw model / no agent", "366", "−$2,087", "−28.8%", "$2,326"]]
    table(s, rows, 6.95, TOP + 0.1, [2.3, 0.9, 1.05, 0.95, 1.13], row_h=0.7, pt=16, emphasis_row=1,
          header_h=0.5, center_cols=(1, 2, 3, 4))
    textbox(s, MARGIN, 5.0, CW, 1.9, [
        "Same recorded Kalshi prices for every setup; $20 stake; fills at the ask + fee, capped by volume",
        "Raw model / no agent: **10 kill-switch days** (days losing > $100); the full agent: 0",
        "Raw model → market anchor → anchor + learning: **−$2,087 → −$247 → −$32**",
    ], pt=18, bullets=True, space=6, slide_no=n)

    # 11 Selectivity -------------------------------------------------------
    n = 11
    s = new_slide(prs, "Selectivity: from 771 decision points to 74 trades", (
        "This funnel shows how selective the agent is. Of 771 decision points in the test period, only 153 "
        "had a gap over 4 points after fees, and after notebook rules and the one-position-per-game limit, "
        "74 were traded. 27 of those beat or matched the closing price and 41 won. Most of the value comes from "
        "the trades the agent declines."), n)
    picture(s, "trade_funnel.png", MARGIN, TOP, 8.7, 5.7, align="left")
    textbox(s, 9.35, TOP + 0.4, W - MARGIN - 9.35, 5.3, [
        "**771** decision points",
        "**153** with gap > 4 pts after fees",
        "**74** trades (10%)",
        "**27** beat or matched the close (36%)",
        "**41** won (55%)",
        ("Value comes from skipping bad trades", {"bold": True, "color": ORANGE}),
    ], pt=20, bullets=True, space=12, slide_no=n)

    # 12 Safety and honesty ------------------------------------------------
    n = 12
    s = new_slide(prs, "Safety and honest caveats", (
        "On safety: all 9 planted policy violations were blocked, including over-cap orders, post-tip orders, "
        "orders from the team and media channels, and 'guaranteed lock' copy. The replay never shows future "
        "information, we use public data only, and all money is paper money. On honesty: mean CLV is still "
        "slightly negative in every setup, so we avoid bad trades rather than beat the close. The kill switch "
        "is measured, not enforced; injury news is stamped 30 minutes before tip, which is conservative; and "
        "'development rules frozen' equals 'no learning' because both development rules expired."), n)
    card(s, MARGIN, TOP + 0.15, 5.9, 5.55, "Safety", [
        "**9 / 9** planted policy violations blocked (over-cap, post-tip, team / media orders, "
        "“guaranteed lock” copy, retail without confirm, …)",
        "No future information: as-of view + leakage tests",
        "Public data only; paper money",
        "Risk limits are code; the LLM cannot override them",
    ], GREEN, pt=18, n=n)
    card(s, MARGIN + 6.2, TOP + 0.15, CW - 6.2, 5.55, "Honest caveats", [
        "Mean CLV still slightly negative everywhere: we avoid bad trades rather than beat the close",
        "Kill switch is measured, not enforced",
        "Inactive-list news stamped tip − 30 min (conservative; real reports come earlier)",
        "“Dev rules frozen” = “no learning”: both dev rules expired (45 days) before the test",
    ], RED, pt=18, n=n)

    # 13 Coach -------------------------------------------------------------
    n = 13
    s = new_slide(prs, "Coach: learning how the market works", (
        "The Coach tab is for learners. At a chosen decision time it explains the game using only what was "
        "public then: the news, our model before and after, the market now versus 24 hours earlier, and the "
        "break-even price after spread and fee. Lesson cards are triggered by the situation. The user can back "
        "a team or pass with a paper stake, then see the closing price, result and what the agent did. It is "
        "educational and uses paper money only."), n)
    steps = [("1  What's going on", ["News so far, model before / after", "Market now vs 24 h earlier; break-even after spread + fee"], BLUE),
             ("2  Lessons", ["Price = probability · spread + fee break-even · injury shift",
                             "Priced in / chasing · long shots · closing line · passing"], ORANGE),
             ("3  Your call", ["Back a team or pass, paper stake", "Then: close, CLV, result, P&L, and what the agent did"], GREEN),
             ("4  Habit feedback", ["Paying above the close, long shots, chasing", "Small samples are mostly luck"], NAVY)]
    cw2 = (CW - 0.3) / 2
    for i, (head, lines, col) in enumerate(steps):
        x = MARGIN + (i % 2) * (cw2 + 0.3)
        y = TOP + 0.15 + (i // 2) * 2.45
        card(s, x, y, cw2, 2.25, head, lines, col, pt=18, n=n)
    callout(s, MARGIN, 6.3, CW, 0.55,
            "Educational, paper money only · only information public at that moment · no “lock” wording",
            pt=18, n=n)

    # 14 Demo --------------------------------------------------------------
    n = 14
    s = new_slide(prs, "Live demo: streamlit run app.py", (
        "Now the live demo. I'll start on the replayed night, where late news changes a status and the briefs "
        "and model-versus-market view update. Then the Coach tab, briefly the channel briefs, the learning tab "
        "with a rule the gate accepted, the safety tab with planted violations being blocked, and the models "
        "tab. If anything fails, we have a backup recording."), n)
    rows = [["Tab", "What we show"],
            ["Replayed night", "Late news → forecast before / after, model vs market, guarded paper order"],
            ["Coach", "Explain one decision time; take a paper trade and compare with the agent"],
            ["Channel briefs", "Same forecast, four channels: platform, media, team, retail"],
            ["Learning", "Reviewer proposes a rule; the gate keeps or rejects it; rule reused later"],
            ["Safety", "Planted violations blocked (9 / 9)"],
            ["Models", "Held-out results and M4 vs the market"]]
    table(s, rows, MARGIN, TOP + 0.15, [2.8, CW - 2.8], row_h=0.66, pt=18)
    callout(s, MARGIN, 6.3, CW, 0.55, "Backup: recorded run of the same demo", pt=18, n=n)

    # 15 Conclusion --------------------------------------------------------
    n = 15
    s = new_slide(prs, "Conclusion, next steps and Q&A", (
        "To conclude: the market is a stronger forecaster than our model, and we report that. By anchoring to "
        "the market and learning gated rules, test-period losses went from minus $2,087 to minus $32 on far "
        "fewer trades, while every planted safety violation was blocked. Next we would enforce a daily-loss "
        "stop, use timestamped injury reports, extend to player-prop markets, and evaluate the LLM mode against "
        "the offline rules. Thank you, we are happy to take questions."), n)
    card(s, MARGIN, TOP + 0.15, 6.4, 4.0, "What we showed", [
        "The market beats our win model; we use it as the base rate",
        "Anchor + gated learning: −$2,087 → −$32 on 74 trades",
        "Guardrails are code: 9 / 9 violations blocked",
    ], NAVY, pt=18, n=n)
    card(s, MARGIN + 6.7, TOP + 0.15, CW - 6.7, 4.0, "Next steps", [
        "Enforce a daily-loss stop",
        "Timestamped injury reports instead of the tip − 30 min proxy",
        "Player-prop markets",
        "Evaluate LLM (Gemini) mode vs offline rules",
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
