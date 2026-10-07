"""Small HTML components for a controlled, presentation-focused Streamlit surface."""
from __future__ import annotations

from html import escape

import streamlit as st

from demo_ui.styles import CSS


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(model_status: str) -> None:
    st.markdown(
        f"""
        <section class="mg-hero">
          <div class="mg-hero-top"><div class="mg-brand">MarketGrade &nbsp;/&nbsp; NBA research platform</div>
          <div class="mg-pills">
            <span class="mg-pill bright">GUIDED DEMO</span>
            <span class="mg-pill">PAPER PRACTICE ONLY</span>
          </div></div>
          <h1>Follow one decision from prediction to lesson.</h1>
          <p>Choose a past game, move through three simple steps, and see whether the agent's decision held up afterward.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def phase_heading(step: str, title: str, question: str) -> None:
    st.markdown(
        f"""
        <section class="mg-phasehead">
          <div class="mg-stepbadge">{escape(step)}</div>
          <div><h2>{escape(title)}</h2><p>{escape(question)}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def journey_bar(current: str) -> None:
    phases = [
        ("Before game", "1", "Compare beliefs"),
        ("In play", "2", "Update with score"),
        ("After game", "3", "Grade the decision"),
    ]
    active = next((i for i, item in enumerate(phases) if item[0] == current), 0)
    items = []
    for index, (name, number, description) in enumerate(phases):
        state = "active" if index == active else "done" if index < active else "later"
        items.append(
            f'<div class="mg-journey-step {state}"><div class="mg-journey-number">{number}</div>'
            f'<div><strong>{escape(name)}</strong><span>{escape(description)}</span></div></div>'
        )
    st.markdown('<div class="mg-journey">' + "".join(items) + "</div>", unsafe_allow_html=True)


def plain_language_guide() -> None:
    st.markdown(
        """
        <section class="mg-guide">
          <div class="mg-guide-title">How to read this screen</div>
          <div class="mg-guide-grid">
            <div><span>1</span><strong>Model</strong><p>Our statistical estimate of who wins.</p></div>
            <div><span>2</span><strong>Market</strong><p>The probability implied by the archived price.</p></div>
            <div><span>3</span><strong>Agent</strong><p>Compares both, checks costs and chooses act or pass.</p></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def game_header(analysis: dict, provenance: str) -> None:
    game = analysis["game"]
    st.markdown(
        f"""
        <section class="mg-gamebar">
          <div>
            <div class="mg-kicker">Decision snapshot · T−{analysis['minutes_to_tip']} minutes</div>
            <div class="mg-matchup">{escape(str(game.away_team))} <span style="color:#93a19f;font-weight:500">at</span> {escape(str(game.home_team))}</div>
            <div class="mg-meta">{escape(str(game.date)[:10])} · archived quotes · as-of safe</div>
          </div>
          <div class="mg-source">{escape(provenance)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def probability_cards(analysis: dict) -> None:
    items = [
        ("Our model", analysis["raw_probability"], "Team history plus known absences", ""),
        ("Market price", analysis["market_mid"], "What the archived crowd price implied", ""),
        ("Agent's final view", analysis["agent_probability"], "Market starting point plus new information", "agent"),
    ]
    cards = []
    for label, value, note, cls in items:
        width = max(1, min(100, value * 100))
        cards.append(
            f'<div class="mg-prob {cls}">'
            f'<div class="mg-prob-label">{escape(label)}</div>'
            f'<div class="mg-prob-value">{value:.0%}</div>'
            f'<div class="mg-prob-note">{escape(note)}</div>'
            f'<div class="mg-track"><div class="mg-fill" style="width:{width:.1f}%"></div></div>'
            f'</div>'
        )
    st.markdown('<div class="mg-grid3">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def action_banner(analysis: dict) -> None:
    acted = bool(analysis["orders"])
    if acted:
        title = f"Agent decision · paper practice on {analysis['backed_team']}"
        copy = f"Why: its estimate cleared the market price and costs by {analysis['edge']:+.1%}."
        mark = "ACT"
        cls = ""
    else:
        title = "Agent decision · pass"
        copy = f"Why: the best advantage after costs was only {analysis['edge']:+.1%}. No action is a valid action."
        mark = "—"
        cls = "pass"
    st.markdown(
        f"""
        <section class="mg-action {cls}">
          <div class="mg-action-mark">{mark}</div>
          <div><div class="mg-action-title">{escape(title)}</div><div class="mg-action-copy">{escape(copy)}</div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def execution_rail(stages: list[dict]) -> None:
    cells = []
    for stage in stages:
        cells.append(
            f'<div class="mg-stage {escape(stage["state"])}">'
            f'<div class="mg-stage-dot"></div>'
            f'<div class="mg-stage-name">{escape(stage["label"])}</div>'
            f'<div class="mg-stage-detail">{escape(stage["detail"])}</div>'
            f'</div>'
        )
    st.markdown('<div class="mg-rail">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def panel(title: str, body: str, eyebrow: str = "") -> None:
    st.markdown(
        f"""
        <section class="mg-panel">
          {f'<div class="mg-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ''}
          <h3>{escape(title)}</h3><div class="mg-copy">{body}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def news_panel(news, names: dict) -> None:
    if news.empty:
        rows = '<div class="mg-copy">No official status update was available at this decision time.</div>'
    else:
        latest = news.sort_values("published_at").groupby("player_id").tail(1)
        rows = "".join(
            f'<div class="mg-news"><div class="mg-news-time">{row.published_at:%H:%M}</div>'
            f'<div class="mg-news-text"><strong>{escape(str(names.get(row.player_id, row.player_id)))}</strong> · '
            f'{escape(str(row.status).upper())}<br>{escape(str(getattr(row, "text", "")))}</div></div>'
            for row in latest.itertuples(index=False)
        )
    panel("Context available to the agent", rows, "Observe")


def scoreboard(away: str, home: str, away_score: int, home_score: int, clock: str) -> None:
    st.markdown(
        f"""
        <section class="mg-score">
          <div><div class="mg-team">{escape(away)}</div><div class="mg-score-num">{int(away_score)}</div></div>
          <div class="mg-clock">{escape(clock)}</div>
          <div><div class="mg-team">{escape(home)}</div><div class="mg-score-num">{int(home_score)}</div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(team: str, value: float, model_name: str) -> None:
    width = max(0.1, min(99.9, value * 100))
    st.markdown(
        f"""
        <section class="mg-gauge">
          <div class="mg-gauge-top"><div><div class="mg-eyebrow">Updated in-play estimate</div>
          <div style="font-weight:800">{escape(team)} win probability</div></div>
          <div class="mg-gauge-value">{value:.0%}</div></div>
          <div class="mg-bigtrack"><div class="mg-bigfill" style="width:{width:.1f}%"></div></div>
          <div class="mg-meta">{escape(model_name)} · uncalibrated fallback unless a fitted artifact is present</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def disclosure(text: str) -> None:
    st.markdown(f'<div class="mg-disclosure">{escape(text)}</div>', unsafe_allow_html=True)


def grade_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="mg-grade"><div class="mg-grade-label">{escape(label)}</div>'
        f'<div class="mg-grade-value">{escape(value)}</div><div class="mg-grade-note">{escape(note)}</div></div>'
        for label, value, note in items
    )
    st.markdown('<div class="mg-gradegrid">' + cards + "</div>", unsafe_allow_html=True)


def result_summary(analysis: dict) -> None:
    game, grade = analysis["game"], analysis["grade"]
    winner = game.home_team if game.home_pts > game.away_pts else game.away_team
    if not grade.get("filled"):
        title = "The agent passed—and there is nothing to settle."
        copy = "No entry, fee, market grade or paper profit is invented for a pass."
        tone = "pass"
        label = "DISCIPLINED PASS"
    else:
        clv = float(grade["clv"])
        quality = "beat" if clv > 0 else "trailed"
        title = f"{winner} won. The decision {quality} the closing market."
        copy = (
            f"The paper entry moved from {grade['price']:.0%} to {grade['close_price']:.0%} at tip-off, "
            f"for a market grade of {clv * 100:+.1f} cents per contract."
        )
        tone = "good" if clv > 0 else "caution"
        label = "POSITIVE MARKET GRADE" if clv > 0 else "NEGATIVE MARKET GRADE"
    st.markdown(
        f"""
        <section class="mg-result {tone}">
          <div class="mg-result-label">{escape(label)}</div>
          <h2>{escape(title)}</h2>
          <p>{escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def coach_card(advice: dict) -> None:
    lesson = advice["lesson"]
    st.markdown(
        f"""
        <section class="mg-coach">
          <div class="mg-coach-head"><div><div class="mg-eyebrow">Consumer coach</div>
          <div style="font-size:19px;font-weight:820">Check the decision, not the outcome</div></div>
          <div class="mg-nudge">{escape(advice['nudge'])}</div></div>
          <div class="mg-copy">{escape(advice['message'])}</div>
          <div class="mg-lesson"><div style="font-weight:820;font-size:13px">{escape(lesson['title'])}</div>
          <div class="mg-copy" style="font-size:12px;margin-top:5px">{escape(lesson['body'])}</div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def coach_takeaway(advice: dict) -> None:
    lesson = advice["lesson"]
    tone = "Caution" if advice["nudge"] in {"pass", "caution"} else "Lesson"
    st.markdown(
        f"""
        <section class="mg-takeaway">
          <div class="mg-takeaway-icon">COACH</div>
          <div><div class="mg-eyebrow">{escape(tone)}</div>
          <h3>{escape(lesson['title'])}</h3>
          <p>{escape(lesson['body'])}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def research_cards(inventory: dict) -> None:
    cards = []
    for group in inventory["groups"]:
        status_class = group["status"].lower().replace(" ", "-")
        items = "".join(
            f'<div class="mg-research-item"><div class="mg-research-name">{escape(name)}</div>'
            f'<div class="mg-research-desc">{escape(desc)}</div></div>'
            for name, desc in group["items"]
        )
        cards.append(
            f'<div class="mg-research"><div class="mg-status {status_class}">{escape(group["status"])}</div>{items}</div>'
        )
    st.markdown('<div class="mg-research-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        """
        <div class="mg-footer"><strong>Educational research prototype.</strong> Historical replays use archived
        quotes and as-of data. In-play scenes are explicitly synthetic. Paper actions never place real orders.
        Market prices are comparison signals—not promises, advice, or guarantees.</div>
        """,
        unsafe_allow_html=True,
    )
