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
          <div class="mg-brand">MarketGrade &nbsp;/&nbsp; NBA research platform</div>
          <h1>One game. Five questions. No hindsight.</h1>
          <p>What did the model believe? What did the market believe? What did the agent do?
          What happened next—and what should a consumer learn from it?</p>
          <div class="mg-pills">
            <span class="mg-pill bright">HISTORICAL REPLAY</span>
            <span class="mg-pill">OFFLINE READY</span>
            <span class="mg-pill">{escape(model_status)}</span>
            <span class="mg-pill">PAPER PRACTICE ONLY</span>
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
        ("Model belief", analysis["raw_probability"], "Raw pregame estimate after known absences", ""),
        ("Market belief", analysis["market_mid"], f"Archived bid–ask {analysis['market_bid']:.0%}–{analysis['market_ask']:.0%}", ""),
        ("Agent estimate", analysis["agent_probability"], "Earlier market anchor + model news shift", "agent"),
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
        title = f"Paper action · back {analysis['backed_team']}"
        copy = f"Estimated edge after spread and fee: {analysis['edge']:+.1%}. Existing safety checks passed."
        mark = "GO"
        cls = ""
    else:
        title = "Pass · the threshold was not cleared"
        copy = f"Best estimated edge after costs: {analysis['edge']:+.1%}. Restraint is a valid decision."
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
