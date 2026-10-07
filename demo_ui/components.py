"""Small presentation components for the consumer-facing demo."""
from __future__ import annotations

from html import escape

import streamlit as st

from demo_ui.styles import CSS


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header() -> None:
    st.markdown(
        """
        <header class="cs-header">
          <div class="cs-brand"><span class="cs-ball">●</span> NBA RESEARCH AGENT</div>
          <div class="cs-purpose">Clear market take, explained</div>
          <div class="cs-safety">RESEARCH ONLY</div>
        </header>
        <section class="cs-intro">
          <h1>Is this market worth considering?</h1>
          <p>Choose a game. The Agent compares its estimate with the live market, checks the available context, and gives you one clear take.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="cs-section-label">{escape(text)}</div>', unsafe_allow_html=True)


def game_intro(game, minutes_to_tip: int, provenance: str, fetched_at=None) -> None:
    if provenance == "LIVE POLYMARKET":
        updated = f" · Updated {fetched_at:%H:%M:%S UTC}" if fetched_at is not None else ""
        timing = f"LIVE POLYMARKET{updated} · {minutes_to_tip} minutes before tip-off"
        source_class = "live"
    else:
        timing = f"HISTORICAL EXAMPLE · {minutes_to_tip} minutes before tip-off"
        source_class = "history"
    st.markdown(
        f"""
        <section class="cs-game">
          <div><span>{escape(str(game.away_team))}</span><b>vs</b><span>{escape(str(game.home_team))}</span></div>
          <p class="{source_class}">{escape(str(game.date)[:10])} · {escape(timing)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def live_unavailable(message: str, reasons: list[str]) -> None:
    detail = reasons[0] if reasons else "The live source did not return a usable NBA moneyline."
    st.markdown(
        f"""
        <section class="cs-live-error">
          <div class="cs-live-dot"></div>
          <div><span>LIVE POLYMARKET UNAVAILABLE</span>
          <h2>{escape(message)}</h2>
          <p>{escape(detail)}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def agent_take(take: str, message: str, confidence: str, risk: str, team: str) -> None:
    tone = take.lower()
    st.markdown(
        f"""
        <section class="cs-take {tone}">
          <div class="cs-take-eyebrow">AGENT'S TAKE · {escape(team)}</div>
          <div class="cs-take-main">
            <div class="cs-take-word">{escape(take)}</div>
            <div class="cs-take-meta">
              <div><span>CONFIDENCE</span><strong>{escape(confidence)}</strong></div>
              <div><span>RISK</span><strong>{escape(risk)}</strong></div>
            </div>
          </div>
          <p>{escape(message)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def reason_list(reasons: list[tuple[bool, str]]) -> None:
    items = "".join(
        f'<li class="{"positive" if positive else "caution"}"><span>{"✓" if positive else "×"}</span>{escape(text)}</li>'
        for positive, text in reasons
    )
    st.markdown(
        f'<section class="cs-why"><h2>Why?</h2><ul>{items}</ul></section>',
        unsafe_allow_html=True,
    )


def numbers_row(estimate: float, market: float, gap_points: float, team: str, is_live: bool = False) -> None:
    market_note = "Current Polymarket view" if is_live else "Archived market view"
    st.markdown(
        f"""
        <section class="cs-numbers">
          <div><span>OUR ESTIMATE</span><strong>{estimate:.0%}</strong><small>{escape(team)} win chance</small></div>
          <div><span>MARKET</span><strong>{market:.0%}</strong><small>{escape(market_note)}</small></div>
          <div class="gap"><span>DIFFERENCE</span><strong>{gap_points:+.1f} pts</strong><small>Percentage points</small></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def context_list(items: list[str]) -> None:
    rows = "".join(
        f'<div class="cs-context-row"><span>i</span><p>{escape(item)}</p></div>' for item in items[:3]
    )
    st.markdown(f'<section class="cs-context">{rows}</section>', unsafe_allow_html=True)


def agent_answer(question: str, answer: str) -> None:
    st.markdown(
        f"""
        <section class="cs-answer">
          <div class="cs-answer-icon">A</div>
          <div><span>{escape(question)}</span><p>{escape(answer)}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def conclusion_path(
    news: list[str], estimate: float, market: float, risk: str, take: str
) -> None:
    news_text = news[0] if news else "No material update"
    steps = [
        ("News", news_text),
        ("Our view", f"{estimate:.0%} win chance"),
        ("Market", f"{market:.0%} implied chance"),
        ("Risk check", risk),
        ("Decision", take),
    ]
    cells = "".join(
        f'<div class="cs-path-step"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in steps
    )
    st.markdown(f'<div class="cs-path">{cells}</div>', unsafe_allow_html=True)


def outcome_summary(
    winner: str,
    away_score: int,
    home_score: int,
    away_team: str,
    home_team: str,
    close_for_team: float | None,
    estimate: float,
    team: str,
) -> None:
    if close_for_team is None:
        comparison = "A reliable closing market price is not available for this replay."
    else:
        gap = (estimate - close_for_team) * 100
        relation = "above" if gap > 0 else "below" if gap < 0 else "equal to"
        comparison = (
            f"The market closed at {close_for_team:.0%} for {team}. "
            f"Our {estimate:.0%} estimate was {abs(gap):.1f} points {relation} that final market view."
        )
    st.markdown(
        f"""
        <section class="cs-outcome">
          <div class="cs-outcome-label">WHAT HAPPENED AFTERWARD?</div>
          <h2>{escape(winner)} won</h2>
          <p class="cs-score">{escape(away_team)} {away_score} · {escape(home_team)} {home_score}</p>
          <p>{escape(comparison)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def footer(provenance: str) -> None:
    source = "Live public Polymarket data" if provenance == "LIVE POLYMARKET" else "Historical research replay"
    st.markdown(
        f'<footer class="cs-footer">{escape(source)} · Not betting advice · No real orders are placed</footer>',
        unsafe_allow_html=True,
    )
