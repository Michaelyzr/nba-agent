"""Simple consumer-facing NBA betting research demo.

Launch from the repository root:

    NBA_AGENT_OFFLINE=1 streamlit run demo_app.py

The page is a thin, read-only presentation layer over the existing historical
replay, market, news and MarketAgent outputs.
"""
from __future__ import annotations

import streamlit as st

from data_sources.polymarket_live import PolymarketLiveProvider
from demo_ui.adapters import (
    analyse_live_scenario,
    analyse_scenario,
    live_scenarios,
    load_runtime,
    scenarios,
)
from demo_ui.components import (
    agent_answer,
    agent_take,
    conclusion_path,
    context_list,
    footer,
    game_intro,
    inject_styles,
    live_unavailable,
    numbers_row,
    outcome_summary,
    page_header,
    reason_list,
    section_label,
)


st.set_page_config(
    page_title="NBA Research Agent",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def runtime():
    return load_runtime()


@st.cache_data(ttl=30, show_spinner=False)
def current_live_context():
    return PolymarketLiveProvider().refresh(save_history=False)


def recommendation(analysis: dict) -> str:
    if analysis["orders"]:
        return "CONSIDER"
    return "WAIT" if float(analysis["edge"]) > 0 else "PASS"


def risk_level(analysis: dict) -> str:
    steps = analysis["coach"].get("steps", [])
    if any(bool(step.get("severe")) for step in steps):
        return "High"
    if any(bool(step.get("flag")) for step in steps):
        return "Medium"
    return "Low"


def confidence_level(analysis: dict) -> str:
    edge = abs(float(analysis["edge"]))
    if edge >= 0.08:
        return "High"
    if edge >= 0.03:
        return "Medium"
    return "Low"


def take_message(analysis: dict, take: str) -> str:
    team = analysis["backed_team"]
    price_wording = "current Polymarket price" if analysis.get("is_live") else "archived market price"
    if take == "CONSIDER":
        return f"{team} may be worth considering at this {price_wording}."
    if take == "WAIT":
        return "There may be a small advantage, but it is not strong enough yet."
    return "We don't see enough value at the current market price."


def decision_reasons(analysis: dict, take: str) -> list[tuple[bool, str]]:
    team = analysis["backed_team"]
    estimate = float(analysis["agent_probability"])
    market = float(analysis["market_mid"])
    gap_points = (estimate - market) * 100
    candidate = analysis["candidate"]
    results = analysis["coach"].get("results", {})
    reasons: list[tuple[bool, str]] = []

    reasons.append(
        (
            gap_points > 0,
            f"Our estimate gives {team} a {abs(gap_points):.1f}-point "
            f"{'advantage over' if gap_points > 0 else 'shortfall versus'} the market.",
        )
    )
    shift_points = abs(float(candidate.get("shift", 0))) * 100
    if shift_points >= 0.5:
        reasons.append((True, f"The latest player availability news moved our view by {shift_points:.1f} points."))
    elif analysis.get("is_live"):
        reasons.append((True, "The live quote passed the source freshness and moneyline mapping checks."))
    priced_in = results.get("priced_in", {})
    if bool(priced_in.get("flag")):
        move = abs(float(priced_in.get("move", 0))) * 100
        reasons.append((False, f"The market already moved {move:.1f} points after the news."))
    net_edge = float(analysis["edge"]) * 100
    threshold = float(candidate.get("min_edge", 0.04)) * 100
    if take == "CONSIDER":
        reasons.append((True, f"The remaining advantage after price and costs clears our {threshold:.0f}-point bar."))
    else:
        reasons.append((False, f"Only {net_edge:+.1f} points remain after price and costs; our bar is {threshold:.0f}."))
    return reasons[:4]


def key_context(analysis: dict) -> list[str]:
    if analysis.get("context_items"):
        return list(analysis["context_items"])[:3]
    news = analysis["news"]
    if news.empty:
        return ["No official player-status update was available at this decision time."]
    latest = news.sort_values("published_at").groupby("player_id").tail(1).tail(3)
    return [str(row.text) for row in latest.itertuples(index=False)]


def answer_for(question: str, analysis: dict, take: str, risk: str) -> str:
    team = analysis["backed_team"]
    edge_points = float(analysis["edge"]) * 100
    candidate = analysis["candidate"]
    results = analysis["coach"].get("results", {})
    break_even = results.get("break_even", {})
    priced_in = results.get("priced_in", {})

    if question == "Why this take?":
        if take == "CONSIDER":
            return f"After the current price and estimated costs, the Agent still sees about {edge_points:.1f} points of room on {team}."
        return f"After the current price and estimated costs, only {edge_points:+.1f} points remain. That does not clear the Agent's safety bar."
    if question == "What's the biggest risk?":
        if bool(priced_in.get("flag")):
            move = abs(float(priced_in.get("move", 0))) * 100
            return f"The biggest risk is paying after the move: the market has already shifted {move:.1f} points since the earlier price."
        if bool(break_even.get("severe")):
            return "The current price plus estimated costs is already above the Agent's estimate, leaving no cushion."
        return f"Risk is {risk.lower()} because the estimate can still be wrong and one game is highly uncertain."
    if question == "What would change it?":
        threshold = float(candidate.get("min_edge", 0.04))
        if take == "CONSIDER":
            return "A higher market price, a reversal in player news, or a lower estimate could turn this into WAIT or PASS."
        needed = float(break_even.get("breakeven", analysis["market_ask"])) + threshold
        return f"The Agent would need a better price or an estimate near {needed:.0%} for {team} before it would consider acting."
    if question == "Is the news priced in?":
        if analysis.get("is_live"):
            return "This refresh contains a current quote but not enough earlier live prices to measure the move reliably. The Agent will not pretend otherwise."
        if bool(priced_in.get("flag")):
            move = float(priced_in.get("move", 0)) * 100
            return f"Probably at least partly. The {team} market moved {move:+.1f} points from its earlier level after the news."
        return "The structured checks did not flag a large prior market move, but that does not guarantee the news is unpriced."
    estimate = float(analysis["agent_probability"])
    market = float(analysis["market_mid"])
    gap = (estimate - market) * 100
    return (
        f"Think of the percentages as two opinions about {team}: ours is {estimate:.0%}, while the market says {market:.0%}. "
        f"The {gap:+.1f}-point gap must still be large enough to survive the price and estimated costs."
    )


def main() -> None:
    inject_styles()
    rt = runtime()
    page_header()
    requested_mode = st.query_params.get("mode")
    st.session_state.setdefault("demo_data_mode", "history" if requested_mode == "history" else "live")
    mode = st.session_state["demo_data_mode"]

    if mode == "live":
        with st.spinner("Checking current NBA markets on Polymarket..."):
            live_context = current_live_context()
        slate = live_scenarios(live_context)
        if not slate:
            reasons = [*live_context.snapshot.quality.reasons, *live_context.agent_tables.reasons]
            live_unavailable(live_context.snapshot.message, list(dict.fromkeys(reasons)))
            retry, history = st.columns(2)
            if retry.button("Retry live markets", type="primary", use_container_width=True):
                current_live_context.clear()
                st.rerun()
            if history.button("View a historical example", use_container_width=True):
                st.session_state["demo_data_mode"] = "history"
                st.rerun()
            footer("LIVE POLYMARKET")
            return
        labels = {item.game_id: item.label.split(" · ")[0].replace(" at ", " vs ") for item in slate}
        label_col, refresh_col = st.columns([4, 1])
        with label_col:
            section_label("Choose a live game")
        with refresh_col:
            if st.button("Refresh live", use_container_width=True):
                current_live_context.clear()
                st.rerun()
        selected = st.selectbox(
            "Game",
            list(labels),
            format_func=labels.get,
            label_visibility="collapsed",
            key="live_game_selector",
        )
        analysis = analyse_live_scenario(rt, live_context, selected)
    else:
        top_left, top_right = st.columns([3, 1])
        with top_left:
            section_label("Historical example")
            st.caption("Offline replay—not a current market.")
        with top_right:
            if st.button("Back to live markets", use_container_width=True):
                st.session_state["demo_data_mode"] = "live"
                st.query_params.clear()
                current_live_context.clear()
                st.rerun()
        slate = scenarios(rt)
        labels = {item.game_id: item.label.split(" · ")[0].replace(" at ", " vs ") for item in slate}
        selected = st.selectbox(
            "Historical game",
            list(labels),
            format_func=labels.get,
            label_visibility="collapsed",
            key="historical_game_selector",
        )
        analysis = analyse_scenario(rt, selected)

    game = analysis["game"]
    take = recommendation(analysis)
    risk = risk_level(analysis)
    confidence = confidence_level(analysis)

    game_intro(
        game,
        analysis["minutes_to_tip"],
        analysis["provenance"],
        analysis.get("fetched_at"),
    )
    agent_take(take, take_message(analysis, take), confidence, risk, analysis["backed_team"])
    reason_list(decision_reasons(analysis, take))
    numbers_row(
        float(analysis["agent_probability"]),
        float(analysis["market_mid"]),
        (float(analysis["agent_probability"]) - float(analysis["market_mid"])) * 100,
        analysis["backed_team"],
        bool(analysis.get("is_live")),
    )

    section_label("Key information")
    context_list(key_context(analysis))

    section_label("Ask the Agent")
    questions = [
        "Why this take?",
        "What's the biggest risk?",
        "What would change it?",
        "Is the news priced in?",
        "Explain it simply",
    ]
    question = st.pills(
        "Ask a question",
        questions,
        default=questions[0],
        selection_mode="single",
        label_visibility="collapsed",
    ) or questions[0]
    agent_answer(question, answer_for(question, analysis, take, risk))

    with st.expander("See how the Agent reached this conclusion", expanded=False):
        conclusion_path(
            key_context(analysis),
            float(analysis["agent_probability"]),
            float(analysis["market_mid"]),
            risk,
            take,
        )

    if not analysis.get("is_live"):
        close_for_team = analysis["close_home"]
        if close_for_team is not None and analysis["backed_team"] != game.home_team:
            close_for_team = 1 - float(close_for_team)
        winner = game.home_team if game.home_pts > game.away_pts else game.away_team
        outcome_summary(
            winner,
            int(game.away_pts),
            int(game.home_pts),
            game.away_team,
            game.home_team,
            close_for_team,
            float(analysis["agent_probability"]),
            analysis["backed_team"],
        )
    footer(analysis["provenance"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        inject_styles()
        st.error("This game could not be loaded. Please choose another game and try again.")
