"""Polished lifecycle demo for the market-graded consumer education platform.

Launch from the repository root:

    NBA_AGENT_OFFLINE=1 streamlit run demo_app.py

This entry point is additive.  It reads the existing agent, replay, Coach,
forecast and evaluation interfaces without changing their state or training.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agents import coach, coach_agent
from demo_ui.adapters import analyse_scenario, inplay_story, load_runtime, research_inventory, scenarios
from demo_ui.components import (
    action_banner,
    coach_card,
    coach_takeaway,
    disclosure,
    execution_rail,
    footer,
    grade_cards,
    hero,
    inject_styles,
    journey_bar,
    news_panel,
    panel,
    phase_heading,
    plain_language_guide,
    probability_cards,
    probability_gauge,
    research_cards,
    result_summary,
    scoreboard,
)
from demo_ui.lifecycle import execution_stages


st.set_page_config(
    page_title="MarketGrade · NBA research platform",
    page_icon="MG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=False)
def runtime():
    return load_runtime()


def set_phase(phase: str) -> None:
    st.session_state["lifecycle_phase"] = phase
    st.session_state["scroll_request_id"] = st.session_state.get("scroll_request_id", 0) + 1
    st.session_state["scroll_to_story"] = True


def scroll_to_story() -> None:
    """Return the presenter to the active step after a Streamlit rerun."""
    if not st.session_state.pop("scroll_to_story", False):
        return
    scroll_request = st.session_state.get("scroll_request_id", 1)
    phase_key = st.session_state.get("lifecycle_phase", "Before game").lower().replace(" ", "-")
    components.html(
        f"""
        <script>
          // A changing request id forces this helper to remount after each step.
          const scrollRequest = {scroll_request};
          const activePhase = "{phase_key}";
          const revealActiveStep = () => {{
            const doc = window.parent.document;
            const target = doc.querySelector('.mg-journey');
            if (target) target.scrollIntoView({{behavior: 'auto', block: 'start'}});
          }};
          [120, 600, 1400, 2400].forEach(delay => window.setTimeout(revealActiveStep, delay));
        </script>
        """,
        height=0,
    )


def navigation_controls(phase: str) -> None:
    st.markdown('<div class="mg-control-label">Continue the guided story</div>', unsafe_allow_html=True)
    if phase == "Before game":
        _, primary = st.columns([1, 1.15], gap="medium")
        with primary:
            st.button(
                "Next: see the in-play update →",
                type="primary",
                use_container_width=True,
                on_click=set_phase,
                args=("In play",),
                key="next_inplay",
            )
    elif phase == "In play":
        back, primary = st.columns([1, 1.15], gap="medium")
        with back:
            st.button(
                "← Back to pregame",
                use_container_width=True,
                on_click=set_phase,
                args=("Before game",),
                key="back_before",
            )
        with primary:
            st.button(
                "Next: reveal result and grade →",
                type="primary",
                use_container_width=True,
                on_click=set_phase,
                args=("After game",),
                key="next_after",
            )
    else:
        back, restart = st.columns([1, 1.15], gap="medium")
        with back:
            st.button(
                "← Back to in-play",
                use_container_width=True,
                on_click=set_phase,
                args=("In play",),
                key="back_inplay",
            )
        with restart:
            st.button(
                "Start this story again",
                type="primary",
                use_container_width=True,
                on_click=set_phase,
                args=("Before game",),
                key="restart_story",
            )


def price_chart(frame: pd.DataFrame, team: str, decision_time) -> alt.Chart:
    if frame.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
    line = (
        alt.Chart(frame)
        .mark_line(color="#0d7b72", strokeWidth=2.5)
        .encode(
            x=alt.X("ts:T", title=None, axis=alt.Axis(format="%H:%M", labelColor="#667673", grid=False)),
            y=alt.Y(
                "mid:Q",
                title=f"P({team} wins)",
                scale=alt.Scale(domain=[max(0, frame.mid.min() - 0.06), min(1, frame.mid.max() + 0.06)]),
                axis=alt.Axis(format="%", labelColor="#667673", gridColor="#edf0ef"),
            ),
            tooltip=[alt.Tooltip("ts:T", title="UTC"), alt.Tooltip("mid:Q", title="Market", format=".1%")],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"ts": [decision_time]}))
        .mark_rule(color="#ef735d", strokeDash=[4, 4])
        .encode(x="ts:T")
    )
    return (line + marker).properties(height=225).configure_view(strokeWidth=0)


def before_game(rt, analysis: dict) -> None:
    phase_heading("1 / 3", "Before the game", "Question: Do the model and market disagree enough for the agent to act?")
    st.caption("Source: HISTORICAL REPLAY · Archived prices and information available 30 minutes before tip-off.")
    action_banner(analysis)
    plain_language_guide()
    probability_cards(analysis)

    with st.expander("See the evidence the agent checked (optional)", expanded=False):
        st.markdown("#### Structured execution")
        execution_rail(
            execution_stages(
                analysis["decision"].get("trace", []), "Before game", bool(analysis["orders"])
            )
        )
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            st.markdown("#### Archived market movement")
            st.altair_chart(
                price_chart(analysis["price_frame"], analysis["backed_team"], analysis["now"]),
                use_container_width=True,
            )
        with right:
            news_panel(analysis["news"], rt.names)
        reason = analysis["orders"][0].reason if analysis["orders"] else analysis["candidate"].get("why_not", "No action")
        st.markdown("#### Why the agent acted—or passed")
        st.write(reason)
        st.caption("Structured system output. Internal model reasoning and chain-of-thought are not displayed.")


def in_play(rt, analysis: dict) -> None:
    phase_heading("2 / 3", "During the game", "Question: How do score and time change the win probability?")
    story = inplay_story(rt, analysis)
    if not story["available"]:
        disclosure(f"In-play scene unavailable: {story['reason']}")
        return
    disclosure("Demo mode: this score and event sequence is synthetic. It demonstrates the existing in-play model without pretending it is the archived game feed.")
    checkpoints = story["checkpoints"]
    index = st.select_slider(
        "Replay checkpoint",
        options=list(range(len(checkpoints))),
        value=min(5, len(checkpoints) - 1),
        format_func=lambda i: checkpoints[i]["clock_label"],
    )
    point = checkpoints[index]
    game = analysis["game"]
    scoreboard(game.away_team, game.home_team, point["away_score"], point["home_score"], point["clock_label"])
    probability_gauge(game.home_team, point["p_home"], story["model_name"])
    left, right = st.columns([1, 1], gap="large")
    with left:
        body = (
            f"The pregame prior was <strong>{story['prior']:.0%}</strong> for {game.home_team}. "
            f"At {point['clock_label']}, score and time move the estimate to <strong>{point['p_home']:.0%}</strong>."
        )
        panel("What changed?", body, "Update")
    with right:
        events = point["events"]
        if events:
            body = "<br>".join(f"<strong>{e['status'].replace('_', ' ').title()}</strong> · {e['text']}" for e in events[-3:])
        else:
            body = "No synthetic context event has appeared at this checkpoint."
        panel("Context feed", body, "Synthetic events")
    with st.expander("See the in-play system boundary (optional)", expanded=False):
        execution_rail(
            [
                {"label": "Observe", "state": "done", "detail": "score snapshot"},
                {"label": "Forecast", "state": "done", "detail": "time + margin"},
                {"label": "Compare market", "state": "later", "detail": "no in-play market feed"},
                {"label": "Safety check", "state": "done", "detail": "source disclosed"},
                {"label": "Act / pass", "state": "later", "detail": "display only"},
                {"label": "Market grade", "state": "later", "detail": "not applicable"},
                {"label": "Review / learn", "state": "later", "detail": "separate replay"},
            ]
        )


def after_game(analysis: dict) -> None:
    phase_heading("3 / 3", "After the game", "Question: Was the decision well-timed, regardless of one win or loss?")
    game = analysis["game"]
    scoreboard(game.away_team, game.home_team, game.away_pts, game.home_pts, "FINAL · HISTORICAL")
    result_summary(analysis)
    grade = analysis["grade"]
    if grade.get("filled"):
        clv = float(grade["clv"])
        pnl = float(grade["pnl"])
        items = [
            ("Game result", "Won" if pnl > 0 else "Lost", "One result is mostly luck"),
            ("Entry price", f"{grade['price']:.0%}", "Archived price when the agent acted"),
            ("Closing price", f"{grade['close_price']:.0%}", "Market price at tip-off"),
            ("Market grade", f"{clv * 100:+.1f}¢", "Positive means the entry beat the close"),
        ]
    else:
        items = [
            ("Agent call", "Pass", "No paper position"),
            ("Entry → close", "—", "Nothing was purchased"),
            ("Market grade", "No grade", "CLV requires an entry"),
            ("Paper result", "$0.00", "Passing costs nothing"),
        ]
    grade_cards(items)
    left, right = st.columns([1, 1], gap="large")
    with left:
        market_text = "unavailable" if analysis["market_brier"] is None else f"{analysis['market_brier']:.3f}"
        panel(
            "Forecasts are graded, not celebrated",
            f"For this game, squared probability error was <strong>{analysis['model_brier']:.3f}</strong> for the raw model "
            f"and <strong>{market_text}</strong> for the closing market. Lower is better; one game cannot establish skill.",
            "Outcome review",
        )
    with right:
        if grade.get("filled"):
            verdict = "beat" if grade["clv"] > 0 else "trailed"
            body = (
                f"The paper entry {verdict} the closing price by <strong>{abs(grade['clv']) * 100:.1f} cents</strong>. "
                "That market grade is more informative about decision quality than one win or loss."
            )
        else:
            body = "The agent passed, so the honest review is simple: there is no position, CLV, fee, or simulated profit to report."
        panel("What should we learn?", body, "Market grade")
    with st.expander("See the completed system lifecycle (optional)", expanded=False):
        execution_rail(
            execution_stages(
                analysis["decision"].get("trace", []), "After game", bool(analysis["orders"])
            )
        )
        if grade.get("filled"):
            st.caption(f"Paper settlement after archived fees: {'+' if pnl >= 0 else '−'}${abs(pnl):.2f}.")


def coach_section(rt, analysis: dict, phase: str) -> None:
    st.markdown("## Ask the Coach")
    st.caption("Try a team—or choose Pass. Coach explains the decision in plain language using only information available at that moment.")
    options = [analysis["game"].away_team, analysis["game"].home_team, "Pass"]
    default = options.index(analysis["backed_team"]) if analysis["backed_team"] in options else 2
    choice = st.pills("Learner's paper call", options, default=options[default], selection_mode="single") or options[default]
    advice = coach_agent.advise(analysis["snapshot"], None if choice == "Pass" else choice, history=[], llm=False)
    coach_takeaway(advice)
    if phase != "After game":
        practice = (
            f"Your paper call is <strong>{choice}</strong>. The outcome stays hidden until After game. "
            "The existing League interface repeats this exercise over a fixed slate and ranks sustained closing-line value, not one-game luck."
        )
    elif choice == "Pass":
        practice = (
            "Your paper call was <strong>Pass</strong>: no entry, no fee, no CLV and no profit or loss. "
            "League records restraint as a completed decision."
        )
    else:
        side = analysis["snapshot"]["sides"].get(choice)
        trade = None if side is None else coach.paper_trade(
            rt.replay, analysis["game"], analysis["now"], side, 10.0, analysis["snapshot"]
        )
        if not trade or not trade.get("filled"):
            practice = f"The {choice} paper call could not fill at this archived snapshot; no value is fabricated."
        else:
            practice = (
                f"A $10 paper call on <strong>{choice}</strong> entered at <strong>{trade['price']:.0%}</strong>, "
                f"closed at <strong>{trade['close_price']:.0%}</strong> and earned a market grade of "
                f"<strong>{trade['clv'] * 100:+.1f} cents</strong> per contract. League uses this same replay settlement."
            )
    panel("Paper practice", practice, "League preview")
    with st.expander("Read the Coach's full explanation and checks", expanded=False):
        coach_card(advice)
        table = coach_agent.trace_table(advice)
        if table.empty:
            st.write("Passing needs no fee, price, or timing check.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("Paper education only. The League module uses the same Coach snapshot and replay settlement interfaces.")


def main() -> None:
    inject_styles()
    rt = runtime()
    hero(rt.model_status)
    slate = scenarios(rt)
    labels = {item.game_id: item.label for item in slate}
    st.session_state.setdefault("lifecycle_phase", "Before game")
    st.markdown('<div class="mg-control-label">Start here · Choose a past game</div>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Scenario",
        list(labels),
        format_func=labels.get,
        label_visibility="collapsed",
        key="scenario_id",
        on_change=set_phase,
        args=("Before game",),
    )
    phase = st.session_state["lifecycle_phase"]

    analysis = analyse_scenario(rt, selected)
    journey_bar(phase)
    scroll_to_story()
    if phase == "Before game":
        before_game(rt, analysis)
    elif phase == "In play":
        in_play(rt, analysis)
    else:
        after_game(analysis)

    navigation_controls(phase)
    coach_section(rt, analysis, phase)
    with st.expander("Research Lab · core, experimental and evaluation-only", expanded=False):
        inventory = research_inventory(rt)
        if inventory["metrics"]:
            cols = st.columns(len(inventory["metrics"]))
            for column, metric in zip(cols, inventory["metrics"]):
                column.metric(f"{metric['label']} · {metric['unit']}", metric["value"])
        research_cards(inventory)
        st.caption("Metrics are read from committed evaluation result files; opening this panel does not rerun experiments.")
    footer()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        inject_styles()
        st.error("The local demo could not assemble this scenario. Check that data/sample is present and try another replay.")
        st.caption("The presentation UI intentionally suppresses raw stack traces. Run the test suite for diagnostics.")
