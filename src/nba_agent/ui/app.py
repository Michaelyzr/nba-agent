"""Demo page: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from nba_agent.agents import llm, steps
from nba_agent.agents.graph import reply, start
from nba_agent.data.tables import EVAL

st.set_page_config(page_title="NBA analytics agent", layout="wide")
st.title("NBA analytics agent (prototype)")
st.caption("One LangGraph loop over a frozen 2025-26 season. Every number cites saved games; every quote cites a "
           "saved paragraph. Sample data is synthetic until download_season.py is run.")

examples = []
if (EVAL / "questions.json").exists():
    qs = json.loads((EVAL / "questions.json").read_text())
    picks = {}
    for q in qs:
        picks.setdefault(q["type"], q)
    examples = [(q["type"], q["question"], q["expect"].get("plant")) for q in picks.values()]

with st.sidebar:
    st.subheader("Settings")
    st.write("Chat model:", "Gemini" if llm.available() else "offline rules (no API key)")
    plant = st.selectbox("Inject a fault on the first try", ["none"] + sorted(steps.FAULTS))
    st.subheader("Example questions")
    for kind, text, ex_plant in examples:
        if st.button(f"{kind}: {text[:60]}", key=text + str(ex_plant)):
            st.session_state.question = text
            st.session_state.example_plant = ex_plant

question = st.text_input("Question", key="question")
col1, col2 = st.columns([1, 5])
if col1.button("Ask", type="primary") and question:
    chosen = st.session_state.pop("example_plant", None) or (None if plant == "none" else plant)
    with st.spinner("Running the loop..."):
        st.session_state.out = start(question, plant=chosen)

out = st.session_state.get("out")
if out:
    if out["paused"]:
        st.warning(out["followup"])
        answer = st.text_input("Your answer", key="answer_box")
        if st.button("Send answer") and answer:
            with st.spinner("Running the loop..."):
                st.session_state.out = reply(answer, out["config"])
            st.rerun()
    else:
        status = out.get("status")
        (st.success if status == "answered" else st.error)(f"Status: {status}")
        body, _, sources = out["note"].partition("\nSources")
        st.markdown(body)
        if sources:
            st.code("Sources" + sources, language=None)

    st.subheader("What the loop did")
    st.dataframe(pd.DataFrame(out["trace"]), use_container_width=True, hide_index=True)
    with st.expander("Form, plan, code, result"):
        st.json({"form": out.get("form"), "plan": out.get("plan")}, expanded=False)
        if out.get("code"):
            st.code(out["code"], language="python")
        if out.get("result"):
            st.json(steps.compact(out["result"]), expanded=False)
