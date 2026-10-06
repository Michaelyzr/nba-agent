"""Independent read-only in-play monitor: streamlit run inplay_app.py."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.inplay import INPLAY_RUNS

st.set_page_config(page_title="NBA in-play news and odds", layout="wide")
st.title("NBA in-play news, win probability and fair odds")
st.caption("Independent of the pregame demo. Start agents.inplay in a terminal; this page reads its logs.")
paths = sorted(INPLAY_RUNS.glob("*/inplay_snapshots.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
if not paths:
    st.info("No in-play runs yet. Run the synthetic replay or start the live in-play runner.")
    st.code("python -m agents.inplay --mode replay --source sample --game-id 401811041 --name demo")
    st.stop()
selected = st.selectbox("In-play run", paths, format_func=lambda p: p.parent.name)
refresh = st.fragment(run_every="5s") if hasattr(st, "fragment") else lambda f: f


@refresh
def render():
    st.button("Refresh")
    rows = []
    for line in Path(selected).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # The runner may currently be writing the final line.
    if not rows:
        return
    last = rows[-1]
    st.write(f"Game {last['game_id']} · {last['quote_state']} · observed {last['as_of']}")
    score = last.get("score")
    if score and score.get("source") == "synthetic_inplay_demo":
        st.info("Synthetic score/news demonstration; these are not actual game events.")
    if score:
        st.subheader(f"Home {score['home_score']} — Away {score['away_score']}")
        st.caption(f"Period {score['period']} · clock {score['clock_seconds']:.1f} seconds · source {score['source']}")
    elapsed = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last["as_of"])).total_seconds())
    if elapsed > 30 and last["quote_state"] not in {"final", "waiting_for_tip"}:
        st.warning("This run has no recent snapshot. Displayed history is not a current live quote.")
    if not last.get("model_trained", False):
        st.info("Prototype score/time model: probabilities and injury effects still need in-play calibration.")
    elif last.get("heldout_validation"):
        st.caption("Trained in-play model; holdout metrics: " + str(last["heldout_validation"]))
    if last["news_health"] == "degraded":
        st.warning("News coverage is incomplete or a source failed. See retrieval coverage below.")
    if last.get("freshness") == "source_timestamp_unverified":
        st.caption("The score endpoint has no verified update timestamp; request time alone cannot prove freshness.")
    cols = st.columns(3)
    for col, key, title in zip(cols, ("p_home", "home_decimal_odds", "away_decimal_odds"),
                               ("Home win probability", "Home fair decimal odds", "Away fair decimal odds")):
        value = last.get(key)
        col.metric(title, "Unavailable" if value is None else f"{value:.1%}" if key == "p_home" else f"{value:.3f}")
    frame = pd.DataFrame(rows)
    frame["as_of"] = pd.to_datetime(frame.as_of, utc=True)
    st.line_chart(frame.set_index("as_of")[["p_home", "p_away"]])
    for key, title in (("factors", "Player events applied"), ("player_effects", "Remaining player impact"),
                       ("source_coverage", "Retrieval coverage"), ("errors", "Source errors")):
        if last.get(key):
            with st.expander(title):
                st.dataframe(pd.DataFrame(last[key]), hide_index=True)
    if last.get("conflicts"):
        with st.expander("Conflicting evidence"):
            st.json(last["conflicts"])
    st.caption("Fair odds = 1 / probability, without bookmaker margin. Synthetic replay is not live NBA play-by-play.")


render()
