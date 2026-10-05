"""E4: Streamlit demo of one replayed night across channels (README section 7).

    streamlit run app.py

Reads data/frozen/ if it has prices, otherwise the committed data/sample/.
Models come from models/ (python -m forecast.train); runs from runs/.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import replay
from agents.briefs import channel_briefs, outlook
from agents.graph import FAULTS, OUT_STATUSES, MarketAgent
from agents.notebook import Notebook
from data_sources import FROZEN, ROOT

SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "evaluation" / "results"
st.set_page_config(page_title="NBA late-news agent", layout="wide")


@st.cache_resource
def load():
    folder = FROZEN if (FROZEN / "prices.parquet").exists() else SAMPLE
    tables = replay.load_tables(folder)
    players = pd.read_parquet(folder / "players.parquet") if (folder / "players.parquet").exists() else None
    names = dict(zip(players.player_id, players.player_name)) if players is not None else {}
    return folder, tables, names


@st.cache_resource
def forecaster():
    from forecast.api import Forecaster
    return Forecaster.load()


def runs() -> list:
    return sorted({p.parent for p in replay.RUNS.rglob("trace.jsonl")}, key=lambda p: p.stat().st_mtime, reverse=True)


def traces(run: Path) -> list:
    return [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]


def view_at(tables, now):
    rp = st.session_state.setdefault("rp", {})
    if id(tables) not in rp:
        rp[id(tables)] = replay.Replay(tables, lambda *a: []).price_index
    return replay.AsOf(tables, now, rp[id(tables)])


def out_list(news):
    return news.loc[news.status.astype(str).str.lower().isin(OUT_STATUSES), "player_id"].tolist()


def run_night(tables, day, notebook, game_id=None, plant=None, channel="platform", confirm=None):
    kwargs = {"confirm": confirm} if confirm else {}
    agent = MarketAgent(notebook, forecaster=forecaster(), learn=False, plant=plant, channel=channel, **kwargs)
    policy = agent.policy if game_id is None else (lambda v, g, n: agent.policy(v, g, n) if g.game_id == game_id else [])
    decisions, fills = replay.Replay(tables, policy).run(day, day)
    return agent, decisions, fills


folder, tables, names = load()
games = tables["games"]
traded = games[games.game_id.isin(tables["markets"].game_id)]
st.sidebar.title("NBA late-news agent")
st.sidebar.caption(f"data: {folder.relative_to(ROOT)} · {traded.game_id.nunique()} games with markets")
run_dirs = runs()
run = st.sidebar.selectbox("Rule notebook from run", run_dirs, format_func=lambda p: str(p.relative_to(replay.RUNS)),
                           index=0) if run_dirs else None
notebook = Notebook.load(run / "notebook.json") if run and (run / "notebook.json").exists() else Notebook()
day = st.sidebar.selectbox("Night", sorted(traded.date.unique(), reverse=True))
tonight = traded[traded.date == day]
label = {g.game_id: f"{g.away_team} @ {g.home_team}" for g in tonight.itertuples()}
gid = st.sidebar.selectbox("Game", list(label), format_func=label.get)
game = next(g for g in tonight.itertuples() if g.game_id == gid)

night, briefs, learning, safety, models = st.tabs(["Replayed night", "Channel briefs", "Learning", "Safety", "Models"])

with night:
    st.header(f"{label[gid]} · tip {game.tip_time:%H:%M} UTC · final {game.away_pts}-{game.home_pts}")
    final_view = view_at(tables, game.tip_time)
    news = final_view.news(gid)
    st.subheader("News published before tip-off")
    st.dataframe(news[["published_at", "status", "text", "source"]], hide_index=True, width="stretch")
    m = tables["markets"]
    home_m = m[(m.game_id == gid) & (m.kind == "game") & (m.team == game.home_team)]
    if st.button("Run the agent on this night", type="primary"):
        st.session_state["night"] = run_night(tables, day, notebook)
    agent, decisions, fills = st.session_state.get("night", (None, pd.DataFrame(), pd.DataFrame()))
    if len(home_m):
        p = final_view.prices(home_m.market_ticker.iloc[0])
        p = p[p.ts >= pd.Timestamp(game.tip_time) - pd.Timedelta(hours=7)].assign(mid=lambda d: (d.bid + d.ask) / 2)
        chart = alt.Chart(p).mark_line().encode(x=alt.X("ts:T", title="time (UTC)"),
                                                y=alt.Y("mid:Q", title=f"{game.home_team} win price",
                                                        scale=alt.Scale(zero=False)))
        marks = [alt.Chart(pd.DataFrame({"ts": news.published_at})).mark_rule(color="orange").encode(x="ts:T"),
                 alt.Chart(pd.DataFrame({"ts": [game.tip_time]})).mark_rule(color="red").encode(x="ts:T")]
        st.altair_chart(alt.layer(chart, *marks), width="stretch")
        st.caption("orange: news published · red: tip-off (closing price)")
    if agent is not None:
        mine = [t for t in agent.traces if t["game_id"] == gid]
        for t in mine:
            with st.expander(f"decision at {pd.Timestamp(t['as_of']):%H:%M} UTC: {t['status']}", expanded=True):
                st.text(t["brief"])
                st.dataframe(pd.DataFrame(t["trace"]), hide_index=True, width="stretch")
        if len(fills):
            st.subheader("Fills tonight (settled)")
            st.dataframe(fills[["market_ticker", "side", "p_model", "price", "contracts", "fee", "close_price", "clv",
                                "pnl"]].round(3), hide_index=True)

with briefs:
    times = sorted(set(news.published_at) | {game.tip_time - replay.LEAD})
    when = st.select_slider("Decision time", options=times, value=times[-1],
                            format_func=lambda t: f"{pd.Timestamp(t):%H:%M} UTC")
    view = view_at(tables, when)
    now_news = view.news(gid)
    o = outlook(forecaster(), view, game, out_list(now_news))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{game.home_team} win, before news", f"{o['p_home_before']:.0%}")
    c2.metric(f"{game.home_team} win, after news", f"{o['p_home_after']:.0%}",
              f"{(o['p_home_after'] - o['p_home_before']) * 100:+.1f} pts")
    q = view.quote(home_m.market_ticker.iloc[0]) if len(home_m) else None
    c3.metric("Market (bid-ask)", f"{q.bid:.0%}-{q.ask:.0%}" if q is not None else "none")
    b = channel_briefs(game, o, now_news, names, {"bid": q.bid, "ask": q.ask} if q is not None else None)
    cols = st.columns(2)
    for i, ch in enumerate(["platform", "media", "team", "retail"]):
        with cols[i % 2]:
            st.subheader(ch.title())
            st.write(b[ch])
    st.subheader("Player distributions (minutes and points medians; 10-90% minutes range)")
    st.dataframe(b["table"], hide_index=True, width="stretch")

with learning:
    st.subheader("Rule notebook" + (f" ({run.relative_to(replay.RUNS)})" if run else ""))
    if notebook.rules:
        st.dataframe(pd.DataFrame([{"rule": r["rule_id"], "status": r["status"], "when": json.dumps(r["when"]),
                                    "do": json.dumps(r["do"]), "valid_from": r.get("valid_from"),
                                    "gate": r.get("gate", {}).get("reason"), "why": r.get("rationale")}
                                   for r in notebook.rules]), hide_index=True, width="stretch")
    else:
        st.info("This run has no rules yet.")
    if (RESULTS / "headline_clv.png").exists():
        st.image(str(RESULTS / "headline_clv.png"))
    if (RESULTS / "ablations.md").exists():
        st.subheader("Ablations (test period)")
        st.markdown((RESULTS / "ablations.md").read_text())

with safety:
    from evaluation.scorer import policy_tests
    st.subheader("Policy tests: planted orders the risk code must block")
    st.dataframe(policy_tests(), hide_index=True)
    st.subheader("Planted fault on this game")
    fault = st.selectbox("Fault", FAULTS)
    if st.button("Run with the fault"):
        agent, _, f = run_night(tables, day, notebook, gid, plant=fault)
        for t in agent.traces:
            st.write(f"**{pd.Timestamp(t['as_of']):%H:%M} UTC: {t['status']}**")
            st.dataframe(pd.DataFrame(t["trace"]), hide_index=True, width="stretch")
        st.write(f"orders filled: {len(f)}")

with models:
    for name, title in (("calibration.csv", "Win model vs market, every test game"),
                        ("m2_heldout.csv", "Points and minutes: GRU vs baselines (held out)"),
                        ("m3_heldout.csv", "Play classifier (held out)"),
                        ("m4_heldout.csv", "Win model (held out)")):
        if (RESULTS / name).exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(RESULTS / name).round(4), hide_index=True)
