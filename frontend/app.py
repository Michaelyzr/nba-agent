"""E4: Streamlit demo of one replayed night across channels (README section 7).

    streamlit run frontend/app.py

Reads data/frozen/ if it has prices, otherwise the committed data/sample/.
Models come from models/ (python -m forecast.train); runs from runs/.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import replay
from agents import coach
from agents.briefs import channel_briefs, outlook
from agents.graph import FAULTS, OUT_STATUSES, MarketAgent
from agents.notebook import Notebook
from data_sources import FROZEN, ROOT
from nba_agent_paths import RESULTS, SAMPLE

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


def engine(tables) -> replay.Replay:
    rp = st.session_state.setdefault("rp", {})
    if id(tables) not in rp:
        rp[id(tables)] = replay.Replay(tables, lambda *a: [])
    return rp[id(tables)]


def view_at(tables, now):
    return replay.AsOf(tables, now, engine(tables).price_index)


def price_chart(view, ticker, game, news, title):
    p = view.prices(ticker)
    p = p[p.ts >= pd.Timestamp(game.tip_time) - pd.Timedelta(hours=7)].assign(mid=lambda d: (d.bid + d.ask) / 2)
    chart = alt.Chart(p).mark_line().encode(x=alt.X("ts:T", title="time (UTC)"),
                                            y=alt.Y("mid:Q", title=title, scale=alt.Scale(zero=False)))
    marks = [alt.Chart(pd.DataFrame({"ts": news.published_at})).mark_rule(color="orange").encode(x="ts:T")]
    if view.now >= game.tip_time:
        marks.append(alt.Chart(pd.DataFrame({"ts": [game.tip_time]})).mark_rule(color="red").encode(x="ts:T"))
    return alt.layer(chart, *marks)


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

night, pregame, coach_tab, briefs, learning, safety, models = st.tabs(
    ["Replayed night", "Pregame news loop", "Coach: learn the market", "Channel briefs", "Learning", "Safety", "Models"])
decision_times = sorted({t for t in tables["news"].loc[tables["news"].game_id == gid, "published_at"]
                         if game.tip_time - replay.NEWS_WINDOW <= t < game.tip_time} | {game.tip_time - replay.LEAD})
fmt_time = lambda t: f"{pd.Timestamp(t):%H:%M} UTC"

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
        st.altair_chart(price_chart(final_view, home_m.market_ticker.iloc[0], game, news,
                                    f"{game.home_team} win price"), width="stretch")
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

with pregame:
    from uuid import uuid4
    from agents.pregame import PregameAgent, run_loop
    from data_sources.live_news import TableNews
    from data_sources.news_registry import NewsRegistry

    st.header("Pregame news → win probability → fair odds")
    st.caption("Replay the six hours before tip-off at fixed intervals. Historical data sets the initial "
               "probability; player status changes update it. Decimal odds = 1 / probability, without margin.")
    plan = NewsRegistry.load().for_game(game)
    with st.expander("News sources planned for this matchup"):
        source_rows = [{"Source": m["name"], "Role": "Authoritative media", "Team": "Both", "Link": m["url"]}
                       for m in plan["media"]]
        source_rows.append({"Source": "NBA injury report", "Role": "Official report", "Team": "Both",
                            "Link": "https://official.nba.com/nba-injury-report-2025-26-season/"})
        roles = {"team_official": "Team official / PR", "insider": "Insider", "team_reporter": "Team reporter / writer"}
        source_rows.extend({"Source": a["name"] + " (@" + a["handle"] + ")", "Role": roles[a["role"]],
                            "Team": a["team"] or "Both", "Link": a["url"]} for a in plan["x_accounts"])
        st.dataframe(pd.DataFrame(source_rows), hide_index=True,
                     column_config={"Link": st.column_config.LinkColumn("Link")})
        st.caption(f"Account list reviewed {plan['reviewed_at']}. This is the live source plan; "
                   "this demo uses historical replay news and does not query X. "
                   "Official reports and team announcements take priority, then ESPN/CBS, Shams and team reporters.")
    interval = st.selectbox("News check interval (minutes)", [1, 5, 15, 30], index=1)
    model_choice = st.selectbox("Forecast model", ["Trained M4", "Record baseline (demo)"])
    if st.button("Run the pregame loop", type="primary"):
        from agents.graph import record_forecaster
        f = forecaster() if model_choice == "Trained M4" else record_forecaster
        people = pd.DataFrame({"player_id": list(names), "player_name": list(names.values())})
        output = replay.RUNS / f"pregame-ui-{gid}-{uuid4().hex[:8]}"
        agent = PregameAgent(tables, people, game, f, TableNews(tables["news"]), output)
        with st.spinner("Polling replay news and recomputing probabilities..."):
            run_loop(agent, interval * 60, start=game.tip_time - pd.Timedelta(hours=6), replay_mode=True)
        snapshots = [json.loads(line) for line in (output / "snapshots.jsonl").read_text().splitlines()]
        st.session_state[f"pregame|{gid}"] = (snapshots, str(output))
    result = st.session_state.get(f"pregame|{gid}")
    if result:
        snapshots, output = result
        first, last = snapshots[0], snapshots[-1]
        a, b, c = st.columns(3)
        a.metric(f"{game.home_team} win probability", f"{last['p_home']:.1%}",
                 f"{(last['p_home'] - first['p_home']) * 100:+.1f} percentage points")
        b.metric(f"{game.home_team} fair decimal odds", f"{last['home_decimal_odds']:.3f}")
        c.metric(f"{game.away_team} fair decimal odds", f"{last['away_decimal_odds']:.3f}")
        history = pd.DataFrame(snapshots)
        history["as_of"] = pd.to_datetime(history.as_of, utc=True)
        st.line_chart(history.set_index("as_of")[["p_home", "p_away"]])
        st.dataframe(history[["phase", "as_of", "p_home", "p_away", "home_decimal_odds", "away_decimal_odds"]],
                     hide_index=True)
        if last["factors"]:
            st.subheader("Factors applied before tip-off")
            st.dataframe(pd.DataFrame(last["factors"])[["player_id", "status", "published_at", "source", "confirmation", "url", "text"]],
                         hide_index=True)
        if last.get("conflicts"):
            with st.expander("Conflicting reports retained for review", expanded=True):
                for conflict in last["conflicts"]:
                    st.write(f"Selected {conflict['selected_status']} from {conflict['selected_source']}.")
                    st.dataframe(pd.DataFrame(conflict["alternatives"])[["source", "status", "published_at", "url", "text"]],
                                 hide_index=True)
        if last.get("source_coverage"):
            with st.expander("Retrieval coverage"):
                st.dataframe(pd.DataFrame(last["source_coverage"]), hide_index=True)
        evidence_path = Path(output) / "evidence.jsonl"
        if evidence_path.exists():
            with st.expander("Original articles and posts"):
                st.dataframe(pd.read_json(evidence_path, lines=True), hide_index=True)
        st.caption("Questionable/probable/doubtful weights are scenario assumptions pending calibration. "
                   f"Snapshot and source logs: {output}")

with coach_tab:
    st.info(coach.NOTICE)
    c_when = st.select_slider("Decision time (you only see what was public then)", options=decision_times,
                              value=decision_times[0], format_func=fmt_time, key="coach_when")
    c_view = view_at(tables, c_when)
    s = coach.snapshot(forecaster(), c_view, game, names)
    key = f"{gid}|{pd.Timestamp(c_when).isoformat()}"
    paper = st.session_state.setdefault("paper", {})
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What's going on")
        for line in coach.explain(s):
            st.write(line)
        if s["sides"]:
            cols = st.columns(len(s["sides"]))
            for col, side in zip(cols, s["sides"].values()):
                col.metric(f"Back {side['team']}: our estimate vs break-even", f"{side['p']:.0%}",
                           f"{side['gap'] * 100:+.1f} pts vs {side['breakeven']:.1%}")
        if len(home_m):
            shown = view_at(tables, game.tip_time) if key in paper else c_view
            st.altair_chart(price_chart(shown, home_m.market_ticker.iloc[0], game, shown.news(gid),
                                        f"{game.home_team} win price"), width="stretch")
            st.caption("price so far (orange: news). After your call the rest of the night is revealed "
                       "(red: tip-off, the closing price).")
    with right:
        st.subheader("Concepts in this game")
        for i, card in enumerate(coach.lessons(s)):
            with st.expander(card["title"], expanded=i < 2):
                st.write(card["body"])

    st.subheader("Your call")
    if key not in paper:
        if not s["sides"]:
            st.warning("No fresh market price at this time; pick another decision time.")
        else:
            with st.form(f"call-{key}"):
                choice = st.radio("What do you do?", [f"Back {s['away']}", f"Back {s['home']}", "Pass"],
                                  index=2, horizontal=True)
                stake = st.slider("Stake (paper dollars)", 5, 50, 20, 5)
                reason = st.text_input("Why? (one line, optional)")
                if st.form_submit_button("Submit my call", type="primary"):
                    if choice == "Pass":
                        paper[key] = {"filled": False, "why": "pass", "game_id": gid, "as_of": c_when,
                                      "team": None, "pick": s["pick"], "reason": reason}
                    else:
                        side = s["sides"][choice.split()[-1]]
                        paper[key] = {**coach.paper_trade(engine(tables), game, c_when, side, stake, s),
                                      "reason": reason}
                    st.rerun()
    else:
        t = paper[key]
        st.write(f"**Final score: {game.away_team} {game.away_pts:.0f}, {game.home_team} {game.home_pts:.0f}.**")
        if t.get("filled"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"You backed {t['team']} at", f"{t['price']:.0%}", f"{t['contracts']} contracts")
            m2.metric("Closing price", f"{t['close_price']:.1%}",
                      f"{t['clv'] * 100:+.1f} cents closing-line value")
            m3.metric("Result", "won" if t["won"] else "lost")
            m4.metric("Paper P&L after fees", f"${t['pnl']:+.2f}")
        elif t["why"] == "pass":
            st.write("You passed. What backing each team would have done:")
            what_if = [coach.paper_trade(engine(tables), game, c_when, side, 20, s) for side in s["sides"].values()]
            st.dataframe(pd.DataFrame([{"back": w["team"], "price": w.get("price"), "closing price": w.get("close_price"),
                                        "won": w.get("won"), "P&L on $20": w.get("pnl")} for w in what_if]).round(3),
                         hide_index=True)
        else:
            st.warning(f"Order not filled: {t['why']}.")
        runs_key = f"agent|{day}|{gid}"
        if runs_key not in st.session_state:
            with st.spinner("Running the agent on this game..."):
                st.session_state[runs_key] = run_night(tables, day, notebook, gid)
        agent, _, agent_fills = st.session_state[runs_key]
        same = [x for x in agent.traces if pd.Timestamp(x["as_of"]) == pd.Timestamp(c_when)]
        st.write(f"**Coach's model view at this time:** "
                 + (f"back {s['pick']}." if s["pick"] else "pass (no edge after fees)."))
        if same:
            st.write(f"**What the agent did at this time:** {same[0]['status'].replace('_', ' ')}.")
            st.text(same[0]["brief"])
        if len(agent_fills):
            f = agent_fills.iloc[0]
            st.write(f"The agent's position in this game: {f.side.upper()} {f.market_ticker.rsplit('-', 1)[-1]} at "
                     f"{f.price:.0%}, closing price {f.close_price:.1%}, P&L ${f.pnl:+.2f}.")
        else:
            st.write("The agent took no position in this game.")
        if st.button("Try this decision again"):
            del paper[key]
            st.rerun()

    st.subheader("Your scoreboard")
    fb = coach.feedback(list(paper.values()))
    if fb["trades"]:
        a, b, c, d = st.columns(4)
        a.metric("Paper trades", fb["trades"])
        b.metric("P&L after fees", f"${fb['pnl']:+.2f}")
        c.metric("Closing-line value", f"${fb['clv_dollars']:+.2f}")
        d.metric("Long shots", f"{fb['long_shot_share']:.0%}")
        st.dataframe(pd.DataFrame([x for x in paper.values() if x.get("filled")])[
            ["game_id", "as_of", "team", "price", "close_price", "clv", "won", "pnl", "reason"]].round(3),
            hide_index=True, width="stretch")
    for tip in fb["tips"]:
        st.write(f"- {tip}")

with briefs:
    times = decision_times
    when = st.select_slider("Decision time", options=times, value=times[-1], format_func=fmt_time)
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
