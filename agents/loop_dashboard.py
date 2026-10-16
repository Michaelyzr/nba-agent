"""English betting-oriented Streamlit view, matching the portable demo website."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agents.loop_demo import SAMPLE_OUTPUT
from agents.betting import evaluate, recommend_single, recommend_parlay, parlay_result, hedge_bet
from agents.paper_decision import DAY_CAP, GAME_CAP, ORDER_CAP, trading_agent_decision


def _controller():
    import streamlit as st
    from agents.dashboard_live import LiveDashboard
    return st.cache_resource(LiveDashboard)()


def _remember_paper_prefill(key, current):
    import streamlit as st
    home = next((r for r in current.get('bets') or [] if r.get('kind') == 'moneyline' and r.get('side') == 'home'), None)
    ask = ((home or {}).get('reference_quote') or {}).get('ask')
    probability = None if home is None else home.get('probability')
    if probability is None or ask is None:
        return
    st.session_state[key + '_paper_prefill'] = {
        'match': f"{current.get('away_team', '')} @ {current.get('home_team', '')}",
        'game_id': current.get('game_id'),
        'p': float(probability), 'ask': float(ask),
        'anchor': current.get('anchor_mid', current.get('anchor')),
        'm4_shift': current.get('m4_shift', current.get('news_shift')),
        'live_ask': current.get('live_ask', current.get('kalshi_ask'))}


def _render_trading_tests(key):
    """Archive plus one local paper decision. No order API and no LLM."""
    import streamlit as st
    st.subheader('Trading tests')
    st.caption('Static archive of the published tests, then one paper decision. '
               'This uses the published edge rule, not a new live sceptic call.')
    st.markdown("""
News, or the check about 60 minutes before tip, starts the loop. The agent investigates, then forecasts:
the Kalshi mid from about 24 hours before tip, plus the M4 news shift, clipped to [0.02, 0.98].
The edge after the fee 0.07·p·(1−p) on the contract price must be greater than 4¢. Then checks, an optional
sceptic veto, and the risk caps. The result is a paper fill or a logged pass.

| Test | What it found |
|---|---|
| Decide loop | News or tip−60 → investigate → forecast (Kalshi mid ~24h before tip + M4 news shift, clip [0.02, 0.98]) → edge after fee 0.07·p·(1−p) must be > 4¢ → checks → optional sceptic veto → risk caps → paper fill or logged pass. |
| Sceptic (frozen) | Kept 8 of 182 validated proposals. Dec–Apr CLV +$7.38. 4 trades in Feb–Apr. Later redraws mean about +$3.27. Fleiss κ ≈ 0.15. |
| Walk-forward | Full agent: 157 trades, −$25 after fees (95% CI −$569 to +$564). Without learning: 276 trades, −$407. Raw model: 711 trades, −$2,475. Learning gain +$382; the CI crosses zero. Mean CLV per trade did not improve. Later 87-game holdout: 0 trades. |
| Robustness, labelled after the freeze | Balanced prompt AUC ≈ 0.53 (fail). Truncation 2/174. Learned filter loses versus the frozen +$7.38. Five-vote majority κ ≈ 0.39, but only 2 trades, CLV +$0.83, and 0/8 of the original approvals. LLM probability correlation with the live mid ≈ 1. The M4 anchor baseline was worse and was reverted. The 1–2¢ pre-filter was left off. A perfect filter ceiling is 55 of 182, about +$39.87. |
| Edge scan | Injury timing about −1.4¢, and trade timing about −2¢ after costs. Both are no-go. Player-points markets are the only lead, and they are not fully tested. |
| Conclusion | These tests support gated abstention, not a durable edge on NBA game-winners after fees. |
""")
    st.markdown('**Illustrative archive cards** · not live, and not a real game result')
    cards = st.columns(3)
    cards[0].markdown('**Kept paper trade.** Illustrative inputs: home win probability 0.62, ask 0.50. '
                      'Fee 1.75¢. Edge 10.25¢, above 4¢. Checks pass, the sceptic does not veto, and a $20 paper fill sits inside the $50 order cap.')
    cards[1].markdown('**Sceptic veto.** The edge cleared 4¢ and the checks passed. The sceptic vetoes: "already priced in." Logged pass. No fill.')
    cards[2].markdown('**No-edge pass.** Illustrative inputs: home win probability 0.55, ask 0.54. Edge about −0.74¢, below 4¢. Logged pass before the sceptic. No fill.')
    st.divider()
    st.markdown('**Paper decision by the trading agent**')
    st.caption('Forecast is the Kalshi mid about 24 hours before tip plus the M4 news shift when both are on the match. '
               'Otherwise this says they are missing and uses the form. '
               'edge = p − ask − 0.07·ask·(1−ask). A fill needs that edge, in cents, to be strictly greater than the gap. '
               f'Stake starts at $20 and is clamped by the ${ORDER_CAP:.0f} order, ${GAME_CAP:.0f} game and ${DAY_CAP:.0f} day caps. '
               'The result names the cap that binds. No sceptic, no LLM, and no live order.')
    pre = st.session_state.get(key + '_paper_prefill') or {}
    token = f"{pre.get('match')}|{pre.get('p')}|{pre.get('ask')}"
    if st.session_state.get(key + '_paper_token') != token:
        st.session_state[key + '_paper_token'] = token
        if pre.get('p') is not None:
            st.session_state[key + '_paper_p'] = f"{pre['p']:.4f}"
        if pre.get('ask') is not None:
            st.session_state[key + '_paper_ask'] = f"{pre['ask']:.4f}"
    if pre:
        st.caption(f"Prefilled from {pre['match']} (model home-win probability and reference ask). "
                   'Kalshi 24h anchor and M4 news shift are used only when this match has them. You can edit the fields.')
    else:
        st.caption('No selected match with both a model probability and a reference price. Enter them below, or open Demo or Live data first.')
    gap = st.number_input('Edge gap (cents)', min_value=0.0, value=4.0, step=0.5, key=key + '_paper_gap',
                          help='Minimum edge after the fee.')
    stake = st.number_input('Stake (dollars)', min_value=0.0, value=20.0, step=1.0, key=key + '_paper_stake')
    p_text = st.text_input('Home win probability p (0.02–0.98)', key=key + '_paper_p')
    ask_text = st.text_input('Executable price (ask)', key=key + '_paper_ask')
    spent = st.session_state.setdefault(key + '_paper_spent', {'game': 0.0, 'day': 0.0, 'game_id': None})
    if pre.get('game_id') != spent.get('game_id'):
        spent['game_id'] = pre.get('game_id')
        spent['game'] = 0.0
    if st.button('Run paper decision by the trading agent', key=key + '_paper_run'):
        result = trading_agent_decision(
            p_text.strip() or None, ask_text.strip() or None, gap, stake,
            anchor=pre.get('anchor'), m4_shift=pre.get('m4_shift'), live_ask=pre.get('live_ask'),
            spent_game=spent['game'], spent_day=spent['day'])
        if result['status'] == 'PAPER FILL' and result['stake']:
            spent['game'] += result['stake']
            spent['day'] += result['stake']
        st.session_state[key + '_paper_result'] = result
    result = st.session_state.get(key + '_paper_result')
    if result:
        line = f"{result['status']}. {result['reason']}"
        if result['status'] == 'PAPER FILL':
            st.success(line)
        else:
            st.warning(line)


def render_demo(key='news_loop_sample'):
    import streamlit as st
    import pandas as pd
    import altair as alt

    def reset():
        st.session_state[key+'_selected'] = []
        st.session_state[key+'_index'] = st.session_state[key+'_seek'] = 0
        st.session_state[key+'_playing'] = True
        st.session_state[key+'_tick'] = time.monotonic()
        st.session_state.pop(key+'_pool', None)
        st.session_state.pop(key+'_game', None)
    source = st.radio('Data', ('Demo', 'Live data', 'Trading tests'), horizontal=True, key=key+'_source', on_change=reset)
    if source == 'Trading tests':
        _render_trading_tests(key)
        return

    path = SAMPLE_OUTPUT/'demo.html'
    if not path.exists():
        st.info('Run python -m agents.loop_demo to generate the demo.')
        return
    # demo.html is generated as UTF-8; do not rely on the Windows locale (often GBK).
    html = path.read_text(encoding='utf-8')
    match = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', html, re.S)
    if not match:
        st.error(f'Invalid demo file: missing loop-data in {path}')
        return
    data = json.loads(match[1])
    choices = st.columns(3)
    style = choices[0].radio('Bet slip', ('Single bet', 'Parlay'), horizontal=True, key=key+'_style', on_change=lambda:st.session_state.update({key+'_selected':[]} ))
    stake = choices[1].number_input('Your stake (USD)', min_value=1., max_value=100000., value=20., key=key+'_stake')
    goal_label = choices[2].selectbox('Recommend by', ('Highest estimated profit', 'Highest chance to win'), key=key+'_goal')
    goal = 'profit' if goal_label == 'Highest estimated profit' else 'chance'
    category = st.radio('Bet type', ('All bets', 'Moneyline', 'Spread', 'Total points'), horizontal=True, key=key+'_category')
    kind = {'Moneyline': 'moneyline', 'Spread': 'spread', 'Total points': 'total'}.get(category)
    legs = st.selectbox('Number of picks', (2,3,4), format_func=lambda n: f'{n}-leg parlay', key=key+'_legs') if style == 'Parlay' else 1
    st.session_state.setdefault(key+'_selected', [])
    st.session_state.setdefault(key+'_index', 0)
    st.session_state.setdefault(key+'_playing', True)
    st.session_state.setdefault(key+'_tick',time.monotonic())
    refresh = st.fragment(run_every='5s' if source == 'Live data' else '2s') if hasattr(st, 'fragment') else lambda f:f

    @refresh
    def show():
        if source == 'Demo':
            frames = data['pregame_steps']+data['steps']
            index = min(st.session_state[key+'_index'], len(frames)-1)
            if st.session_state[key+'_playing'] and time.monotonic()-st.session_state.get(key+'_tick', 0) > 1.3:
                index = min(index+1, len(frames)-1)
                st.session_state[key+'_index'] = st.session_state[key+'_seek'] = index
                st.session_state[key+'_tick'] = time.monotonic()
                if index == len(frames)-1: st.session_state[key+'_playing'] = False
            def set_step(i):
                st.session_state[key+'_index'] = st.session_state[key+'_seek'] = i
                st.session_state[key+'_playing'] = False
            def toggle():
                st.session_state[key+'_playing'] = not st.session_state[key+'_playing']
                st.session_state[key+'_tick'] = time.monotonic()
            buttons = st.columns(3)
            buttons[0].button('From the start', key=key+'_reset', on_click=set_step, args=(0,))
            buttons[1].button('Pause' if st.session_state[key+'_playing'] else 'Play', key=key+'_play', on_click=toggle, disabled=not hasattr(st,'fragment'))
            buttons[2].button('Next update', key=key+'_next', on_click=set_step, args=(min(index+1,len(frames)-1),), disabled=index==len(frames)-1)
            st.session_state.setdefault(key+'_seek',index)
            st.slider('Replay progress',0,len(frames)-1,key=key+'_seek',on_change=lambda:set_step(st.session_state[key+'_seek']))
            slate = frames[st.session_state[key+'_index']]['slate']
        else:
            try:
                controller = _controller()
                games = controller.games()
                if not games:
                    st.info('No NBA games available right now. Discovery retries automatically.');return
                game_id = st.selectbox('Choose a game',[g['game_id'] for g in games],format_func=lambda v:next(g['away_team']+' @ '+g['home_team'] for g in games if g['game_id']==v),key=key+'_game')
                wanted = [g for g in games if g['game_id']==game_id]
                if style=='Parlay': wanted += [g for g in games if g['game_id']!=game_id][:11]
                with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
                    slate = list(pool.map(lambda g:controller.betting(g['game_id']),wanted))
            except Exception:
                st.info('Waiting for current model and reference data. Retrying automatically.');return
        if source=='Demo': game_id = st.selectbox('Choose a game',[g['game_id'] for g in slate],format_func=lambda v:next(g['away_team']+' @ '+g['home_team'] for g in slate if g['game_id']==v),key=key+'_game')
        current = next(g for g in slate if g['game_id']==game_id)
        _remember_paper_prefill(key, current)
        phase = 'Pre-game' if current['phase']=='pregame' else 'In-play'
        st.caption(('Final' if current['quote_state']=='final' else phase)+' · auto-updating · phase follows the match automatically')
        all_rows = [r for g in slate for r in g['bets']]
        ids = {r['id'] for r in all_rows}
        selected = [v for v in st.session_state[key+'_selected'] if v in ids]
        if style == 'Single bet': selected = [v for v in selected if next(r for r in all_rows if r['id']==v)['game_id']==game_id]
        pool = st.multiselect('Games for your parlay',[g['game_id'] for g in slate],default=[g['game_id'] for g in slate],format_func=lambda v:next(g['away_team']+' @ '+g['home_team'] for g in slate if g['game_id']==v),key=key+'_pool') if style == 'Parlay' else [game_id]
        score = current['score']
        st.subheader(current['away_team']+' @ '+current['home_team'])
        st.caption((f"{score['away_score']} – {score['home_score']} · " if score else '')+current['clock'])
        st.markdown('**Our estimated win chance**' if phase=='Pre-game' else '**Our live win chance**')
        win = next((r['probability'] for r in current['bets'] if r['kind']=='moneyline' and r['side']=='home'),None)
        forecasts=st.columns(2)
        forecasts[0].metric(current['away_team']+' win chance',f'{1-win:.1%}' if win is not None else '—')
        forecasts[1].metric(current['home_team']+' win chance',f'{win:.1%}' if win is not None else '—')
        model_label = current.get('model') or 'Our model'
        st.caption(f'{model_label} and retrieved news supply these estimates. Polymarket is a price reference.')
        st.markdown('**Live odds comparison**')
        rows = [r for r in current['bets'] if kind is None or r['kind']==kind]
        for market in ('moneyline','spread','total'):
            options = [r for r in rows if r['kind']==market]
            if not options: continue
            st.caption({'moneyline':'Winner · full game','spread':'Point spread · full game','total':'Total points · full game'}[market])
            columns = st.columns(2)
            for i,row in enumerate(options):
                r = evaluate(row,stake)
                with columns[i%2]:
                    if st.button(row['label'],key=key+'_pick_'+row['id'],type='primary' if row['id'] in selected else 'secondary',disabled=row['probability'] is None):
                        selected = [row['id']] if style == 'Single bet' else [v for v in selected if next(x for x in all_rows if x['id']==v)['game_id']!=row['game_id']]+[row['id']]
                        selected = selected[-4:]
                    st.caption(f"Our chance {row['probability']:.1%} · Our odds {row['model_odds']:.2f}" if row['model_odds'] else 'Model data pending')
                    st.caption((f"Polymarket odds {1/row['reference_quote']['ask']:.2f} before fees · " if (row.get('reference_quote') or {}).get('ask') else '')+('Last seen' if row.get('reference_quote') and not r['reference_current'] else row['reference_status']))
        st.markdown('**Odds hedge & bet slip**')
        if current['quote_state']=='final': st.info('Game finished. Betting is closed; replay an earlier update or choose another game.')
        if st.button('Find my best parlay' if style=='Parlay' else 'Find my best single bet',key=key+'_recommend',disabled=current['quote_state']=='final'):
            pool_rows = [r for r in all_rows if r['game_id'] in pool and (kind is None or r['kind']==kind)]
            result = recommend_single(rows,stake,goal) if style=='Single bet' else recommend_parlay(pool_rows,stake,legs,goal)
            pick = result.get('pick') or result.get('parlay')
            if pick: selected = [pick['id']] if style=='Single bet' else [r['id'] for r in pick['picks']]
            st.info(result['reason'])
        st.session_state[key+'_selected'] = selected
        picked = [next(r for r in all_rows if r['id']==v) for v in selected]
        st.markdown('**Your bet slip**')
        for row in picked:
            st.write(row['label']+' · '+row['match'])
        if not picked: st.caption('Tap a betting option to see your recommendation.')
        result = evaluate(picked[0],stake) if style=='Single bet' and picked else parlay_result(picked,stake) if len(picked)>=2 else None
        if result:
            metrics = st.columns(3)
            metrics[0].metric('Our estimated chance',f"{result['probability']:.1%}" if result['probability'] is not None else '—')
            metrics[1].metric('Illustrative return if all win' if style=='Parlay' else 'Return if it wins',f"${result['estimated_return']:.2f}" if result['estimated_return'] is not None else '—')
            metrics[2].metric('Estimated profit on average',f"${result['expected_profit']:.2f}" if result['expected_profit'] is not None else '—')
            st.caption('Different games assumed independent. Confirm an actual combined quote.' if style=='Parlay' else 'Return includes stake. Research model estimates; no bets are placed.')
        curve=current['curve'] if phase=='In-play' else []
        if curve:
            frame = pd.DataFrame(curve)
            st.altair_chart(alt.Chart(frame).mark_line(color='#08775c').encode(x=alt.X('minute:Q',title='Pre-game updates' if phase=='Pre-game' else 'Game minutes · Q1–Q4',scale=alt.Scale(domain=[0,max(1 if phase=='Pre-game' else 48,frame.minute.max())])),y=alt.Y('probability:Q',title='Home win chance',scale=alt.Scale(domain=[0,1])),tooltip=['clock','probability']).properties(height=220),use_container_width=True)
        if current['quote_state']!='final':
            with st.expander('Already placed a bet? Protect it'):
                original_id = st.selectbox('Original bet',[r['id'] for r in current['bets']],format_func=lambda v:next(r['label'] for r in current['bets'] if r['id']==v),key=key+'_heldpick')
                original_stake = st.number_input('Amount bet (USD)',min_value=1.,value=20.,key=key+'_heldstake')
                original_odds = st.number_input('Odds when you placed it',min_value=1.01,value=2.,key=key+'_heldodds')
                if st.button('Compare a hedge',key=key+'_protect'):
                    try:
                        h=hedge_bet(current['bets'],original_id,original_stake,original_odds,stake)
                        st.write(f"{h['label']} · Hedge stake ${h['amount']:.2f} · Lower game-result profit ${h['before']:.2f} → ${h['floor']:.2f}")
                    except ValueError as e: st.info(str(e))
        st.markdown('**Latest match updates**')
        st.caption(('Some feeds are unavailable. ' if current.get('news_health')=='degraded' else '')+'News checked '+current['as_of'])
        for event in reversed(current['events'][-20:]):
            st.write(event['headline'])
            st.caption((event['clock'] or 'Pre-game')+' · '+event['source']+' · '+str(event.get('published_at') or event['as_of']))
            if not event.get('synthetic') and event.get('url','').startswith('https://'): st.link_button('Read source',event['url'])
        if not current['events']: st.write('No new match news retrieved yet. Monitoring continues.')
        st.caption('Demo scores and reference prices are synthetic.' if source=='Demo' else current.get('connection_note') or 'Polymarket is a price reference; our model supplies all probabilities.')
    show()
