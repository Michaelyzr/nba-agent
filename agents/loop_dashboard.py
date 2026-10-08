"""English betting-oriented Streamlit view, matching the portable demo website."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agents.loop_demo import SAMPLE_OUTPUT
from agents.betting import evaluate, recommend_single, recommend_parlay, parlay_result, hedge_bet


def _controller():
    import streamlit as st
    from agents.dashboard_live import LiveDashboard
    return st.cache_resource(LiveDashboard)()


def render_demo(key='news_loop_sample'):
    import streamlit as st
    import pandas as pd
    import altair as alt

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
    def reset():
        st.session_state[key+'_selected'] = []
        st.session_state[key+'_index'] = st.session_state[key+'_seek'] = 0
        st.session_state[key+'_playing'] = True
        st.session_state[key+'_tick'] = time.monotonic()
        st.session_state.pop(key+'_pool',None)
        st.session_state.pop(key+'_game',None)
    source = st.radio('Data', ('Demo', 'Live data'), horizontal=True, key=key+'_source', on_change=reset)
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
        st.caption('Our historical model and retrieved news supply these estimates. Polymarket is a price reference.')
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
