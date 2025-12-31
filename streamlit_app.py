import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# --- CONFIGURATION ---
BASE_URL = "https://api.sleeper.app/v1"

# --- API FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_all_nfl_players():
    try:
        data = requests.get("https://api.sleeper.app/v1/players/nfl").json()
        simple_map = {}
        for pid, val in data.items():
            name = f"{val.get('first_name', '')} {val.get('last_name', '')}".strip()
            if not name: name = pid 
            simple_map[pid] = {
                'name': name,
                'position': val.get('position', 'bench')
            }
        return simple_map
    except:
        return {}

@st.cache_data(ttl=3600)
def get_league_details(league_id):
    try:
        return requests.get(f"{BASE_URL}/league/{league_id}").json()
    except:
        return None

@st.cache_data(ttl=3600)
def get_current_managers(league_id):
    try:
        users = requests.get(f"{BASE_URL}/league/{league_id}/users").json()
        return {u['display_name'] for u in users}
    except:
        return set()

@st.cache_data(ttl=3600)
def get_users_and_rosters_snapshot(league_id):
    try:
        users = requests.get(f"{BASE_URL}/league/{league_id}/users").json()
        rosters = requests.get(f"{BASE_URL}/league/{league_id}/rosters").json()
        user_map = {u['user_id']: u['display_name'] for u in users}
        roster_to_name = {}
        roster_snapshot = {}
        ir_counts = {}
        for r in rosters:
            rid = r['roster_id']
            oid = r['owner_id']
            name = user_map.get(oid, "Unknown Manager")
            roster_to_name[rid] = name
            roster_snapshot[name] = r.get('players') or []
            # IR Count
            ir_list = r.get('ir') or []
            clean_ir = [x for x in ir_list if x is not None]
            ir_counts[name] = len(clean_ir)
        return roster_to_name, user_map, roster_snapshot, ir_counts
    except:
        return {}, {}, {}, {}

@st.cache_data(ttl=3600)
def get_playoff_results(league_id):
    try:
        winners = requests.get(f"{BASE_URL}/league/{league_id}/winners_bracket").json()
        losers = requests.get(f"{BASE_URL}/league/{league_id}/losers_bracket").json()
        champ_roster_id, runner_up_id, toilet_roster_id = None, None, None
        if winners:
            max_r = max([m['r'] for m in winners])
            finals = [m for m in winners if m['r'] == max_r]
            if finals:
                champ_roster_id = finals[0].get('w')
                runner_up_id = finals[0].get('l')
        if losers:
            max_r_l = max([m['r'] for m in losers])
            finals_l = [m for m in losers if m['r'] == max_r_l]
            if finals_l: toilet_roster_id = finals_l[0].get('w')
        return champ_roster_id, runner_up_id, toilet_roster_id
    except:
        return None, None, None

@st.cache_data(ttl=3600)
def get_matchups_batch(league_id, weeks):
    all_data = []
    for w in range(1, weeks + 1):
        try:
            data = requests.get(f"{BASE_URL}/league/{league_id}/matchups/{w}").json()
            if data:
                for m in data:
                    m['week'] = w
                    all_data.append(m)
        except:
            pass
    return all_data

@st.cache_data(ttl=3600)
def get_transactions_batch(league_id, weeks):
    all_tx = []
    for w in range(1, weeks + 1):
        try:
            data = requests.get(f"{BASE_URL}/league/{league_id}/transactions/{w}").json()
            if data:
                for t in data:
                    t['week'] = w 
                    all_tx.append(t)
        except:
            pass
    return all_tx

@st.cache_data(ttl=3600)
def get_draft_data(league_id):
    try:
        drafts = requests.get(f"{BASE_URL}/league/{league_id}/drafts").json()
        if not drafts: return []
        draft_id = drafts[0]['draft_id']
        picks = requests.get(f"{BASE_URL}/draft/{draft_id}/picks").json()
        return picks
    except:
        return []

@st.cache_data(ttl=86400)
def get_weekly_stats(season, week):
    try:
        url = f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"
        data = requests.get(url).json()
        points_map = {}
        for pid, stats in data.items():
            pts = stats.get('pts_ppr', stats.get('pts_half_ppr', stats.get('pts_std', 0)))
            if pts is None: pts = 0
            points_map[pid] = pts
        return points_map
    except:
        return {}

def recursive_history(league_id):
    history = []
    curr = league_id
    depth = 0
    with st.spinner("Mining League History..."):
        while curr and depth < 10:
            data = get_league_details(curr)
            if not data: break
            history.append(data)
            curr = data.get('previous_league_id')
            depth += 1
    return history

def calculate_optimal_score(players_points, roster_slots):
    starter_slots = [s for s in roster_slots if s not in ['BN', 'IR', 'TAXI']]
    available = sorted(players_points, key=lambda x: x['points'], reverse=True)
    used_ids = set()
    total_score = 0.0
    flex_map = {'FLEX': ['RB','WR','TE'], 'WRRB_FLEX': ['RB','WR'], 'REC_FLEX': ['WR','TE'], 'SUPER_FLEX': ['QB','RB','WR','TE']}
    sorted_slots = sorted(starter_slots, key=lambda s: 1 if 'FLEX' in s else 0)
    for slot in sorted_slots:
        best_player = None
        for p in available:
            if p['id'] in used_ids: continue
            is_eligible = (slot == p['pos']) or ('FLEX' in slot and p['pos'] in flex_map.get(slot, []))
            if is_eligible:
                best_player = p
                break
        if best_player:
            total_score += best_player['points']
            used_ids.add(best_player['id'])
    return total_score

# --- CORE PROCESSING ---
def process_data(history, player_db, include_playoffs, exclude_zeros):
    league_scores, player_stats, draft_registry, transaction_log = [], [], [], []
    roster_snapshot_list, fatal_errors, season_finishes, ir_stats = [], [], [], []
    weekly_points_map = {}
    starter_set = set()
    
    for league in reversed(history):
        lid = league['league_id']
        season = league['season']
        settings = league.get('settings', {})
        roster_positions = league.get('roster_positions') or [] 
        playoff_start = settings.get('playoff_week_start', 15)
        
        roster_names, _, snapshot, ir_counts = get_users_and_rosters_snapshot(lid)
        
        # IR Tracking
        for mgr, count in ir_counts.items():
            if count > 0: ir_stats.append({'Season': season, 'Manager': mgr, 'IR Count': count})

        champ_roster_id, runner_up_id, toilet_roster_id = get_playoff_results(lid)
        
        for mgr, p_list in snapshot.items():
            for pid in p_list:
                roster_snapshot_list.append({'Season': season, 'Manager': mgr, 'Player ID': pid})
        
        def get_p_info(pid):
            p = player_db.get(pid, {'name': pid, 'position': 'UNK'})
            if not p['name']: p['name'] = pid
            return p

        # 0. PRE-FETCH WEEKLY STATS
        weeks_to_fetch = 18 if int(season) >= 2021 else 16
        for w in range(1, weeks_to_fetch + 1):
            if not include_playoffs and w >= playoff_start: continue
            global_stats = get_weekly_stats(season, w)
            for pid, pts in global_stats.items():
                weekly_points_map[(season, w, pid)] = pts

        # 1. DRAFT
        raw_picks = get_draft_data(lid)
        temp_picks = []
        for p in raw_picks:
            pid = p['player_id']
            p_info = get_p_info(pid)
            if p_info['name'] == pid and 'metadata' in p:
                 meta = p['metadata']
                 p_info['name'] = f"{meta.get('first_name','')} {meta.get('last_name','')}"
                 p_info['position'] = meta.get('position', 'UNK')
            mgr_name = roster_names.get(p.get('roster_id'), "Unknown") 
            if mgr_name == "Unknown": mgr_name = "Manager (Draft)" 
            temp_picks.append({'Season': season, 'Round': p['round'], 'Pick': p['pick_no'], 'Manager': mgr_name, 'Player': p_info['name'], 'Position': p_info['position'], 'Player ID': pid})
        if temp_picks:
            df_t = pd.DataFrame(temp_picks)
            df_t['Draft Pos Rank'] = df_t.groupby(['Season', 'Position'])['Pick'].rank(method='min')
            draft_registry.extend(df_t.to_dict('records'))

        # 2. MATCHUPS
        raw_matchups = get_matchups_batch(lid, weeks_to_fetch)
        for m in raw_matchups:
            rid = m['roster_id']
            if not include_playoffs and m['week'] >= playoff_start: continue
            if exclude_zeros and m['points'] == 0: continue
            
            mgr = roster_names.get(rid, "Unknown")
            starters = m.get('starters') or []
            
            for s_id in starters:
                if s_id != '0': starter_set.add((season, m['week'], mgr, s_id))

            players_points_map = m.get('players_points') or {}
            roster_players, starter_objs, bench_objs = [], [], []
            for pid, pts in players_points_map.items():
                p_info = get_p_info(pid)
                roster_players.append({'id': pid, 'pos': p_info['position'], 'points': pts})
                status = 'Starter' if pid in starters else 'Bench'
                player_stats.append({'Season': season, 'Manager': mgr, 'Player': p_info['name'], 'Position': p_info['position'], 'Points': pts, 'Player ID': pid, 'Week': m['week'], 'Status': status})
                if status == 'Starter': starter_objs.append({'name': p_info['name'], 'pos': p_info['position'], 'pts': pts})
                else: bench_objs.append({'name': p_info['name'], 'pos': p_info['position'], 'pts': pts})

            optimal_points = calculate_optimal_score(roster_players, roster_positions)
            league_scores.append({'Season': season, 'Week': m['week'], 'Matchup ID': m['matchup_id'], 'Manager': mgr, 'Points': m['points'], 'Max Points': optimal_points, 'Efficiency': (m['points'] / optimal_points * 100) if optimal_points > 0 else 0, 'Roster ID': rid, 'Starters': starter_objs, 'Bench': bench_objs})

        # 3. TRANSACTIONS
        raw_tx = get_transactions_batch(lid, weeks_to_fetch)
        for tx in raw_tx:
            week = tx['week']
            adds = tx.get('adds') or {}
            drops = tx.get('drops') or {}
            involved_rosters = tx.get('roster_ids') or []
            t_type = tx['type']
            for rid in involved_rosters:
                mgr = roster_names.get(rid, "Unknown")
                my_adds = [pid for pid, r in adds.items() if r == rid]
                my_drops = [pid for pid, r in drops.items() if r == rid]
                if len(my_adds) == 1 and len(my_drops) == 1:
                    add_pid, drop_pid = my_adds[0], my_drops[0]
                    p_add, p_drop = get_p_info(add_pid), get_p_info(drop_pid)
                    if p_add['position'] == p_drop['position']:
                        transaction_log.append({'Season': season, 'Week': week, 'Manager': mgr, 'Type': 'Swap', 'Added': p_add['name'], 'Dropped': p_drop['name'], 'Position': p_add['position'], 'Added ID': add_pid, 'Dropped ID': drop_pid})
                clean_type = "Trade" if t_type == 'trade' else ("Waiver" if t_type == 'waiver' else "Free Agent")
                for p in my_adds: transaction_log.append({'Season': season, 'Week': week, 'Manager': mgr, 'Type': clean_type, 'Action': 'Add', 'Added ID': p})
                for p in my_drops: transaction_log.append({'Season': season, 'Week': week, 'Manager': mgr, 'Type': clean_type, 'Action': 'Drop', 'Dropped ID': p})
        
        # 4. SEASON SUMMARY
        for rid, mgr in roster_names.items():
            outcome = "Contender"
            if rid == champ_roster_id: outcome = "Champion"
            elif rid == runner_up_id: outcome = "Runner Up"
            elif rid == toilet_roster_id: outcome = "Toilet Bowl Champ"
            season_finishes.append({'Season': season, 'Manager': mgr, 'Outcome': outcome, 'Roster ID': rid})

    df_scores = pd.DataFrame(league_scores)
    df_finishes = pd.DataFrame(season_finishes)
    df_ir = pd.DataFrame(ir_stats)
    
    if not df_scores.empty:
        df_opp = df_scores[['Season', 'Week', 'Matchup ID', 'Points', 'Manager']].copy()
        df_opp.columns = ['Season', 'Week', 'Matchup ID', 'Opponent Points', 'Opponent']
        df_final = pd.merge(df_scores, df_opp, on=['Season', 'Week', 'Matchup ID'])
        df_final = df_final[df_final['Manager'] != df_final['Opponent']]
        df_final['Result'] = df_final.apply(lambda x: 'Win' if x['Points'] > x['Opponent Points'] else ('Loss' if x['Points'] < x['Opponent Points'] else 'Tie'), axis=1)
        df_final['Margin'] = df_final['Points'] - df_final['Opponent Points']
        
        # CALC RANKS
        season_wins = df_final[df_final['Week'] < 15].groupby(['Season', 'Manager'])['Result'].apply(lambda x: (x == 'Win').sum()).reset_index(name='Wins')
        season_pts = df_final[df_final['Week'] < 15].groupby(['Season', 'Manager'])['Points'].sum().reset_index(name='RegPts')
        df_finishes = pd.merge(df_finishes, season_wins, on=['Season', 'Manager'], how='left').fillna(0)
        df_finishes = pd.merge(df_finishes, season_pts, on=['Season', 'Manager'], how='left').fillna(0)
        df_finishes['Rank'] = df_finishes.groupby('Season')[['Wins', 'RegPts']].rank(method='min', ascending=False)['Wins']

        # FIX: Override Rank for Champ/RunnerUp so "Avg Finish" is accurate
        def fix_rank(row):
            if row['Outcome'] == 'Champion': return 1
            if row['Outcome'] == 'Runner Up': return 2
            return row['Rank']
        df_finishes['Rank'] = df_finishes.apply(fix_rank, axis=1)

        # FATAL ERRORS LOGIC (Strict Mode V4)
        losses = df_final[df_final['Result'] == 'Loss']
        
        # 1. Bench Errors
        for _, row in losses.iterrows():
            margin = abs(row['Margin'])
            starters, bench = row['Starters'], row['Bench']
            s_by_pos = {}
            for s in starters: s_by_pos.setdefault(s['pos'], []).append(s)
            b_by_pos = {}
            for b in bench: b_by_pos.setdefault(b['pos'], []).append(b)
            for pos, s_list in s_by_pos.items():
                if pos in b_by_pos:
                    worst_starter = min(s_list, key=lambda x: x['pts'])
                    best_bench = max(b_by_pos[pos], key=lambda x: x['pts'])
                    diff = best_bench['pts'] - worst_starter['pts']
                    if diff >= 15 and diff > margin:
                        fatal_errors.append({'Season': row['Season'], 'Week': row['Week'], 'Manager': row['Manager'], 'Opponent': row['Opponent'], 'Margin': margin, 'Mistake': f"Start/Sit: Started {worst_starter['name']} ({worst_starter['pts']}) over {best_bench['name']} ({best_bench['pts']})", 'Points Lost': diff})
                        break 
        
        # 2. Transaction Errors
        for _, row in losses.iterrows():
            margin = abs(row['Margin'])
            relevant_tx = [t for t in transaction_log if t['Manager'] == row['Manager'] and t['Week'] == row['Week'] and t['Season'] == row['Season'] and t.get('Type') == 'Swap']
            for t in relevant_tx:
                if (row['Season'], row['Week'], row['Manager'], t['Added ID']) in starter_set:
                    add_pts = weekly_points_map.get((row['Season'], row['Week'], t['Added ID']), 0)
                    drop_pts = weekly_points_map.get((row['Season'], row['Week'], t['Dropped ID']), 0)
                    diff = drop_pts - add_pts
                    if diff >= 15 and diff > margin:
                         fatal_errors.append({'Season': row['Season'], 'Week': row['Week'], 'Manager': row['Manager'], 'Opponent': row['Opponent'], 'Margin': margin, 'Mistake': f"Bad Drop: Dropped {t['Dropped']} ({drop_pts}) for {t['Added']} ({add_pts})", 'Points Lost': diff})

        return df_final, pd.DataFrame(player_stats), pd.DataFrame(draft_registry), pd.DataFrame(transaction_log), pd.DataFrame(roster_snapshot_list), pd.DataFrame(fatal_errors), weekly_points_map, df_finishes, starter_set, df_ir
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), set(), pd.DataFrame()

# --- APP UI ---
st.set_page_config(page_title="Sleeper Analytics", layout="wide", page_icon="🏆")
st.title("🏆 Ultimate Sleeper Dashboard")

with st.sidebar:
    league_id_input = st.text_input("League ID", placeholder="e.g. 10483...")
    include_playoffs = st.checkbox("Include Playoffs", False)
    exclude_zeros = st.checkbox("Exclude 0-Point Weeks", True)
    if not league_id_input: st.stop()

with st.spinner("Initializing Database..."):
    player_db = get_all_nfl_players()

current_managers = get_current_managers(league_id_input)
history = recursive_history(league_id_input)
if not history: st.stop()

df_scores, df_players, df_drafts, df_tx, df_rosters, df_errors, weekly_map, df_finishes, starter_set, df_ir = process_data(history, player_db, include_playoffs, exclude_zeros)
if df_scores.empty: st.warning("No data found."); st.stop()

df_scores_active = df_scores[df_scores['Manager'].isin(current_managers)]
df_errors_active = df_errors[df_errors['Manager'].isin(current_managers)] if not df_errors.empty else pd.DataFrame()
df_finishes_active = df_finishes[df_finishes['Manager'].isin(current_managers)]
df_players_active = df_players[df_players['Manager'].isin(current_managers)]

seasons_list = sorted(df_scores['Season'].unique(), reverse=True)
def filter_by_season(df, season_selection):
    if season_selection == "All Time": return df
    return df[df['Season'] == season_selection]

# TABS
tab_roast, tab_mgr, tab_pos, tab_trophy = st.tabs(["🔥 The Roast", "👤 Manager Deep Dive", "🏈 Position Lab", "🏆 Trophy Room"])

# --- TAB 1: THE ROAST ---
with tab_roast:
    st.header("The Roast Room 🤡 (Active Only)")
    r_season = st.selectbox("Season", ["All Time"] + seasons_list, key="roast_sel")
    r_df = filter_by_season(df_scores_active, r_season)
    
    col_fraud, col_fatal = st.columns([2, 1])
    with col_fraud:
        st.subheader("The Fraud Quadrant")
        st.caption("PF vs PA (Wins determines bubble size)")
        agg = r_df.groupby('Manager').agg({'Points': 'mean', 'Opponent Points': 'mean', 'Result': lambda x: (x == 'Win').sum()}).reset_index()
        agg.columns = ['Manager', 'Avg PF', 'Avg PA', 'Wins']
        
        if not agg.empty:
            avg_pf, avg_pa = agg['Avg PF'].mean(), agg['Avg PA'].mean()
            fig = px.scatter(agg, x='Avg PF', y='Avg PA', text='Manager', size='Wins', color='Wins', color_continuous_scale='Viridis')
            fig.add_hline(y=avg_pa, line_dash="dash"); fig.add_vline(x=avg_pf, line_dash="dash")
            fig.add_annotation(x=agg['Avg PF'].max(), y=agg['Avg PA'].min(), text="GOD SQUAD", showarrow=False)
            fig.add_annotation(x=agg['Avg PF'].min(), y=agg['Avg PA'].min(), text="FRAUDS", showarrow=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_fatal:
        st.subheader("💀 Fatal Errors")
        r_errors = filter_by_season(df_errors_active, r_season)
        if r_errors.empty: st.success("No fatal errors found."); 
        else:
            st.dataframe(r_errors['Manager'].value_counts().reset_index(name='Errors'), hide_index=True)
            for _, row in r_errors.sort_values('Points Lost', ascending=False).head(3).iterrows():
                st.error(f"**{row['Manager']}** (Wk {row['Week']} {row['Season']}): Lost by {row['Margin']:.1f}. {row['Mistake']}")

    st.divider()
    st.subheader("⚔️ The Nemesis Matrix")
    matrix = r_df[r_df['Result'] == 'Win'].groupby(['Manager', 'Opponent']).size().unstack(fill_value=0)
    st.plotly_chart(px.imshow(matrix, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

# --- TAB 2: MANAGER DEEP DIVE ---
with tab_mgr:
    st.header("Manager Analysis")
    mc1, mc2 = st.columns(2)
    valid_managers = sorted(list(current_managers.intersection(set(df_scores['Manager'].unique()))))
    with mc1: selected_mgr = st.selectbox("Select Manager", valid_managers)
    with mc2: mgr_season = st.selectbox("Filter Season", ["All Time"] + seasons_list, key="mgr_sel")
    
    m_df = filter_by_season(df_scores, mgr_season)
    mgr_games = m_df[m_df['Manager'] == selected_mgr]
    mgr_fin = filter_by_season(df_finishes, mgr_season)
    mgr_fin = mgr_fin[mgr_fin['Manager'] == selected_mgr]
    
    if not mgr_games.empty:
        high_s = mgr_games['Points'].max()
        wins = len(mgr_games[mgr_games['Result'] == 'Win'])
        eff_pct = (mgr_games['Points'].sum() / mgr_games['Max Points'].sum() * 100)
        avg_pts = mgr_games['Points'].mean()
        
        if mgr_season == "All Time":
            avg_finish = df_finishes[df_finishes['Manager'] == selected_mgr]['Rank'].mean()
            finish_str = f"Avg: {avg_finish:.1f}"
        else:
            outcome = mgr_fin.iloc[0]['Outcome']
            rank_reg = mgr_fin.iloc[0]['Rank']
            finish_str = f"{outcome} 👑" if outcome == "Champion" else (outcome if outcome in ['Runner Up', 'Toilet Bowl Champ'] else f"Rank {rank_reg:.0f}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{(wins/len(mgr_games)*100):.1f}%", f"{wins} Wins")
        c2.metric("Avg Score", f"{avg_pts:.1f}")
        c3.metric("Efficiency", f"{eff_pct:.1f}%")
        c4.metric("League Finish", finish_str)
        
        st.divider()
        st.subheader("💀 Fatal Error Audit")
        my_errors = filter_by_season(df_errors, mgr_season)
        my_errors = my_errors[my_errors['Manager'] == selected_mgr]
        if not my_errors.empty:
            st.dataframe(my_errors[['Season', 'Week', 'Opponent', 'Margin', 'Mistake', 'Points Lost']], hide_index=True, use_container_width=True)
        else: st.success("Clean sheet. No errors.")

        st.subheader("📊 Performance")
        if mgr_season == "All Time":
            pf_pa = mgr_games.groupby('Season')[['Points', 'Opponent Points']].mean().reset_index()
            fig_perf = px.bar(pf_pa, x='Season', y=['Points', 'Opponent Points'], barmode='group', title="Yearly Avg Performance")
        else:
            mgr_games_sorted = mgr_games.sort_values('Week')
            fig_perf = px.bar(mgr_games_sorted, x='Week', y=['Points', 'Opponent Points'], barmode='group', title=f"Weekly Performance ({mgr_season})")
        st.plotly_chart(fig_perf, use_container_width=True)

        st.divider()
        sub1, sub2, sub3 = st.tabs(["Draft", "Streamer Grades", "Sim"])
        with sub1:
            mgr_drafts = filter_by_season(df_drafts, mgr_season)
            mgr_drafts = mgr_drafts[mgr_drafts['Manager'] == selected_mgr]
            mgr_roster_data = filter_by_season(df_rosters, mgr_season)
            mgr_roster_ids = set(mgr_roster_data[mgr_roster_data['Manager'] == selected_mgr]['Player ID']) if not mgr_roster_data.empty else set()
            if not mgr_drafts.empty and mgr_season != "All Time":
                board_data = [{'Round': r['Round'], 'Pick': r['Pick'], 'Player': r['Player'], 'Status': "🟢 On Roster" if r['Player ID'] in mgr_roster_ids else "🔴 Gone"} for _, r in mgr_drafts.iterrows()]
                st.dataframe(pd.DataFrame(board_data), hide_index=True, use_container_width=True)
            else: st.info("Select specific season.")
        with sub2:
            mgr_swaps = filter_by_season(df_tx, mgr_season)
            mgr_swaps = mgr_swaps[(mgr_swaps['Manager'] == selected_mgr) & (mgr_swaps['Type'] == 'Swap')]
            if not mgr_swaps.empty:
                grades = []
                for _, row in mgr_swaps.iterrows():
                    if (row['Season'], row['Week'], row['Added ID']) not in weekly_map: continue
                    p_add = weekly_map.get((row['Season'], row['Week'], row['Added ID']), 0)
                    p_drop = weekly_map.get((row['Season'], row['Week'], row['Dropped ID']), 0)
                    grades.append({'Week': row['Week'], 'Added': f"{row['Added']} ({p_add})", 'Dropped': f"{row['Dropped']} ({p_drop})", 'Net': p_add - p_drop})
                st.dataframe(pd.DataFrame(grades).sort_values('Net', ascending=False), hide_index=True)
            else: st.info("No swaps found.")
        with sub3:
            if mgr_season != "All Time":
                s_data = df_scores[df_scores['Season'] == mgr_season]
                my_pts = s_data[s_data['Manager'] == selected_mgr].set_index('Week')['Points'].to_dict()
                sim_res = []
                for opp in s_data['Manager'].unique():
                    if opp == selected_mgr: continue
                    opp_sch = s_data[s_data['Manager'] == opp]
                    w, l = 0, 0
                    for _, r in opp_sch.iterrows():
                        if r['Week'] in my_pts:
                            if my_pts[r['Week']] > r['Opponent Points']: w += 1
                            else: l += 1
                    sim_res.append({'Schedule': f"{opp}", 'Record': f"{w}-{l}", 'Win %': w/(w+l) if w+l>0 else 0})
                st.dataframe(pd.DataFrame(sim_res).sort_values('Win %', ascending=False), hide_index=True)
            else: st.info("Select specific season.")

# --- TAB 3: POSITION LAB ---
with tab_pos:
    st.header("Positional Analysis (Active Only)")
    p_season = st.selectbox("Season", ["All Time"] + seasons_list, key="pos_sel")
    if not df_players_active.empty:
        p_df = filter_by_season(df_players_active, p_season)
        pos = st.radio("Position", ["QB", "RB", "WR", "TE", "K", "DEF"], horizontal=True)
        grouped = p_df[p_df['Position'] == pos].groupby('Manager')['Points'].agg(['sum', 'count']).reset_index()
        grouped.columns = ['Manager', 'Total Pts', 'Starts']
        grouped['Avg Pts per Start'] = (grouped['Total Pts'] / grouped['Starts']).round(2)
        st.dataframe(grouped.sort_values('Total Pts', ascending=False), hide_index=True, use_container_width=True)

# --- TAB 4: TROPHY ROOM ---
with tab_trophy:
    st.header("🏆 The Trophy Room (Active Only)")
    t_season = st.selectbox("Season", ["All Time"] + seasons_list, key="trophy_sel")
    t_scores = filter_by_season(df_scores_active, t_season)
    t_fin = filter_by_season(df_finishes_active, t_season)
    t_ply = filter_by_season(df_players_active, t_season)
    
    if not t_scores.empty:
        champs = t_fin[t_fin['Outcome'] == 'Champion'].groupby('Manager')['Season'].apply(list).reset_index()
        champs['Count'] = champs['Season'].apply(len)
        champs['Years'] = champs['Season'].apply(lambda x: ", ".join(map(str, x)))
        
        toilets = t_fin[t_fin['Outcome'] == 'Toilet Bowl Champ'].groupby('Manager')['Season'].apply(list).reset_index()
        toilets['Count'] = toilets['Season'].apply(len)
        toilets['Years'] = toilets['Season'].apply(lambda x: ", ".join(map(str, x)))
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("👑 The Kings")
            if not champs.empty: st.dataframe(champs[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
            else: st.info("No Active Champions")
        with c2:
            st.subheader("🚽 The Plungers")
            if not toilets.empty: st.dataframe(toilets[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
            else: st.info("No Active Toilet Bowl Champs")
        with c3:
            st.subheader("🏥 Hospital Ward")
            if not df_ir.empty:
                ir_disp = filter_by_season(df_ir[df_ir['Manager'].isin(current_managers)], t_season)
                ir_sum = ir_disp.groupby('Manager')['IR Count'].sum().reset_index().sort_values('IR Count', ascending=False)
                st.dataframe(ir_sum, hide_index=True, use_container_width=True)
            else: st.info("No IR data.")
            
        st.divider()
        st.subheader("Positional Awards (Avg Pts/Start)")
        pos_grp = t_ply.groupby(['Manager', 'Position'])['Points'].mean().reset_index()
        cols = st.columns(3)
        for i, p in enumerate(['QB', 'RB', 'WR', 'TE', 'K', 'DEF']):
            with cols[i % 3]:
                p_data = pos_grp[pos_grp['Position'] == p]
                if not p_data.empty:
                    best = p_data.loc[p_data['Points'].idxmax()]
                    worst = p_data.loc[p_data['Points'].idxmin()]
                    st.success(f"**{p} King**\n\n{best['Manager']} ({best['Points']:.1f})")
                    st.error(f"**{p} Jester**\n\n{worst['Manager']} ({worst['Points']:.1f})")
        
        st.divider()
        st.subheader("Records of Shame & Glory")
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.markdown("#### 🥚 Goose Egg Leaders")
            zeros = t_ply[(t_ply['Points'] == 0) & (t_ply['Status'] == 'Starter')]
            if not zeros.empty:
                counts = zeros['Manager'].value_counts()
                for mgr in counts.index:
                    mgr_zeros = zeros[zeros['Manager'] == mgr]
                    with st.expander(f"{mgr} ({len(mgr_zeros)})"):
                        st.dataframe(mgr_zeros[['Season', 'Week', 'Player']], hide_index=True)
            else: st.info("None")
        with col_rec2:
            st.markdown("#### 🧊 Sub-Zero Starters")
            negs = t_ply[(t_ply['Points'] < 0) & (t_ply['Status'] == 'Starter')]
            if not negs.empty:
                counts = negs['Manager'].value_counts()
                for mgr in counts.index:
                    mgr_negs = negs[negs['Manager'] == mgr]
                    with st.expander(f"{mgr} ({len(mgr_negs)})"):
                        st.dataframe(mgr_negs[['Season', 'Week', 'Player', 'Points']], hide_index=True)
            else: st.info("None")
