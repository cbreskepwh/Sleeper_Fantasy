import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# --- CONFIGURATION ---
BASE_URL = "https://api.sleeper.app/v1"

# --- MANAGER MAPPINGS (MERGE TEAMS) ---
MANAGER_MAPPINGS = {
     "CrazyVagisil": "Tariff Boys",
     "beau32": "Tariff Boys"
}

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
        managers = set()
        for u in users:
            name = u['display_name']
            if name in MANAGER_MAPPINGS:
                managers.add(MANAGER_MAPPINGS[name])
            else:
                managers.add(name)
        return managers
    except:
        return set()

@st.cache_data(ttl=3600)
def get_users_and_rosters_snapshot(league_id):
    try:
        users = requests.get(f"{BASE_URL}/league/{league_id}/users").json()
        rosters = requests.get(f"{BASE_URL}/league/{league_id}/rosters").json()
        
        user_map = {}
        for u in users:
            raw_name = u['display_name']
            final_name = MANAGER_MAPPINGS.get(raw_name, raw_name)
            user_map[u['user_id']] = final_name

        roster_to_name = {}
        roster_snapshot = {}
        for r in rosters:
            rid = r['roster_id']
            oid = r['owner_id']
            name = user_map.get(oid, "Unknown Manager")
            roster_to_name[rid] = name
            roster_snapshot[name] = r.get('players') or []
            
        return roster_to_name, user_map, roster_snapshot, len(rosters)
    except:
        return {}, {}, {}, 0

@st.cache_data(ttl=3600)
def get_playoff_outcomes_booleans(league_id):
    try:
        winners = requests.get(f"{BASE_URL}/league/{league_id}/winners_bracket").json()
        losers = requests.get(f"{BASE_URL}/league/{league_id}/losers_bracket").json()
        champ_id, sacko_id = None, None
        
        if winners:
            max_r = max([m['r'] for m in winners])
            finals = [m for m in winners if m['r'] == max_r]
            if finals: champ_id = finals[0].get('w') 
        
        if losers:
            max_r_l = max([m['r'] for m in losers])
            finals_l = [m for m in losers if m['r'] == max_r_l]
            if finals_l: sacko_id = finals_l[0].get('w')
                
        return champ_id, sacko_id
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_playoff_ranks_granular(league_id):
    try:
        winners = requests.get(f"{BASE_URL}/league/{league_id}/winners_bracket").json()
        losers = requests.get(f"{BASE_URL}/league/{league_id}/losers_bracket").json()
        ranks = {}
        
        def parse_bracket(bracket):
            if not bracket: return
            for m in bracket:
                winner = m.get('w')
                loser = m.get('l')
                place = m.get('p') 
                if winner and loser and place:
                    ranks[winner] = place
                    ranks[loser] = place + 1
        
        parse_bracket(winners)
        parse_bracket(losers)
        
        if winners:
            max_r = max([m['r'] for m in winners])
            finals = [m for m in winners if m['r'] == max_r]
            if finals:
                f = finals[0]
                w, l = f.get('w'), f.get('l')
                if w and l:
                    if w not in ranks: ranks[w] = 1
                    if l not in ranks: ranks[l] = 2     
        return ranks
    except:
        return {}

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
                    if t.get('status') == 'failed': continue
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
    seen_ids = set() 
    with st.spinner("Mining League History..."):
        while curr and depth < 10:
            if curr in seen_ids: break
            data = get_league_details(curr)
            if not data: break
            seen_ids.add(curr)
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

def assign_streamer_grade(net_points):
    if net_points >= 20: return "A+"
    if net_points >= 10: return "A"
    if net_points > 0: return "B"
    if net_points > -5: return "C"
    if net_points > -15: return "D"
    return "F"

# --- CORE PROCESSING ---
def process_data(history, player_db):
    league_scores, player_stats, draft_registry, transaction_log = [], [], [], []
    roster_snapshot_list, fatal_errors, season_finishes = [], [], []
    weekly_points_map = {}
    starter_set = set()
    
    include_playoffs = False 
    
    for league in reversed(history):
        lid = league['league_id']
        season = league['season']
        settings = league.get('settings', {})
        roster_positions = league.get('roster_positions') or [] 
        playoff_start = settings.get('playoff_week_start', 15)
        
        roster_names, _, snapshot, league_size = get_users_and_rosters_snapshot(lid)
        champ_id, sacko_id = get_playoff_outcomes_booleans(lid)
        granular_ranks = get_playoff_ranks_granular(lid)
        
        for mgr, p_list in snapshot.items():
            for pid in p_list:
                roster_snapshot_list.append({'Season': season, 'Manager': mgr, 'Player ID': pid})
        
        def get_p_info(pid):
            p = player_db.get(pid, {'name': pid, 'position': 'UNK'})
            if not p['name']: p['name'] = pid
            return p

        weeks_to_fetch = 18 if int(season) >= 2021 else 16
        for w in range(1, weeks_to_fetch + 1):
            if not include_playoffs and w >= playoff_start: continue
            global_stats = get_weekly_stats(season, w)
            for pid, pts in global_stats.items():
                weekly_points_map[(season, w, pid)] = pts

        # DRAFT
        raw_picks = get_draft_data(lid)
        temp_picks = []
        for p in raw_picks:
            pid = p['player_id']
            p_info = get_p_info(pid)
            if p_info['name'] == pid and 'metadata' in p:
                 meta = p['metadata']
                 p_info['name'] = f"{meta.get('first_name','')} {meta.get('last_name','')}"
                 p_info['position'] = meta.get('position', 'UNK')
            
            # KEEPER CHECK
            is_keeper = False
            meta = p.get('metadata', {})
            if 'is_keeper' in meta:
                val = str(meta['is_keeper']).lower()
                is_keeper = (val == 'true')
            elif 'keeper' in meta:
                val = str(meta['keeper']).lower()
                is_keeper = (val == '1' or val == 'true')
            
            mgr_name = roster_names.get(p.get('roster_id'), "Unknown") 
            if mgr_name == "Unknown": mgr_name = "Manager (Draft)" 
            temp_picks.append({
                'Season': season, 'Round': p['round'], 'Pick': p['pick_no'], 
                'Manager': mgr_name, 'Player': p_info['name'], 
                'Position': p_info['position'], 'Player ID': pid,
                'Is Keeper': is_keeper
            })
        if temp_picks:
            df_t = pd.DataFrame(temp_picks)
            df_t['Draft Pos Rank'] = df_t.groupby(['Season', 'Position'])['Pick'].rank(method='min')
            draft_registry.extend(df_t.to_dict('records'))

        # MATCHUPS
        raw_matchups = get_matchups_batch(lid, weeks_to_fetch)
        for m in raw_matchups:
            rid = m['roster_id']
            if not include_playoffs and m['week'] >= playoff_start: continue
            if m['points'] == 0: continue
            
            mgr = roster_names.get(rid, "Unknown")
            starters = m.get('starters') or []
            
            for s_id in starters:
                if s_id != '0': starter_set.add((season, m['week'], mgr, s_id))

            players_points_map = m.get('players_points') or {}
            roster_players, starter_objs, bench_objs = [], [], []
            bench_pts_sum = 0
            
            for pid, pts in players_points_map.items():
                p_info = get_p_info(pid)
                roster_players.append({'id': pid, 'pos': p_info['position'], 'points': pts})
                status = 'Starter' if pid in starters else 'Bench'
                player_stats.append({'Season': season, 'Manager': mgr, 'Player': p_info['name'], 'Position': p_info['position'], 'Points': pts, 'Player ID': pid, 'Week': m['week'], 'Status': status, 'Roster ID': rid})
                
                if status == 'Starter': 
                    starter_objs.append({'name': p_info['name'], 'pos': p_info['position'], 'pts': pts})
                else: 
                    bench_objs.append({'name': p_info['name'], 'pos': p_info['position'], 'pts': pts})
                    bench_pts_sum += pts

            optimal_points = calculate_optimal_score(roster_players, roster_positions)
            league_scores.append({
                'Season': season, 'Week': m['week'], 'Matchup ID': m['matchup_id'], 
                'Manager': mgr, 'Points': m['points'], 'Max Points': optimal_points, 
                'Bench Points': bench_pts_sum,
                'Efficiency': (m['points'] / optimal_points * 100) if optimal_points > 0 else 0, 
                'Roster ID': rid, 'Starters': starter_objs, 'Bench': bench_objs
            })

        # TRANSACTIONS
        raw_tx = get_transactions_batch(lid, weeks_to_fetch)
        for tx in raw_tx:
            week = tx['week']
            tx_id = tx['transaction_id'] 
            settings_safe = tx.get('settings') or {} 
            bid_amount = settings_safe.get('waiver_bid', 0)
            
            adds = tx.get('adds') or {}
            drops = tx.get('drops') or {}
            involved_rosters = tx.get('roster_ids') or []
            t_type = tx['type']
            for rid in involved_rosters:
                mgr = roster_names.get(rid, "Unknown")
                my_adds = [pid for pid, r in adds.items() if str(r) == str(rid)]
                my_drops = [pid for pid, r in drops.items() if str(r) == str(rid)]
                
                clean_type = "Trade" if t_type == 'trade' else ("Waiver" if t_type == 'waiver' else "Free Agent")
                
                for p in my_adds: 
                    transaction_log.append({'Season': season, 'Week': week, 'Manager': mgr, 'Type': clean_type, 'Action': 'Add', 'Added ID': p, 'Transaction ID': tx_id, 'Bid': bid_amount})
                for p in my_drops: 
                    transaction_log.append({'Season': season, 'Week': week, 'Manager': mgr, 'Type': clean_type, 'Action': 'Drop', 'Dropped ID': p, 'Transaction ID': tx_id, 'Bid': 0})
        
        # SEASON SUMMARY
        for rid, mgr in roster_names.items():
            season_finishes.append({
                'Season': season, 'Manager': mgr, 'Roster ID': rid,
                'is_league_champ': (rid == champ_id),
                'is_sacko': (rid == sacko_id),
                'playoff_rank_precise': granular_ranks.get(rid) 
            })

    df_scores = pd.DataFrame(league_scores)
    df_finishes = pd.DataFrame(season_finishes)
    
    if not df_scores.empty:
        df_opp = df_scores[['Season', 'Week', 'Matchup ID', 'Points', 'Manager']].copy()
        df_opp.columns = ['Season', 'Week', 'Matchup ID', 'Opponent Points', 'Opponent']
        df_final = pd.merge(df_scores, df_opp, on=['Season', 'Week', 'Matchup ID'])
        df_final = df_final[df_final['Manager'] != df_final['Opponent']]
        df_final['Result'] = df_final.apply(lambda x: 'Win' if x['Points'] > x['Opponent Points'] else ('Loss' if x['Points'] < x['Opponent Points'] else 'Tie'), axis=1)
        df_final['Margin'] = df_final['Points'] - df_final['Opponent Points']
        
        season_stats = df_final[df_final['Week'] < 15].groupby(['Season', 'Roster ID']).agg(
            ActualWins=('Result', lambda x: (x == 'Win').sum()),
            ActualPts=('Points', 'sum')
        ).reset_index()
        
        df_finishes = pd.merge(df_finishes, season_stats, on=['Season', 'Roster ID'], how='left').fillna(0)
        df_finishes = df_finishes.sort_values(by=['Season', 'ActualWins', 'ActualPts'], ascending=[True, False, False])
        df_finishes['RegRank'] = df_finishes.groupby('Season').cumcount() + 1
        
        season_max = df_finishes.groupby('Season')['RegRank'].max().reset_index(name='MaxRank')
        df_finishes = pd.merge(df_finishes, season_max, on='Season', how='left')
        
        df_finishes['is_reg_season_champ'] = df_finishes['RegRank'] == 1
        df_finishes['is_reg_season_last'] = df_finishes['RegRank'] == df_finishes['MaxRank']

        losses = df_final[df_final['Result'] == 'Loss']
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
                    if diff > margin:
                        fatal_errors.append({'Season': row['Season'], 'Week': row['Week'], 'Manager': row['Manager'], 'Opponent': row['Opponent'], 'Margin': margin, 'Mistake': f"Start/Sit: Started {worst_starter['name']} ({worst_starter['pts']}) over {best_bench['name']} ({best_bench['pts']})", 'Points Lost': diff})
                        break 
        
        for _, row in losses.iterrows():
            margin = abs(row['Margin'])
            relevant_tx = [t for t in transaction_log if t['Manager'] == row['Manager'] and t['Week'] == row['Week'] and t['Season'] == row['Season'] and t.get('Type') == 'Swap']
            for t in relevant_tx:
                if (row['Season'], row['Week'], row['Manager'], t['Added ID']) not in starter_set: continue
                add_pts = weekly_points_map.get((row['Season'], row['Week'], t['Added ID']), 0)
                drop_pts = weekly_points_map.get((row['Season'], row['Week'], t['Dropped ID']), 0)
                diff = drop_pts - add_pts
                if diff > margin:
                     fatal_errors.append({'Season': row['Season'], 'Week': row['Week'], 'Manager': row['Manager'], 'Opponent': row['Opponent'], 'Margin': margin, 'Mistake': f"Bad Drop: Dropped {t['Dropped']} ({drop_pts}) for {t['Added']} ({add_pts})", 'Points Lost': diff})

        return df_final, pd.DataFrame(player_stats), pd.DataFrame(draft_registry), pd.DataFrame(transaction_log), pd.DataFrame(roster_snapshot_list), pd.DataFrame(fatal_errors), weekly_points_map, df_finishes, starter_set
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), set()

# --- APP UI ---
st.set_page_config(page_title="Sleeper Analytics", layout="wide", page_icon="🏆")
st.title("🏆 League Dashboard")

with st.sidebar:
    league_id_input = st.text_input("League ID", placeholder="e.g. 10483...")
    if not league_id_input: st.stop()

with st.spinner("Initializing Database..."):
    player_db = get_all_nfl_players()

current_managers = get_current_managers(league_id_input)
history = recursive_history(league_id_input)
if not history: st.stop()

df_scores, df_players, df_drafts, df_tx, df_rosters, df_errors, weekly_map, df_finishes, starter_set = process_data(history, player_db)
if df_scores.empty: st.warning("No data found."); st.stop()

df_finishes_agg = df_finishes.groupby(['Season', 'Manager']).agg({
    'is_league_champ': 'max', 'is_sacko': 'max', 'is_reg_season_champ': 'max', 
    'is_reg_season_last': 'max', 'RegRank': 'min', 'playoff_rank_precise': 'min',
    'ActualWins': 'max', 'ActualPts': 'max'
}).reset_index()

df_finishes_active = df_finishes_agg[df_finishes_agg['Manager'].isin(current_managers)]
df_scores_active = df_scores[df_scores['Manager'].isin(current_managers)]
df_errors_active = df_errors[df_errors['Manager'].isin(current_managers)] if not df_errors.empty else pd.DataFrame()
df_players_active = df_players[df_players['Manager'].isin(current_managers)]
df_tx_active = df_tx[df_tx['Manager'].isin(current_managers)] if not df_tx.empty else pd.DataFrame()
df_rosters_active = df_rosters[df_rosters['Manager'].isin(current_managers)] if not df_rosters.empty else pd.DataFrame()

seasons_list = sorted(df_scores['Season'].unique(), reverse=True)
def filter_by_season(df, season_selection):
    if season_selection == "All Time": return df
    return df[df['Season'] == season_selection]

# --- DRAFT ANALYSIS CALCS ---
player_season_totals = df_players.groupby(['Season', 'Player ID', 'Position', 'Player'])['Points'].sum().reset_index()
player_season_totals = player_season_totals.sort_values(by=['Season', 'Position', 'Points'], ascending=[True, True, False])
player_season_totals['Season Pos Rank'] = player_season_totals.groupby(['Season', 'Position']).cumcount() + 1

if not df_drafts.empty:
    df_draft_analysis = pd.merge(df_drafts, player_season_totals[['Season', 'Player ID', 'Points', 'Season Pos Rank']], on=['Season', 'Player ID'], how='inner')
    df_draft_analysis['Value Delta'] = df_draft_analysis['Draft Pos Rank'] - df_draft_analysis['Season Pos Rank']
    max_pick = df_draft_analysis['Pick'].max()
    df_draft_analysis['Draft Capital'] = (max_pick + 1) - df_draft_analysis['Pick']
else:
    df_draft_analysis = pd.DataFrame()

# --- LOYALTY CALCULATION ---
if not df_drafts.empty and not df_rosters_active.empty:
    drafted = df_drafts.groupby(['Season', 'Manager'])['Player ID'].apply(set).reset_index(name='DraftedSet')
    rostered = df_rosters.groupby(['Season', 'Manager'])['Player ID'].apply(set).reset_index(name='RosterSet')
    loyalty_df = pd.merge(drafted, rostered, on=['Season', 'Manager'], how='inner')
    def calc_retention(row):
        draft_set = row['DraftedSet']; roster_set = row['RosterSet']
        return len(draft_set.intersection(roster_set)) / len(draft_set) if draft_set else 0
    loyalty_df['Retention'] = loyalty_df.apply(calc_retention, axis=1)
    loyalty_agg = loyalty_df.groupby('Manager')['Retention'].mean().reset_index()
else:
    loyalty_agg = pd.DataFrame(columns=['Manager', 'Retention'])

# TABS
tab_dash, tab_mgr, tab_pos, tab_trophy = st.tabs(["🏠 League Dashboard", "👤 Manager Deep Dive", "🏈 Positional Analysis", "🏆 Trophy Room"])

# --- TAB 1: MAIN DASHBOARD ---
with tab_dash:
    st.header("League Standings")
    r_season = st.selectbox("Season", ["All Time"] + seasons_list, key="dash_sel")
    
    scores_view = filter_by_season(df_scores_active, r_season)
    tx_view = filter_by_season(df_tx_active, r_season)
    
    standings = scores_view.groupby('Manager').agg(
        Wins=('Result', lambda x: (x == 'Win').sum()),
        Losses=('Result', lambda x: (x == 'Loss').sum()),
        Ties=('Result', lambda x: (x == 'Tie').sum()),
        PF=('Points', 'sum'),
        PA=('Opponent Points', 'sum')
    ).reset_index()
    
    weekly_groups = scores_view.groupby(['Season', 'Week'])
    ap_records = {mgr: {'W': 0, 'L': 0, 'T': 0} for mgr in standings['Manager']}
    for _, week_data in weekly_groups:
        scores = week_data[['Manager', 'Points']].values
        for m1, s1 in scores:
            if m1 not in ap_records: continue
            for m2, s2 in scores:
                if m1 == m2: continue
                if s1 > s2: ap_records[m1]['W'] += 1
                elif s1 < s2: ap_records[m1]['L'] += 1
                else: ap_records[m1]['T'] += 1
    
    standings['AP_W'] = standings['Manager'].map(lambda x: ap_records.get(x, {}).get('W', 0))
    standings['AP_L'] = standings['Manager'].map(lambda x: ap_records.get(x, {}).get('L', 0))
    standings['AP_T'] = standings['Manager'].map(lambda x: ap_records.get(x, {}).get('T', 0))
    
    if not tx_view.empty:
        moves = tx_view[tx_view['Type'] != 'Trade'].groupby('Manager')['Transaction ID'].nunique().reset_index(name='Moves')
        trades = tx_view[tx_view['Type'] == 'Trade'].groupby('Manager')['Transaction ID'].nunique().reset_index(name='Trades')
        standings = pd.merge(standings, moves, on='Manager', how='left').fillna(0)
        standings = pd.merge(standings, trades, on='Manager', how='left').fillna(0)
    else:
        standings['Moves'] = 0; standings['Trades'] = 0
        
    standings['Diff'] = standings['PF'] - standings['PA']
    total_games = standings['Wins'] + standings['Losses'] + standings['Ties']
    standings['Avg PF'] = standings['PF'] / total_games
    standings['Avg PA'] = standings['PA'] / total_games
    standings['Avg Diff'] = standings['Diff'] / total_games
    standings['Win %'] = (standings['Wins'] + (0.5 * standings['Ties'])) / total_games
    total_ap_games = standings['AP_W'] + standings['AP_L'] + standings['AP_T']
    standings['All-Play %'] = (standings['AP_W'] + (0.5 * standings['AP_T'])) / total_ap_games
    standings['Luck'] = standings['Win %'] - standings['All-Play %']
    
    standings['Record'] = standings.apply(lambda x: f"{int(x['Wins'])}-{int(x['Losses'])}-{int(x['Ties'])}", axis=1)
    standings['All-Play Record'] = standings.apply(lambda x: f"{int(x['AP_W'])}-{int(x['AP_L'])}-{int(x['AP_T'])}", axis=1)
    
    standings = standings.sort_values(by=['Wins', 'PF'], ascending=False).reset_index(drop=True)
    standings.index += 1
    
    cols_order = ['Manager', 'Record', 'Win %', 'All-Play Record', 'All-Play %', 'Luck', 'PF', 'PA', 'Avg PF', 'Avg PA', 'Avg Diff', 'Moves', 'Trades']
    format_dict = {'PF': '{:,.1f}', 'PA': '{:,.1f}', 'Avg PF': '{:.1f}', 'Avg PA': '{:.1f}', 'Avg Diff': '{:.1f}', 'Win %': '{:.1%}', 'All-Play %': '{:.1%}', 'Luck': '{:+.1%}', 'Moves': '{:.0f}', 'Trades': '{:.0f}'}
    def color_luck(val): return f'color: {"green" if val > 0 else "red"}'
    st.dataframe(standings[cols_order].style.format(format_dict).applymap(color_luck, subset=['Luck']), use_container_width=True)
    
    st.divider()
    
    col_fraud, col_fatal = st.columns([2, 1])
    with col_fraud:
        st.subheader("The Fraud Quadrant")
        if r_season == "All Time":
            x_col = 'Avg PF'; y_col = 'Avg PA'; title = "The Fraud Quadrant (Avg Pts/Game)"
        else:
            x_col = 'PF'; y_col = 'PA'; title = "The Fraud Quadrant (Total Pts)"
        fig = px.scatter(standings, x=x_col, y=y_col, text='Manager', size='Wins', color='Wins', color_continuous_scale='Viridis', title=title)
        fig.add_hline(y=standings[y_col].mean(), line_dash="dash"); fig.add_vline(x=standings[x_col].mean(), line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with col_fatal:
        st.subheader("💀 Fatal Errors")
        r_errors = filter_by_season(df_errors_active, r_season)
        if r_errors.empty: st.success("No fatal errors found.")
        else:
            r_errors = r_errors.sort_values('Points Lost', ascending=False)
            error_managers = r_errors['Manager'].unique()
            for mgr in error_managers:
                mgr_errors = r_errors[r_errors['Manager'] == mgr]
                with st.expander(f"{mgr} ({len(mgr_errors)})"):
                    for _, row in mgr_errors.iterrows():
                        st.error(f"**Wk {row['Week']} {row['Season']}**: Lost by {row['Margin']:.1f}. {row['Mistake']}")

    st.divider()
    st.subheader("⚔️ The Nemesis Matrix")
    matrix = scores_view[scores_view['Result'] == 'Win'].groupby(['Manager', 'Opponent']).size().unstack(fill_value=0)
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
    
    if mgr_season == "All Time":
        mgr_fin = df_finishes[df_finishes['Manager'] == selected_mgr]
    else:
        mgr_fin = df_finishes[(df_finishes['Manager'] == selected_mgr) & (df_finishes['Season'] == mgr_season)]
    
    if not mgr_games.empty:
        high_s = mgr_games['Points'].max()
        wins = len(mgr_games[mgr_games['Result'] == 'Win'])
        avg_pts = mgr_games['Points'].mean()
        
        opp_stats = mgr_games.groupby('Opponent').agg(Wins=('Result', lambda x: (x == 'Win').sum()), Losses=('Result', lambda x: (x == 'Loss').sum()), Games=('Result', 'count'), AvgMargin=('Margin', 'mean')).reset_index()
        opp_stats['NetWins'] = opp_stats['Wins'] - opp_stats['Losses']
        nemesis = opp_stats.sort_values(by=['NetWins', 'Games', 'AvgMargin'], ascending=[True, False, True]).iloc[0] if not opp_stats.empty else None
        easy_beat = opp_stats.sort_values(by=['NetWins', 'Games', 'AvgMargin'], ascending=[False, False, False]).iloc[0] if not opp_stats.empty else None
        
        reg_str = ""; post_str = ""
        if mgr_season == "All Time":
            reg_str = f"Avg: {mgr_fin['RegRank'].mean():.1f}"
            post_str = f"Avg: {mgr_fin['playoff_rank_precise'].mean():.1f}"
        else:
            if not mgr_fin.empty:
                row = mgr_fin.iloc[0]
                reg_str = f"Rank {row['RegRank']:.0f}"
                p_rank = row['playoff_rank_precise']
                if pd.notna(p_rank):
                    if p_rank == 1: post_str = "🏆 Champion"
                    elif p_rank == 2: post_str = "🥈 Runner Up"
                    else: post_str = f"{p_rank:.0f}th Place"
                else:
                    if row['is_league_champ']: post_str = "🏆 Champion"
                    elif row['is_sacko']: post_str = "🚽 Sacko"
                    else: post_str = "Missed Playoffs"
            else: reg_str = "N/A"; post_str = "N/A"
        
        if not loyalty_agg.empty:
            l_score = loyalty_agg[loyalty_agg['Manager'] == selected_mgr]['Retention'].values
            l_val = l_score[0] if len(l_score) > 0 else 0
            if l_val > 0.6: l_label = "💎 Diamond Hands"
            elif l_val < 0.3: l_label = "♻️ The Churn"
            else: l_label = "⚖️ Balanced"
        else: l_label = "N/A"; l_val = 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Win Rate", f"{(wins/len(mgr_games)*100):.1f}%", f"{wins} Wins")
        c2.metric("Avg Score", f"{avg_pts:.1f}")
        c3.metric("Reg Season", reg_str)
        c4.metric("League Finish", post_str)
        c5.metric("Loyalty Style", l_label, f"{l_val:.0%}")
        
        c6, c7 = st.columns(2)
        if nemesis is not None:
            c6.metric("😈 Biggest Nemesis", f"{nemesis['Opponent']}", f"Record: {nemesis['Wins']}-{nemesis['Losses']} (Avg {nemesis['AvgMargin']:.1f})")
        if easy_beat is not None:
            c7.metric("🧸 Easiest Beat", f"{easy_beat['Opponent']}", f"Record: {easy_beat['Wins']}-{easy_beat['Losses']} (Avg +{easy_beat['AvgMargin']:.1f})")
            
        st.divider()
        st.subheader("💥 Records")
        wins_df = mgr_games[mgr_games['Result'] == 'Win']; losses_df = mgr_games[mgr_games['Result'] == 'Loss']
        big_win = wins_df.loc[wins_df['Margin'].idxmax()] if not wins_df.empty else None
        close_loss = losses_df.loc[losses_df['Margin'].idxmax()] if not losses_df.empty else None
        r1, r2 = st.columns(2)
        if big_win is not None:
            r1.info(f"**Biggest Win:** +{big_win['Margin']:.1f} vs {big_win['Opponent']} (Wk {big_win['Week']} {big_win['Season']})")
        if close_loss is not None:
            r2.error(f"**Closest Loss:** {close_loss['Margin']:.1f} vs {close_loss['Opponent']} (Wk {close_loss['Week']} {close_loss['Season']})")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("🧬 Roster DNA")
            mgr_players = filter_by_season(df_players_active[df_players_active['Manager'] == selected_mgr], mgr_season)
            mgr_players = mgr_players[mgr_players['Status'] == 'Starter']
            if not mgr_players.empty:
                pos_dna = mgr_players.groupby('Position')['Points'].sum().reset_index()
                fig_dna = px.pie(pos_dna, values='Points', names='Position', hole=0.4, title="Points by Position")
                st.plotly_chart(fig_dna, use_container_width=True)
            else: st.info("No starter data.")
            if mgr_season != "All Time":
                st.subheader("📈 Efficiency Trend")
                eff_data = mgr_games.sort_values(by=['Season', 'Week'])
                eff_data['Rolling Eff'] = eff_data['Efficiency'].expanding().mean()
                fig_eff = px.line(eff_data, x='Week', y='Rolling Eff', markers=True, title=f"YTD Efficiency ({mgr_season})")
                st.plotly_chart(fig_eff, use_container_width=True)
        with col_g2:
            st.subheader("💼 GM Activity")
            if not df_tx_active.empty:
                mgr_tx = df_tx_active[df_tx_active['Manager'] == selected_mgr]
                if not mgr_tx.empty:
                    if mgr_season == "All Time":
                        tx_counts = mgr_tx.groupby(['Season', 'Type'])['Transaction ID'].nunique().reset_index(name='Count')
                        fig_tx = px.line(tx_counts, x='Season', y='Count', color='Type', markers=True, title="Activity Trend")
                        st.plotly_chart(fig_tx, use_container_width=True)
                        faab_data = mgr_tx.groupby('Season')['Bid'].sum().reset_index()
                        fig_faab = px.line(faab_data, x='Season', y='Bid', markers=True, title="FAAB Spend per Season")
                        st.plotly_chart(fig_faab, use_container_width=True)
                    else:
                        season_tx = mgr_tx[mgr_tx['Season'] == mgr_season].copy()
                        if not season_tx.empty:
                            wk_tx = season_tx.groupby(['Week', 'Type'])['Transaction ID'].nunique().reset_index(name='Count')
                            wk_pivot = wk_tx.pivot(index='Week', columns='Type', values='Count').fillna(0)
                            wk_cum = wk_pivot.cumsum().reset_index().melt(id_vars='Week', var_name='Type', value_name='Cumulative Moves')
                            fig_tx = px.line(wk_cum, x='Week', y='Cumulative Moves', color='Type', markers=True, title=f"Running Total Moves")
                            st.plotly_chart(fig_tx, use_container_width=True)
                            wk_faab = season_tx.groupby('Week')['Bid'].sum().reset_index()
                            wk_faab['Cumulative FAAB'] = wk_faab['Bid'].cumsum()
                            fig_faab = px.line(wk_faab, x='Week', y='Cumulative FAAB', markers=True, title=f"Cumulative FAAB Spend")
                            st.plotly_chart(fig_faab, use_container_width=True)
                        else: fig_tx = px.bar(title="No Moves")
                else: st.info("No transactions found.")
            else: st.info("No transactions found.")
        
        st.divider()
        st.subheader(f"🌟 Best {mgr_season} Players")
        all_players_filtered = filter_by_season(df_players_active, mgr_season)
        ranking_db = all_players_filtered.groupby(['Position', 'Player', 'Season'])['Points'].sum().reset_index()
        ranking_db['Rank'] = ranking_db.groupby(['Season', 'Position'])['Points'].rank(method='min', ascending=False)
        my_roster = filter_by_season(df_players_active[df_players_active['Manager'] == selected_mgr], mgr_season)
        my_best = my_roster.groupby(['Position', 'Player', 'Season'])['Points'].sum().reset_index()
        cols = st.columns(6)
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        for i, pos in enumerate(positions):
            with cols[i]:
                pos_players = my_best[my_best['Position'] == pos]
                if not pos_players.empty:
                    top_n = 2 if pos in ['RB', 'WR'] else 1
                    best_players = pos_players.nlargest(top_n, 'Points')
                    for _, best_player in best_players.iterrows():
                        global_rank_row = ranking_db[(ranking_db['Position'] == pos) & (ranking_db['Player'] == best_player['Player']) & (ranking_db['Season'] == best_player['Season'])]
                        rank_val = global_rank_row['Rank'].values[0] if not global_rank_row.empty else 999
                        p_id = my_roster[my_roster['Player'] == best_player['Player']].iloc[0]['Player ID']
                        p_seas = best_player['Season']
                        starts = len(my_roster[(my_roster['Player'] == best_player['Player']) & (my_roster['Season'] == p_seas) & (my_roster['Status'] == 'Starter')])
                        acq_str = "Rostered"
                        if not df_drafts.empty:
                            d_row = df_drafts[(df_drafts['Player ID'] == p_id) & (df_drafts['Season'] == p_seas) & (df_drafts['Manager'] == selected_mgr)]
                            if not d_row.empty: acq_str = f"Drafted (Rd {d_row.iloc[0]['Round']})"
                        if acq_str == "Rostered" and not df_tx_active.empty:
                            t_row = df_tx_active[(df_tx_active['Added ID'] == p_id) & (df_tx_active['Season'] == p_seas) & (df_tx_active['Manager'] == selected_mgr) & (df_tx_active['Type'] == 'Trade')]
                            if not t_row.empty: acq_str = "Traded"
                            w_row = df_tx_active[(df_tx_active['Added ID'] == p_id) & (df_tx_active['Season'] == p_seas) & (df_tx_active['Manager'] == selected_mgr) & (df_tx_active['Type'].isin(['Waiver', 'Free Agent']))]
                            if not w_row.empty: acq_str = "FA/Waiver"
                        st.markdown(f"**{best_player['Player']}** *{p_seas}* **{pos}{int(rank_val)}** ({int(best_player['Points'])})<br><span style='font-size:12px; color:gray'>{acq_str}<br>{starts} Starts</span><hr style='margin: 5px 0;'>", unsafe_allow_html=True)
                else: st.markdown(f"**{pos}**<br>-", unsafe_allow_html=True)

        st.divider()
        sub1, sub2, sub3 = st.tabs(["📜 Draft Board", "🧐 Streamer Grades", "🎲 Sim"])
        with sub1:
            if not df_draft_analysis.empty and mgr_season != "All Time":
                my_draft = df_draft_analysis[(df_draft_analysis['Manager'] == selected_mgr) & (df_draft_analysis['Season'] == mgr_season)].copy()
                if not my_draft.empty:
                    my_draft = my_draft.sort_values('Pick')
                    my_draft['Drafted'] = my_draft['Position'] + my_draft['Draft Pos Rank'].astype(int).astype(str)
                    my_draft['Finished'] = my_draft['Position'] + my_draft['Season Pos Rank'].astype(int).astype(str)
                    my_draft['Value'] = my_draft['Value Delta']
                    def color_value(val):
                        if val > 10: return 'background-color: #d4edda; color: black'
                        if val < -10: return 'background-color: #f8d7da; color: black'
                        return ''
                    st.dataframe(my_draft[['Round', 'Pick', 'Player', 'Drafted', 'Finished', 'Value']].style.applymap(color_value, subset=['Value']), use_container_width=True)
                else: st.info("No draft data found.")
            else: st.info("Select specific season.")
        with sub2:
            mgr_tx = filter_by_season(df_tx_active, mgr_season)
            mgr_tx = mgr_tx[mgr_tx['Manager'] == selected_mgr]
            swap_groups = mgr_tx.groupby(['Transaction ID', 'Season', 'Week'])
            swaps = []
            for (tid, seas, wk), group in swap_groups:
                adds = group[group['Action'] == 'Add']; drops = group[group['Action'] == 'Drop']
                if len(adds) == 1 and len(drops) == 1:
                    add_row = adds.iloc[0]; drop_row = drops.iloc[0]
                    if (seas, wk, selected_mgr, add_row['Added ID']) not in starter_set: continue
                    p_add_pts = weekly_map.get((seas, wk, add_row['Added ID']), 0)
                    p_drop_pts = weekly_map.get((seas, wk, drop_row['Dropped ID']), 0)
                    if p_drop_pts == 0: continue
                    net = p_add_pts - p_drop_pts
                    grade = assign_streamer_grade(net)
                    add_name = get_all_nfl_players().get(add_row['Added ID'], {}).get('name', 'Unknown')
                    drop_name = get_all_nfl_players().get(drop_row['Dropped ID'], {}).get('name', 'Unknown')
                    swaps.append({'Week': f"{seas} Wk {wk}", 'Added': f"{add_name} ({p_add_pts})", 'Dropped': f"{drop_name} ({p_drop_pts})", 'Net': net, 'Grade': grade})
            if swaps: st.dataframe(pd.DataFrame(swaps).sort_values('Net', ascending=False), hide_index=True)
            else: st.info("No started streamer swaps found.")
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
    st.header("🏈 Positional Analysis")
    p_season = st.selectbox("Season", ["All Time"] + seasons_list, key="pos_sel")
    if not df_players_active.empty:
        p_df = filter_by_season(df_players_active, p_season)
        pos = st.radio("Position", ["QB", "RB", "WR", "TE", "K", "DEF"], horizontal=True)
        grouped = p_df[p_df['Position'] == pos].groupby('Manager')['Points'].agg(['sum', 'count', 'mean']).reset_index()
        grouped.columns = ['Manager', 'Total Pts', 'Starts', 'Avg Pts']
        granular = p_df[p_df['Position'] == pos].groupby(['Manager', 'Week'])['Points'].sum().reset_index()
        std_devs = granular.groupby('Manager')['Points'].std().reset_index(name='Volatility')
        final_df = pd.merge(grouped, std_devs, on='Manager', how='left').fillna(0)
        
        col_list, col_chart = st.columns([1, 2])
        with col_list:
            st.dataframe(final_df[['Manager', 'Total Pts', 'Avg Pts', 'Volatility']].sort_values('Total Pts', ascending=False), hide_index=True)
        with col_chart:
            fig_bb = px.scatter(final_df, x='Avg Pts', y='Volatility', text='Manager', size='Starts', color='Total Pts', title="Boom vs Bust Analysis")
            st.plotly_chart(fig_bb, use_container_width=True)
            
        st.divider()
        if not df_draft_analysis.empty and p_season != "All Time":
            st.subheader(f"Draft ROI: {pos} ({p_season})")
            pos_draft = df_draft_analysis[(df_draft_analysis['Position'] == pos) & (df_draft_analysis['Season'] == p_season)]
            if not pos_draft.empty:
                roi_stats = pos_draft.groupby('Manager').agg({'Draft Capital': 'sum', 'Points': 'sum'}).reset_index()
                fig_roi = px.scatter(roi_stats, x='Draft Capital', y='Points', text='Manager', size='Points', title=f"Draft Capital Spent vs Points Returned ({pos})")
                st.plotly_chart(fig_roi, use_container_width=True)
            else: st.info(f"No {pos} drafted.")
        elif p_season == "All Time": st.info("Select a season for Draft ROI.")

# --- TAB 4: TROPHY ROOM ---
with tab_trophy:
    st.header("🏆 The Trophy Room")
    t_season = st.selectbox("Season", ["All Time"] + seasons_list, key="trophy_sel")
    
    if t_season == "All Time":
        st.subheader("🐐 The GOAT Table (All-Time)")
        goat_df = df_finishes_active.groupby('Manager').agg({'ActualWins': 'sum', 'ActualPts': 'sum', 'Season': 'count', 'is_league_champ': 'sum', 'is_sacko': 'sum'}).reset_index()
        goat_df.columns = ['Manager', 'Wins', 'Points', 'Seasons', 'Titles', 'Sackos']
        goat_df['Win %'] = goat_df['Wins'] / (goat_df['Seasons'] * 14) 
        goat_df = goat_df.sort_values(by=['Titles', 'Wins'], ascending=False)
        st.dataframe(goat_df.style.format({'Win %': '{:.1%}', 'Points': '{:,.0f}'}), use_container_width=True)
        st.divider()

    # DYNAMIC TROPHIES (Work for Single Season too now)
    
    # 1. Draft Awards
    current_draft_data = df_draft_analysis if t_season == "All Time" else df_draft_analysis[df_draft_analysis['Season'] == t_season]
    if not current_draft_data.empty:
        st.subheader(f"📜 Draft Hall of Fame ({t_season})")
        
        # Best Value (Non-Keeper)
        potential_steals = current_draft_data[(current_draft_data['Round'] > 2) & (current_draft_data['Is Keeper'] == False)]
        if not potential_steals.empty:
            best_val = potential_steals.sort_values('Value Delta', ascending=False).iloc[0]
            st.success(f"**💎 The Tom Brady (Best Value)**\n\n**{best_val['Player']}** ({best_val['Season']})\nDrafted: {best_val['Position']}{int(best_val['Draft Pos Rank'])} (Rd {best_val['Round']}) ➡️ Finished: {best_val['Position']}{int(best_val['Season Pos Rank'])}\n**Manager: {best_val['Manager']}**")
        
        # Biggest Bust (Any)
        if not current_draft_data.empty:
            worst_bust = current_draft_data.sort_values('Value Delta', ascending=True).iloc[0]
            st.error(f"**📉 The Jamarcus Russell (Biggest Bust)**\n\n**{worst_bust['Player']}** ({worst_bust['Season']})\nDrafted: {worst_bust['Position']}{int(worst_bust['Draft Pos Rank'])} (Rd {worst_bust['Round']}) ➡️ Finished: {worst_bust['Position']}{int(worst_bust['Season Pos Rank'])}\n**Manager: {worst_bust['Manager']}**")
        
        # Worst Keeper
        keeper_drafts = current_draft_data[current_draft_data['Is Keeper'] == True]
        if not keeper_drafts.empty:
            worst_keeper = keeper_drafts.sort_values('Value Delta', ascending=True).iloc[0]
            st.warning(f"**💩 The Worst Keeper**\n\n**{worst_keeper['Player']}** ({worst_keeper['Season']})\nKept: {worst_keeper['Position']}{int(worst_keeper['Draft Pos Rank'])} (Rd {worst_keeper['Round']}) ➡️ Finished: {worst_keeper['Position']}{int(worst_keeper['Season Pos Rank'])}\n**Manager: {worst_keeper['Manager']}**")
            
    st.divider()
    
    # 2. Loyalty
    if not loyalty_agg.empty:
        st.subheader(f"🛡️ Loyalty Awards ({t_season})")
        if t_season == "All Time":
            loyal_champ = loyalty_agg.sort_values('Retention', ascending=False).iloc[0]
            churn_champ = loyalty_agg.sort_values('Retention', ascending=True).iloc[0]
        else:
            # Re-calc for single season from raw loyalty_df (before aggregation)
            # Need to access loyalty_df which was calculated globally. 
            # Ideally passed in process_data but currently global.
            # Re-deriving for safety:
            if not df_drafts.empty and not df_rosters_active.empty:
                # We need historical rosters for single seasons? 
                # Sleeper API only gives current rosters for the league ID passed.
                # History loop gives us historical leagues.
                # We need to ensure loyalty_df has all seasons. It does.
                current_loyalty = loyalty_df[loyalty_df['Season'] == t_season]
                if not current_loyalty.empty:
                    loyal_champ = current_loyalty.sort_values('Retention', ascending=False).iloc[0]
                    churn_champ = current_loyalty.sort_values('Retention', ascending=True).iloc[0]
                else: loyal_champ = None
            else: loyal_champ = None

        if loyal_champ is not None:
            c1, c2 = st.columns(2)
            c1.success(f"**Most Loyal: {loyal_champ['Manager']}** ({loyal_champ['Retention']:.1%} Retention)")
            c2.error(f"**Least Loyal: {churn_champ['Manager']}** ({churn_champ['Retention']:.1%} Retention)")
    
    st.divider()
    
    # 3. FAAB
    current_tx = df_tx_active if t_season == "All Time" else df_tx_active[df_tx_active['Season'] == t_season]
    if not current_tx.empty:
        st.subheader(f"💸 FAAB Awards ({t_season})")
        faab_agg = current_tx.groupby('Manager')['Bid'].sum().reset_index()
        if not faab_agg.empty:
            whale = faab_agg.sort_values('Bid', ascending=False).iloc[0]
            pincher = faab_agg.sort_values('Bid', ascending=True).iloc[0]
            c1, c2 = st.columns(2)
            c1.info(f"**🐳 The Whale: {whale['Manager']}** (${whale['Bid']})")
            c2.warning(f"**🐁 Penny Pincher: {pincher['Manager']}** (${pincher['Bid']})")

    st.divider()

    # Standard Trophies (League Outcomes)
    # Filter based on selection
    display_finishes = df_finishes_active if t_season == "All Time" else df_finishes_active[df_finishes_active['Season'] == t_season]
    
    champs = display_finishes[display_finishes['is_league_champ']].groupby('Manager')['Season'].apply(list).reset_index()
    champs['Count'] = champs['Season'].apply(len)
    champs['Years'] = champs['Season'].apply(lambda x: ", ".join(map(str, x)))
    
    sackos = display_finishes[display_finishes['is_sacko']].groupby('Manager')['Season'].apply(list).reset_index()
    sackos['Count'] = sackos['Season'].apply(len)
    sackos['Years'] = sackos['Season'].apply(lambda x: ", ".join(map(str, x)))
    
    reg_leaders = display_finishes[display_finishes['is_reg_season_champ']].groupby('Manager')['Season'].apply(list).reset_index()
    reg_leaders['Count'] = reg_leaders['Season'].apply(len)
    reg_leaders['Years'] = reg_leaders['Season'].apply(lambda x: ", ".join(map(str, x)))
    
    reg_losers = display_finishes[display_finishes['is_reg_season_last']].groupby('Manager')['Season'].apply(list).reset_index()
    reg_losers['Count'] = reg_losers['Season'].apply(len)
    reg_losers['Years'] = reg_losers['Season'].apply(lambda x: ", ".join(map(str, x)))
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("👑 League Champs")
        if not champs.empty: st.dataframe(champs[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
        else: st.info("None")
    with c2:
        st.subheader("🚽 The Sacko")
        if not sackos.empty: st.dataframe(sackos[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
        else: st.info("None")
    with c3:
        st.subheader("🥇 Reg Season #1")
        if not reg_leaders.empty: st.dataframe(reg_leaders[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
        else: st.info("None")
    with c4:
        st.subheader("📉 Reg Season Last")
        if not reg_losers.empty: st.dataframe(reg_losers[['Manager', 'Count', 'Years']].sort_values('Count', ascending=False), hide_index=True)
        else: st.info("None")
        
    st.divider()
    
    st.subheader("📈 Records of Extremes")
    t_scores_display = filter_by_season(df_scores_active, t_season)
    if not t_scores_display.empty:
        best_week = t_scores_display.loc[t_scores_display['Points'].idxmax()]
        worst_week = t_scores_display.loc[t_scores_display['Points'].idxmin()]
        seas_pts = t_scores_display.groupby(['Season', 'Manager'])['Points'].sum().reset_index()
        best_season = seas_pts.loc[seas_pts['Points'].idxmax()]
        worst_season = seas_pts.loc[seas_pts['Points'].idxmin()]
        t_tx_display = filter_by_season(df_tx_active, t_season)
        
        if not t_tx_display.empty:
            move_counts = t_tx_display[t_tx_display['Type'] != 'Trade'].groupby('Manager')['Transaction ID'].nunique().reset_index(name='Moves').sort_values('Moves', ascending=False)
            best_tx = move_counts.iloc[0] if not move_counts.empty else None
            trade_counts = t_tx_display[t_tx_display['Type'] == 'Trade'].groupby('Manager')['Transaction ID'].nunique().reset_index(name='Trades').sort_values('Trades', ascending=False)
            best_trader = trade_counts.iloc[0] if not trade_counts.empty else None
        else: best_tx = None; best_trader = None
            
        bench_king = t_scores_display.groupby('Manager')['Bench Points'].sum().reset_index().sort_values('Bench Points', ascending=False).iloc[0]
        bench_nuke = t_scores_display.loc[t_scores_display['Bench Points'].idxmax()]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Highest Weekly Score", f"{best_week['Points']}", f"{best_week['Manager']} ({best_week['Season']} Wk {best_week['Week']})")
        m2.metric("Lowest Weekly Score", f"{worst_week['Points']}", f"{worst_week['Manager']} ({worst_week['Season']} Wk {worst_week['Week']})")
        m3.metric("Highest Season Total", f"{best_season['Points']}", f"{best_season['Manager']} ({best_season['Season']})")
        m4.metric("Lowest Season Total", f"{worst_season['Points']}", f"{worst_season['Manager']} ({worst_season['Season']})")
        
        m5, m6, m7, m8 = st.columns(4)
        if best_tx is not None: m5.metric("Most Transactions", f"{best_tx['Moves']}", f"{best_tx['Manager']}")
        else: m5.metric("Most Transactions", "0", "N/A")
        if best_trader is not None: m6.metric("Most Trades", f"{best_trader['Trades']}", f"{best_trader['Manager']}")
        else: m6.metric("Most Trades", "0", "N/A")
        m7.metric("Benchwarmer King (Total)", f"{bench_king['Bench Points']:.0f}", f"{bench_king['Manager']}")
        m8.metric("Bench Nuke (1 Wk)", f"{bench_nuke['Bench Points']:.0f}", f"{bench_nuke['Manager']} (Wk {bench_nuke['Week']})")

    st.divider()
    
    st.subheader("Positional Awards (Starters Only)")
    df_players_starters = df_players_active[df_players_active['Status'] == 'Starter']
    t_ply = filter_by_season(df_players_starters, t_season)
    
    pos_grp = t_ply.groupby(['Manager', 'Position', 'Season'])['Points'].mean().reset_index()
    
    cols = st.columns(3)
    for i, p in enumerate(['QB', 'RB', 'WR', 'TE', 'K', 'DEF']):
        with cols[i % 3]:
            p_data = pos_grp[pos_grp['Position'] == p]
            if not p_data.empty:
                best = p_data.loc[p_data['Points'].idxmax()]
                worst = p_data.loc[p_data['Points'].idxmin()]
                with st.expander(f"👑 {p} King: {best['Manager']} ({best['Season']})"):
                    st.metric("Avg Pts", f"{best['Points']:.1f}")
                    details = t_ply[(t_ply['Manager'] == best['Manager']) & (t_ply['Season'] == best['Season']) & (t_ply['Position'] == p)]
                    st.dataframe(details[['Week', 'Player', 'Points']].sort_values('Points', ascending=False), hide_index=True)
                with st.expander(f"🤡 {p} Jester: {worst['Manager']} ({worst['Season']})"):
                    st.metric("Avg Pts", f"{worst['Points']:.1f}")
                    details = t_ply[(t_ply['Manager'] == worst['Manager']) & (t_ply['Season'] == worst['Season']) & (t_ply['Position'] == p)]
                    st.dataframe(details[['Week', 'Player', 'Points']].sort_values('Points', ascending=True), hide_index=True)

    st.divider()
    st.subheader("Records of Shame")
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
