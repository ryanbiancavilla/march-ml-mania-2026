"""
Vilpom — NCAA Tournament Analytics & Betting Intelligence
Advanced ML-powered predictions, efficiency ratings, and edge detection.
"""

import os
import math
import hmac
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEASON = 2026
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(
    page_title="Vilpom",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────── Password Gate ────────────────────────────
def check_password():
    """Block access until the correct password is entered."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        "<div style='text-align:center; margin-top:80px;'>"
        "<div style='font-size:56px; font-weight:900; letter-spacing:-2px; margin-bottom:4px;'>"
        "<span style='color:#FF6B35;'>VIL</span><span style='color:#FAFAFA;'>POM</span></div>"
        "<div style='color:#555; font-size:11px; letter-spacing:3px; font-weight:600;'>"
        "NCAA ANALYTICS & BETTING INTELLIGENCE</div>"
        "<div style='width:60px; height:3px; background:#FF6B35; margin:20px auto 0;'></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    # CSS to disable copy/paste/select on the password field
    st.markdown(
        """<style>
        input[type="password"] {
            -webkit-user-select: none; -moz-user-select: none;
            -ms-user-select: none; user-select: none;
            background: #12141a !important; border: 1px solid #2a2d36 !important;
            border-radius: 8px !important; padding: 12px !important;
            font-size: 14px !important; letter-spacing: 2px !important;
        }
        input[type="password"]:focus {
            border-color: #FF6B35 !important;
            box-shadow: 0 0 0 1px #FF6B35 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
    with col2:
        st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
        pwd = st.text_input("ACCESS CODE", type="password", key="pwd_input",
                           label_visibility="visible")
        if st.button("UNLOCK", use_container_width=True):
            if hmac.compare_digest(pwd, st.secrets["app_password"]):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
        st.markdown(
            "<div style='text-align:center; color:#444; font-size:10px; letter-spacing:1px; margin-top:20px;'>"
            "MEMBERS ONLY &middot; DO NOT SHARE</div>",
            unsafe_allow_html=True,
        )
    return False


if not check_password():
    st.stop()

# ──────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    .block-container { padding-top: 1rem; max-width: 1200px; }
    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Premium sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0c10 0%, #12141a 100%);
        border-right: 1px solid #1e2028;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500; letter-spacing: 0.3px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #1e2028; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px; font-weight: 600; letter-spacing: 0.3px;
        border-radius: 6px 6px 0 0;
    }

    /* Premium card base */
    .vp-card {
        background: linear-gradient(135deg, #1a1d24 0%, #14161c 100%);
        border: 1px solid #2a2d36; border-radius: 10px;
        padding: 16px 20px; margin: 6px 0;
        transition: border-color 0.2s;
    }
    .vp-card:hover { border-color: #FF6B35; }

    /* Section headers */
    .vp-section {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        color: #FF6B35; text-transform: uppercase; margin-bottom: 12px;
    }

    /* Style all Streamlit subheaders */
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 16px !important; font-weight: 700 !important;
        letter-spacing: 0.3px; color: #e2e8f0 !important;
        border-bottom: 1px solid #2a2d36; padding-bottom: 8px; margin-top: 8px;
    }
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 20px !important; font-weight: 800 !important;
        letter-spacing: -0.3px; color: #FAFAFA !important;
    }

    /* Style Streamlit captions */
    [data-testid="stCaptionContainer"] p {
        font-size: 11px !important; letter-spacing: 0.3px;
    }

    /* Style selectbox/inputs */
    [data-testid="stSelectbox"] label, [data-testid="stTextInput"] label,
    [data-testid="stCheckbox"] label, [data-testid="stSlider"] label,
    [data-testid="stMultiSelect"] label {
        font-size: 12px !important; font-weight: 600 !important;
        letter-spacing: 0.5px; text-transform: uppercase; color: #888 !important;
    }

    /* Streamlit hr dividers */
    hr { border-color: #1e2028 !important; margin: 20px 0 !important; }

    /* Radio button styling */
    [data-testid="stRadio"] > div > label {
        font-size: 13px !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #e55a2b 100%) !important;
        color: #fff !important; font-weight: 700 !important;
        border: none !important; border-radius: 8px !important;
        letter-spacing: 0.5px; padding: 8px 24px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff7e4d 0%, #FF6B35 100%) !important;
        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.3) !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #FF6B35 !important; }

    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #2a2d36 !important; border-radius: 10px !important;
        background: linear-gradient(135deg, #1a1d24 0%, #14161c 100%);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important; font-size: 13px !important;
    }

    /* Rankings table */
    .vp-table { width: 100%; border-collapse: separate; border-spacing: 0; }
    .vp-table thead th {
        background: #12141a; color: #888; font-size: 11px; font-weight: 700;
        letter-spacing: 0.8px; text-transform: uppercase; padding: 10px 12px;
        border-bottom: 2px solid #FF6B35; position: sticky; top: 0; z-index: 1;
        text-align: left;
    }
    .vp-table tbody tr { transition: background 0.15s; }
    .vp-table tbody tr:hover { background: rgba(255, 107, 53, 0.06); }
    .vp-table tbody td {
        padding: 8px 12px; border-bottom: 1px solid #1a1d24;
        font-size: 13px; font-variant-numeric: tabular-nums;
    }
    .vp-table .rank-cell {
        font-weight: 800; color: #FF6B35; font-size: 14px; width: 36px; text-align: center;
    }
    .vp-table .team-cell { font-weight: 600; color: #FAFAFA; }
    .vp-table .seed-badge {
        display: inline-block; background: #FF6B35; color: #000; font-weight: 700;
        font-size: 10px; padding: 1px 6px; border-radius: 3px; min-width: 20px; text-align: center;
    }
    .vp-table .stat-good { color: #4ade80; }
    .vp-table .stat-bad { color: #f87171; }
    .vp-table .stat-neutral { color: #aaa; }

    /* Tier badges */
    .tier-elite { color: #000; background: #FF6B35; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; }
    .tier-strong { color: #000; background: #4ade80; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; }
    .tier-solid { color: #000; background: #60a5fa; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; }
    .tier-avg { color: #000; background: #888; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10px; }

    /* Bracket */
    .bracket { display: flex; gap: 6px; min-height: 540px; overflow-x: auto; }
    .round {
        display: flex; flex-direction: column; justify-content: space-around;
        min-width: 165px;
    }
    .round-title {
        text-align: center; font-size: 11px; color: #FF6B35;
        margin-bottom: 4px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    }
    .matchup {
        border: 1px solid #2a2d36; border-radius: 6px; overflow: hidden;
        margin: 2px 0; background: linear-gradient(135deg, #1a1d24 0%, #14161c 100%);
    }
    .team-slot {
        padding: 3px 8px; font-size: 12px; display: flex;
        justify-content: space-between; align-items: center;
        border-bottom: 1px solid #2a2d34; color: #ccc;
    }
    .team-slot:last-child { border-bottom: none; }
    .team-slot.winner {
        background: rgba(255, 107, 53, 0.12); color: #FF6B35;
        font-weight: 700;
    }
    .seed-tag { color: #666; margin-right: 4px; font-size: 11px; }
    .prob-tag { color: #666; font-size: 10px; }
    .big-prob {
        font-size: 48px; font-weight: 900; text-align: center;
        line-height: 1.1; margin: 10px 0; letter-spacing: -2px;
    }
    .stat-bar { height: 6px; border-radius: 3px; margin: 2px 0; }
    .ff-bracket { display: flex; gap: 12px; min-height: 180px; align-items: center; }
    .ff-round {
        display: flex; flex-direction: column; justify-content: space-around;
        min-width: 180px; min-height: 160px;
    }
    .champ-banner {
        text-align: center; font-size: 28px; font-weight: 900;
        padding: 20px; border: 2px solid #FF6B35; border-radius: 10px;
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.12) 0%, rgba(255, 107, 53, 0.04) 100%);
        color: #FF6B35; margin: 12px 0; letter-spacing: -0.5px;
    }

    /* Metric cards */
    .vp-metric {
        background: linear-gradient(135deg, #1a1d24 0%, #14161c 100%);
        border: 1px solid #2a2d36; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .vp-metric .label { color: #888; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
    .vp-metric .value { font-size: 28px; font-weight: 800; margin: 4px 0; }
    .vp-metric .sub { color: #aaa; font-size: 12px; }

    /* Hide streamlit dataframe styling overrides */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] > div { border-radius: 10px; }

    /* Betting card upgrades */
    .vp-bet-card {
        background: linear-gradient(135deg, #1a1d24 0%, #14161c 100%);
        border: 1px solid #2a2d36; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
    }
    .vp-bet-type {
        background: #0e1117; border-radius: 8px; padding: 12px 16px;
        flex: 1; min-width: 180px;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2a2d36; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF6B35; }
</style>
""", unsafe_allow_html=True)


def _styled_table(rows, columns=None, max_height=None):
    """Render a list of dicts as a premium styled HTML table."""
    if not rows:
        return
    if columns is None:
        columns = list(rows[0].keys())
    # Filter out internal columns starting with _
    columns = [c for c in columns if not c.startswith("_")]
    h = max_height or 0
    wrapper_style = f'max-height:{h}px; overflow-y:auto; ' if h else ''
    html = f'<div style="{wrapper_style}border-radius:10px; border:1px solid #2a2d36;">'
    html += '<table class="vp-table"><thead><tr>'
    for c in columns:
        html += f'<th>{c.upper()}</th>'
    html += '</tr></thead><tbody>'
    for r in rows:
        html += '<tr>'
        for c in columns:
            val = r.get(c, "")
            style = ''
            sval = str(val)
            if sval.startswith('+') or 'HIT' in sval:
                style = ' style="color:#4ade80; font-weight:600;"'
            elif 'MISS' in sval:
                style = ' style="color:#f87171; font-weight:600;"'
            html += f'<td{style}>{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)


def _styled_df(df, max_height=None):
    """Convert a pandas DataFrame to a premium styled HTML table."""
    rows = df.to_dict('records')
    columns = list(df.columns)
    _styled_table(rows, columns, max_height)


# ──────────────────────────── Data Loading ────────────────────────────

@st.cache_data
def load_teams():
    m = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
    w = pd.read_csv(os.path.join(DATA_DIR, "WTeams.csv"))
    return dict(zip(m.TeamID, m.TeamName)) | dict(zip(w.TeamID, w.TeamName))


@st.cache_data
def load_predictions():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
    sub = pd.read_csv(path)
    preds = {}
    for _, row in sub.iterrows():
        parts = row["ID"].split("_")
        preds[(int(parts[0]), int(parts[1]), int(parts[2]))] = row["Pred"]
    return preds


@st.cache_data
def load_seeds():
    m = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    w = pd.read_csv(os.path.join(DATA_DIR, "WNCAATourneySeeds.csv"))
    seeds = pd.concat([m, w])
    seeds = seeds[seeds.Season == SEASON].copy()
    seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    seeds["Region"] = seeds["Seed"].str[0]
    return seeds


@st.cache_data
def load_slots():
    m = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySlots.csv"))
    w = pd.read_csv(os.path.join(DATA_DIR, "WNCAATourneySlots.csv"))
    return m[m.Season == SEASON].copy(), w[w.Season == SEASON].copy()


@st.cache_data
def load_conferences():
    tc_m = pd.read_csv(os.path.join(DATA_DIR, "MTeamConferences.csv"))
    tc_w = pd.read_csv(os.path.join(DATA_DIR, "WTeamConferences.csv"))
    tc = pd.concat([tc_m, tc_w])
    tc = tc[tc.Season == SEASON]
    return dict(zip(tc.TeamID, tc.ConfAbbrev))


# ──────────────────────────── Coach Data ────────────────────────────

@st.cache_data
def load_coach_data():
    """Load coach info: current coach per team + tournament history."""
    coaches = pd.read_csv(os.path.join(DATA_DIR, "MTeamCoaches.csv"))
    trn = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))

    # Current coach per team (2026)
    c26 = coaches[coaches.Season == SEASON].sort_values("LastDayNum").groupby("TeamID").last().reset_index()
    current = dict(zip(c26.TeamID, c26.CoachName))

    # Coach name → all-time coaching map by (season, team)
    end_coaches = coaches.sort_values("LastDayNum").groupby(["Season", "TeamID"]).last().reset_index()
    coach_map = dict(zip(zip(end_coaches.Season, end_coaches.TeamID), end_coaches.CoachName))

    # Tenure: consecutive seasons with same team
    tenure = {}
    for tid, coach in current.items():
        t = 0
        for yr in range(SEASON, 1984, -1):
            if coach_map.get((yr, tid)) == coach:
                t += 1
            else:
                break
        tenure[tid] = t

    # Tournament record per coach (all-time prior to 2026)
    trn_pre = trn[trn.Season < SEASON]
    coach_wins = {}
    coach_games = {}
    for _, r in trn_pre.iterrows():
        wc = coach_map.get((r.Season, r.WTeamID))
        lc = coach_map.get((r.Season, r.LTeamID))
        if wc:
            coach_wins[wc] = coach_wins.get(wc, 0) + 1
            coach_games[wc] = coach_games.get(wc, 0) + 1
        if lc:
            coach_games[lc] = coach_games.get(lc, 0) + 1

    def fmt_name(raw):
        return " ".join(w.capitalize() for w in raw.replace("_", " ").split())

    coach_info = {}
    for tid, raw_name in current.items():
        name = fmt_name(raw_name)
        w = coach_wins.get(raw_name, 0)
        g = coach_games.get(raw_name, 0)
        coach_info[tid] = {
            "name": name, "tenure": tenure.get(tid, 1),
            "tourney_wins": w, "tourney_games": g,
            "tourney_pct": round(w / g * 100, 1) if g > 0 else 0,
        }
    return coach_info


# ──────────────────────────── KNN Similar Opponents ────────────────────────────

KNN_PROFILE_COLS = [
    "WinPct", "PPG", "OppPPG", "FGPct", "FG3Pct", "FTPct",
    "RPG", "APG", "TOPG", "SPG", "BPG", "Margin",
]

@st.cache_data
def build_knn_lookup(prefix):
    """Build KNN for finding similar teams + game results lookup."""
    stats = compute_season_stats(prefix)
    det = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonDetailedResults.csv"))
    det = det[det.Season == SEASON]

    # Profile all teams
    profile_df = stats[KNN_PROFILE_COLS].fillna(0)
    team_ids = profile_df.index.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profile_df.values)

    nn = NearestNeighbors(n_neighbors=min(6, len(team_ids)), metric="euclidean")
    nn.fit(scaled)
    dists, indices = nn.kneighbors(scaled)

    # Map: team_id -> list of (neighbor_id, distance)
    neighbors_map = {}
    for i, tid in enumerate(team_ids):
        neighbors_map[tid] = [
            (int(team_ids[j]), round(float(dists[i][k]), 2))
            for k, j in enumerate(indices[i]) if team_ids[j] != tid
        ][:5]

    # Game results lookup: (team, opp) -> list of (win, margin)
    game_results = {}
    for _, r in det.iterrows():
        w, l = int(r.WTeamID), int(r.LTeamID)
        margin = int(r.WScore - r.LScore)
        game_results.setdefault((w, l), []).append((1, margin))
        game_results.setdefault((l, w), []).append((0, -margin))

    return neighbors_map, game_results


# ──────────────────────────── Head-to-Head History ────────────────────────────

@st.cache_data
def build_h2h_history(prefix):
    """All-time H2H records between team pairs."""
    reg = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonCompactResults.csv"))
    try:
        trn = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneyCompactResults.csv"))
        df = pd.concat([reg, trn])
    except FileNotFoundError:
        df = reg
    df = df[df.Season < SEASON]  # only prior seasons

    h2h = {}
    for _, r in df.iterrows():
        w, l = int(r.WTeamID), int(r.LTeamID)
        low, high = min(w, l), max(w, l)
        if (low, high) not in h2h:
            h2h[(low, high)] = {"low_wins": 0, "high_wins": 0, "games": []}
        if w == low:
            h2h[(low, high)]["low_wins"] += 1
        else:
            h2h[(low, high)]["high_wins"] += 1
        h2h[(low, high)]["games"].append({
            "season": int(r.Season), "winner": w, "loser": l,
            "score": f"{int(r.WScore)}-{int(r.LScore)}",
        })
    return h2h


# ──────────────────────────── Seed Matchup History ────────────────────────────

@st.cache_data
def build_seed_history(prefix):
    """Historical win rates by seed matchup in NCAA tournament."""
    seeds_all = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneySeeds.csv"))
    trn = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneyCompactResults.csv"))
    trn = trn[trn.Season < SEASON]

    seed_map = {}
    for _, r in seeds_all.iterrows():
        seed_map[(r.Season, r.TeamID)] = int(r.Seed[1:3])

    matchups = {}  # (low_seed, high_seed) -> [low_seed_wins, total]
    for _, r in trn.iterrows():
        ws = seed_map.get((r.Season, r.WTeamID))
        ls = seed_map.get((r.Season, r.LTeamID))
        if ws is None or ls is None:
            continue
        low_s, high_s = min(ws, ls), max(ws, ls)
        if (low_s, high_s) not in matchups:
            matchups[(low_s, high_s)] = [0, 0]
        matchups[(low_s, high_s)][1] += 1
        if ws == low_s:
            matchups[(low_s, high_s)][0] += 1

    return matchups


# ──────────────────────────── Elo Computation ────────────────────────────

@st.cache_data
def compute_elo(prefix):
    reg = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonCompactResults.csv"))
    try:
        trn = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneyCompactResults.csv"))
        df = pd.concat([reg, trn])
    except FileNotFoundError:
        df = reg
    df = df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    K, HOME_ADV, DECAY = 20, 100, 0.75
    elo = {}
    seasons = df["Season"].values
    w_ids = df["WTeamID"].values
    l_ids = df["LTeamID"].values
    w_scores = df["WScore"].values
    l_scores = df["LScore"].values
    has_loc = "WLoc" in df.columns
    w_locs = df["WLoc"].values if has_loc else np.full(len(df), "N")

    prev_season = None
    for i in range(len(df)):
        season = int(seasons[i])
        if season != prev_season:
            if prev_season is not None:
                for tid in elo:
                    elo[tid] = 1500 + DECAY * (elo[tid] - 1500)
            prev_season = season

        w_id = int(w_ids[i])
        l_id = int(l_ids[i])
        w_elo = elo.get(w_id, 1500)
        l_elo = elo.get(l_id, 1500)
        loc = str(w_locs[i]) if has_loc else "N"
        w_adj = w_elo + (HOME_ADV if loc == "H" else 0)
        l_adj = l_elo + (HOME_ADV if loc == "A" else 0)

        exp_w = 1.0 / (1.0 + 10 ** ((l_adj - w_adj) / 400))
        mov = int(w_scores[i]) - int(l_scores[i])
        diff = abs(w_adj - l_adj)
        damp = 2.2 / (diff * 0.001 + 2.2)
        k_adj = K * math.log(max(mov, 1) + 1) * 0.7 * damp

        elo[w_id] = w_elo + k_adj * (1 - exp_w)
        elo[l_id] = l_elo - k_adj * (1 - exp_w)

    return elo


# ──────────────────────────── Season Stats ────────────────────────────

@st.cache_data
def compute_season_stats(prefix):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonDetailedResults.csv"))
    df = df[df.Season == SEASON]

    records = []
    for _, r in df.iterrows():
        for side, opp_side, win in [("W", "L", 1), ("L", "W", 0)]:
            records.append({
                "TeamID": int(r[f"{side}TeamID"]), "Win": win,
                "Score": r[f"{side}Score"], "OppScore": r[f"{opp_side}Score"],
                "FGM": r[f"{side}FGM"], "FGA": r[f"{side}FGA"],
                "FGM3": r[f"{side}FGM3"], "FGA3": r[f"{side}FGA3"],
                "FTM": r[f"{side}FTM"], "FTA": r[f"{side}FTA"],
                "OR": r[f"{side}OR"], "DR": r[f"{side}DR"],
                "Ast": r[f"{side}Ast"], "TO": r[f"{side}TO"],
                "Stl": r[f"{side}Stl"], "Blk": r[f"{side}Blk"],
                "PF": r[f"{side}PF"],
                "OppFGM": r[f"{opp_side}FGM"], "OppFGA": r[f"{opp_side}FGA"],
                "OppFGM3": r[f"{opp_side}FGM3"], "OppOR": r[f"{opp_side}OR"],
                "OppTO": r[f"{opp_side}TO"], "OppFTA": r[f"{opp_side}FTA"],
            })

    tg = pd.DataFrame(records)
    stats = tg.groupby("TeamID").agg(
        Games=("Win", "count"), Wins=("Win", "sum"),
        PPG=("Score", "mean"), OppPPG=("OppScore", "mean"),
        FGM=("FGM", "mean"), FGA=("FGA", "mean"),
        FGM3=("FGM3", "mean"), FGA3=("FGA3", "mean"),
        FTM=("FTM", "mean"), FTA=("FTA", "mean"),
        ORB=("OR", "mean"), DRB=("DR", "mean"),
        APG=("Ast", "mean"), TOPG=("TO", "mean"),
        SPG=("Stl", "mean"), BPG=("Blk", "mean"),
        PF=("PF", "mean"),
        OppFGM=("OppFGM", "mean"), OppFGA=("OppFGA", "mean"),
        OppFGM3=("OppFGM3", "mean"), OppOR=("OppOR", "mean"),
        OppTO=("OppTO", "mean"), OppFTA=("OppFTA", "mean"),
    )

    stats["WinPct"] = stats["Wins"] / stats["Games"]
    stats["Margin"] = stats["PPG"] - stats["OppPPG"]
    stats["FGPct"] = (stats["FGM"] / stats["FGA"] * 100).round(1)
    stats["FG3Pct"] = (stats["FGM3"] / stats["FGA3"] * 100).round(1)
    stats["FTPct"] = (stats["FTM"] / stats["FTA"] * 100).round(1)
    stats["RPG"] = stats["ORB"] + stats["DRB"]
    stats["EffFGPct"] = ((stats["FGM"] + 0.5 * stats["FGM3"]) / stats["FGA"] * 100).round(1)

    stats["Poss"] = stats["FGA"] - stats["ORB"] + stats["TOPG"] + 0.475 * stats["FTA"]
    stats["OppPoss"] = stats["OppFGA"] - stats["OppOR"] + stats["OppTO"] + 0.475 * stats["OppFTA"]
    stats["OffEff"] = (stats["PPG"] / stats["Poss"] * 100).round(1)
    stats["DefEff"] = (stats["OppPPG"] / stats["OppPoss"] * 100).round(1)
    stats["NetEff"] = (stats["OffEff"] - stats["DefEff"]).round(1)
    stats["Tempo"] = ((stats["Poss"] + stats["OppPoss"]) / 2).round(1)

    return stats


# ──────────────────────────── Utilities ────────────────────────────

def get_pred(preds, t1, t2, season=SEASON):
    low, high = min(t1, t2), max(t1, t2)
    p = preds.get((season, low, high), 0.5)
    return p if t1 == low else 1 - p


def tname(teams, tid):
    return teams.get(tid, f"Team {tid}")


def seed_for_team(seeds_df, tid):
    row = seeds_df[seeds_df.TeamID == tid]
    if len(row) > 0:
        return int(row.iloc[0]["SeedNum"])
    return None


# ──────────────────────────── Betting Lines ────────────────────────────

def prob_to_moneyline(p):
    """Convert win probability to American moneyline odds."""
    if p <= 0.0:
        return "+9999"
    if p >= 1.0:
        return "-9999"
    if p >= 0.5:
        odds = -100 * p / (1 - p)
        return f"{int(round(odds))}"
    else:
        odds = 100 * (1 - p) / p
        return f"+{int(round(odds))}"


def prob_to_spread(p):
    """Convert win probability to projected point spread using logit scaling."""
    if p <= 0.01:
        return 30.0
    if p >= 0.99:
        return -30.0
    logit = math.log(p / (1 - p))
    spread = -logit * 4.8  # ~4.8 points per logit unit (NCAA calibration)
    return round(spread * 2) / 2  # round to nearest 0.5


def project_game_total(stats, t1, t2):
    """Project total points scored using efficiency and tempo."""
    if t1 not in stats.index or t2 not in stats.index:
        return 140.0
    s1, s2 = stats.loc[t1], stats.loc[t2]
    avg_tempo = (s1.Tempo + s2.Tempo) / 2
    # Blend team's offense with opponent's defense to get expected efficiency
    t1_pts = (s1.OffEff + s2.DefEff) / 2 * avg_tempo / 100
    t2_pts = (s2.OffEff + s1.DefEff) / 2 * avg_tempo / 100
    return round((t1_pts + t2_pts) * 2) / 2  # round to 0.5


def project_team_points(stats, team, opp):
    """Project individual team points."""
    if team not in stats.index or opp not in stats.index:
        return 70.0
    st, so = stats.loc[team], stats.loc[opp]
    avg_tempo = (st.Tempo + so.Tempo) / 2
    pts = (st.OffEff + so.DefEff) / 2 * avg_tempo / 100
    return round(pts * 2) / 2


def project_team_props(stats, team, opp):
    """Project team stat props (rebounds, assists, 3PM, etc.)."""
    if team not in stats.index or opp not in stats.index:
        return {}
    st, so = stats.loc[team], stats.loc[opp]
    # Adjust stats based on matchup — blend team avg with opponent allowed avg
    props = {
        "Points": round(project_team_points(stats, team, opp), 1),
        "Total Rebounds": round((st.RPG + (st.RPG * (so.RPG / stats["RPG"].mean()))) / 2, 1),
        "Assists": round((st.APG + (st.APG * (so.OppPPG / stats["OppPPG"].mean()))) / 2, 1),
        "3-Pointers Made": round(st.FGM3, 1),
        "Steals": round((st.SPG + (st.SPG * (so.TOPG / stats["TOPG"].mean()))) / 2, 1),
        "Blocks": round(st.BPG, 1),
        "Turnovers": round((st.TOPG + (st.TOPG * (so.SPG / stats["SPG"].mean()))) / 2, 1),
    }
    return props


def compute_betting_lines(stats, preds, t1, t2):
    """Compute full betting line package for a matchup."""
    p = get_pred(preds, t1, t2)
    spread = prob_to_spread(p)
    total = project_game_total(stats, t1, t2)
    t1_pts = project_team_points(stats, t1, t2)
    t2_pts = project_team_points(stats, t2, t1)

    return {
        "t1_prob": p,
        "t2_prob": 1 - p,
        "t1_ml": prob_to_moneyline(p),
        "t2_ml": prob_to_moneyline(1 - p),
        "spread": spread,  # negative = t1 favored
        "total": total,
        "t1_pts": t1_pts,
        "t2_pts": t2_pts,
        "t1_props": project_team_props(stats, t1, t2),
        "t2_props": project_team_props(stats, t2, t1),
    }


# ──────────────────────────── Bracket Simulation ────────────────────────────

def slot_sort_key(slot):
    if slot.startswith("R"):
        return int(slot[1])
    return 0  # play-in games first


def simulate_bracket(seeds_df, slots_df, preds, deterministic=True):
    seed_to_team = dict(zip(seeds_df.Seed, seeds_df.TeamID))
    sorted_slots = slots_df.sort_values(
        "Slot", key=lambda x: x.map(slot_sort_key)
    )

    results = {}
    slot_winners = {}

    for _, row in sorted_slots.iterrows():
        slot = row["Slot"]
        t1 = slot_winners.get(row["StrongSeed"]) or seed_to_team.get(row["StrongSeed"])
        t2 = slot_winners.get(row["WeakSeed"]) or seed_to_team.get(row["WeakSeed"])
        if t1 is None or t2 is None:
            continue

        p = get_pred(preds, t1, t2)
        if deterministic:
            winner = t1 if p >= 0.5 else t2
        else:
            winner = t1 if np.random.random() < p else t2
        loser = t2 if winner == t1 else t1

        slot_winners[slot] = winner
        results[slot] = {
            "winner": winner, "loser": loser,
            "t1": t1, "t2": t2,
            "t1_prob": p, "prob": p if winner == t1 else 1 - p,
        }
    return results, slot_winners


# ──────────────────────────── Monte Carlo Odds ────────────────────────────

@st.cache_data
def monte_carlo_odds(_seeds_df, _slots_df, _preds, n_sims=10000, eliminated=None):
    """Run Monte Carlo bracket simulations. Eliminated teams (frozenset) get 0% chance."""
    eliminated = eliminated or frozenset()
    seed_to_team = dict(zip(_seeds_df.Seed, _seeds_df.TeamID))
    sorted_slots = _slots_df.sort_values(
        "Slot", key=lambda x: x.map(slot_sort_key)
    )
    slots_list = sorted_slots.to_dict("records")
    all_teams = set(seed_to_team.values())

    round_labels = ["R64", "R32", "S16", "E8", "FF", "CG", "Champ"]
    counts = {tid: [0] * 7 for tid in all_teams}
    for tid in all_teams:
        counts[tid][0] = n_sims  # all teams make R64

    for _ in range(n_sims):
        sw = {}
        for row in slots_list:
            slot = row["Slot"]
            t1 = sw.get(row["StrongSeed"]) or seed_to_team.get(row["StrongSeed"])
            t2 = sw.get(row["WeakSeed"]) or seed_to_team.get(row["WeakSeed"])
            if t1 is None or t2 is None:
                continue

            # If one team is eliminated, the other auto-advances
            t1_elim = t1 in eliminated
            t2_elim = t2 in eliminated
            if t1_elim and t2_elim:
                continue  # both eliminated, skip
            elif t1_elim:
                winner = t2
            elif t2_elim:
                winner = t1
            else:
                p = get_pred(_preds, t1, t2)
                winner = t1 if np.random.random() < p else t2

            sw[slot] = winner

            if slot.startswith("R1"):
                counts[winner][1] += 1
            elif slot.startswith("R2"):
                counts[winner][2] += 1
            elif slot.startswith("R3"):
                counts[winner][3] += 1
            elif slot.startswith("R4"):
                counts[winner][4] += 1
            elif slot.startswith("R5"):
                counts[winner][5] += 1
            elif slot.startswith("R6"):
                counts[winner][6] += 1

    probs = {}
    for tid in all_teams:
        probs[tid] = [c / n_sims for c in counts[tid]]
    return probs, round_labels


# ──────────────────────────── Bracket HTML ────────────────────────────

def matchup_html(t1, t2, t1_prob, winner, teams, team_seed_map):
    s1 = team_seed_map.get(t1, "")
    s2 = team_seed_map.get(t2, "")
    n1 = tname(teams, t1)[:18] if t1 else "TBD"
    n2 = tname(teams, t2)[:18] if t2 else "TBD"
    s1_tag = f'<span class="seed-tag">({s1})</span> ' if s1 else ""
    s2_tag = f'<span class="seed-tag">({s2})</span> ' if s2 else ""
    c1 = "winner" if t1 == winner else ""
    c2 = "winner" if t2 == winner else ""
    p1 = f"{t1_prob*100:.0f}%" if t1 and t2 else ""
    p2 = f"{(1-t1_prob)*100:.0f}%" if t1 and t2 else ""

    return f"""<div class="matchup">
  <div class="team-slot {c1}">{s1_tag}{n1}<span class="prob-tag">{p1}</span></div>
  <div class="team-slot {c2}">{s2_tag}{n2}<span class="prob-tag">{p2}</span></div>
</div>"""


def region_bracket_html(region, sim_results, teams, team_seed_map):
    r1_order = [1, 8, 5, 4, 6, 3, 7, 2]
    r2_order = [1, 4, 3, 2]
    r3_order = [1, 2]
    r4_order = [1]

    rounds_config = [
        ("Round of 64", [f"R1{region}{n}" for n in r1_order]),
        ("Round of 32", [f"R2{region}{n}" for n in r2_order]),
        ("Sweet 16", [f"R3{region}{n}" for n in r3_order]),
        ("Elite 8", [f"R4{region}{n}" for n in r4_order]),
    ]

    html = '<div class="bracket">'
    for title, slots in rounds_config:
        html += f'<div class="round"><div class="round-title">{title}</div>'
        for slot in slots:
            r = sim_results.get(slot, {})
            html += matchup_html(
                r.get("t1"), r.get("t2"), r.get("t1_prob", 0.5),
                r.get("winner"), teams, team_seed_map,
            )
        html += "</div>"
    html += "</div>"
    return html


def final_four_html(sim_results, teams, team_seed_map):
    sf_slots = ["R5WX", "R5YZ"]
    ch_slot = "R6CH"

    html = '<div class="ff-bracket">'

    # Semis
    html += '<div class="ff-round">'
    html += '<div class="round-title">Final Four</div>'
    for slot in sf_slots:
        r = sim_results.get(slot, {})
        html += matchup_html(
            r.get("t1"), r.get("t2"), r.get("t1_prob", 0.5),
            r.get("winner"), teams, team_seed_map,
        )
    html += "</div>"

    # Championship
    html += '<div class="ff-round">'
    html += '<div class="round-title">Championship</div>'
    r = sim_results.get(ch_slot, {})
    html += matchup_html(
        r.get("t1"), r.get("t2"), r.get("t1_prob", 0.5),
        r.get("winner"), teams, team_seed_map,
    )
    html += "</div>"

    # Champion
    champ = sim_results.get(ch_slot, {}).get("winner")
    if champ:
        s = team_seed_map.get(champ, "")
        seed_txt = f" ({s})" if s else ""
        html += f'<div class="ff-round"><div class="champ-banner">'
        html += f'\U0001f3c6 {tname(teams, champ)}{seed_txt}'
        html += "</div></div>"

    html += "</div>"
    return html


# ──────────────────────────── Page: Rankings ────────────────────────────

def _tier_badge(elo_val, max_elo):
    """Return a tier badge based on Elo percentile."""
    pct = (elo_val - 1200) / max(max_elo - 1200, 1)
    if pct >= 0.90:
        return '<span class="tier-elite">ELITE</span>'
    elif pct >= 0.70:
        return '<span class="tier-strong">STRONG</span>'
    elif pct >= 0.45:
        return '<span class="tier-solid">SOLID</span>'
    else:
        return '<span class="tier-avg">AVG</span>'


def _eff_color(val, higher_better=True):
    """Return CSS color class for efficiency values."""
    if higher_better:
        return "stat-good" if val > 5 else "stat-bad" if val < -5 else "stat-neutral"
    else:
        return "stat-good" if val < 0 else "stat-bad" if val > 5 else "stat-neutral"


def page_rankings(prefix, teams, seeds_df, conferences):
    st.markdown(
        f'<div class="vp-section">{"MEN" if prefix == "M" else "WOMEN"}\'S POWER RANKINGS</div>',
        unsafe_allow_html=True,
    )

    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))

    rows = []
    for tid in stats.index:
        s = stats.loc[tid]
        rows.append({
            "tid": tid,
            "Team": tname(teams, tid),
            "Conf": conferences.get(tid, ""),
            "Seed": team_seeds.get(tid, ""),
            "Record": f"{int(s.Wins)}-{int(s.Games - s.Wins)}",
            "Elo": int(elo.get(tid, 1500)),
            "NetEff": round(s.NetEff, 1),
            "OffEff": round(s.OffEff, 1),
            "DefEff": round(s.DefEff, 1),
            "Tempo": round(s.Tempo, 1),
            "PPG": round(s.PPG, 1),
            "OppPPG": round(s.OppPPG, 1),
            "Margin": round(s.Margin, 1),
            "eFG%": round(s.EffFGPct, 1),
            "FG%": round(s.FGPct, 1),
            "3P%": round(s.FG3Pct, 1),
            "FT%": round(s.FTPct, 1),
        })

    rows.sort(key=lambda r: r["Elo"], reverse=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        show_tourney_only = st.checkbox("Tournament teams only", value=False)
        conf_filter = st.selectbox("Conference", ["All"] + sorted(set(r["Conf"] for r in rows if r["Conf"])))
    with col2:
        search = st.text_input("Search team", "")

    if show_tourney_only:
        rows = [r for r in rows if r["Seed"] != ""]
    if conf_filter != "All":
        rows = [r for r in rows if r["Conf"] == conf_filter]
    if search:
        rows = [r for r in rows if search.lower() in r["Team"].lower()]

    max_elo = max((r["Elo"] for r in rows), default=1700)

    # Build premium HTML table
    html = '<div style="max-height:650px; overflow-y:auto; border-radius:10px; border:1px solid #2a2d36;">'
    html += '<table class="vp-table"><thead><tr>'
    html += '<th style="width:36px;">#</th><th>TEAM</th><th>CONF</th><th>SEED</th>'
    html += '<th>RECORD</th><th>TIER</th><th>ELO</th>'
    html += '<th>NET EFF</th><th>OFF EFF</th><th>DEF EFF</th><th>TEMPO</th>'
    html += '<th>PPG</th><th>OPP PPG</th><th>MARGIN</th>'
    html += '<th>eFG%</th><th>FG%</th><th>3P%</th><th>FT%</th>'
    html += '</tr></thead><tbody>'

    for i, r in enumerate(rows, 1):
        seed_html = f'<span class="seed-badge">{r["Seed"]}</span>' if r["Seed"] else ""
        tier_html = _tier_badge(r["Elo"], max_elo)
        net_cls = _eff_color(r["NetEff"])
        off_cls = _eff_color(r["OffEff"])
        def_cls = _eff_color(r["DefEff"], higher_better=False)
        margin_cls = "stat-good" if r["Margin"] > 0 else "stat-bad" if r["Margin"] < 0 else "stat-neutral"

        html += f'<tr>'
        html += f'<td class="rank-cell">{i}</td>'
        html += f'<td class="team-cell">{r["Team"]}</td>'
        html += f'<td style="color:#888;">{r["Conf"]}</td>'
        html += f'<td>{seed_html}</td>'
        html += f'<td>{r["Record"]}</td>'
        html += f'<td>{tier_html}</td>'
        html += f'<td style="font-weight:700;">{r["Elo"]}</td>'
        html += f'<td class="{net_cls}" style="font-weight:600;">{r["NetEff"]:+.1f}</td>'
        html += f'<td class="{off_cls}">{r["OffEff"]:.1f}</td>'
        html += f'<td class="{def_cls}">{r["DefEff"]:.1f}</td>'
        html += f'<td style="color:#aaa;">{r["Tempo"]:.1f}</td>'
        html += f'<td>{r["PPG"]}</td>'
        html += f'<td>{r["OppPPG"]}</td>'
        html += f'<td class="{margin_cls}" style="font-weight:600;">{r["Margin"]:+.1f}</td>'
        html += f'<td>{r["eFG%"]:.1f}</td>'
        html += f'<td>{r["FG%"]:.1f}</td>'
        html += f'<td>{r["3P%"]:.1f}</td>'
        html += f'<td>{r["FT%"]:.1f}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Showing {len(rows)} teams")


# ──────────────────────────── Page: Head-to-Head ────────────────────────────

def page_h2h(prefix, teams, seeds_df, preds, coach_info, knn_data, h2h_history, seed_history):
    st.markdown('<div class="vp-section">HEAD-TO-HEAD MATCHUP</div>', unsafe_allow_html=True)

    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)
    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    neighbors_map, game_results = knn_data

    id_range = range(1101, 1482) if prefix == "M" else range(3101, 3482)
    available = [tid for tid in id_range if tid in stats.index]
    team_options = sorted(available, key=lambda t: elo.get(t, 1500), reverse=True)
    labels = {t: f"{tname(teams, t)} (Elo: {int(elo.get(t, 1500))})" for t in team_options}

    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox("Team A", team_options, format_func=lambda t: labels[t], index=0)
    with col2:
        t2 = st.selectbox("Team B", team_options, format_func=lambda t: labels[t], index=1)

    if t1 == t2:
        st.warning("Select two different teams.")
        return

    p = get_pred(preds, t1, t2)

    # Big probability display
    st.markdown("---")
    n1, n2 = tname(teams, t1), tname(teams, t2)
    s1_seed = team_seeds.get(t1)
    s2_seed = team_seeds.get(t2)
    s1_txt = f'<span style="color:#FF6B35; font-size:13px; font-weight:700;">({s1_seed})</span> ' if s1_seed else ''
    s2_txt = f' <span style="color:#FF6B35; font-size:13px; font-weight:700;">({s2_seed})</span>' if s2_seed else ''
    color1 = "#4ade80" if p >= 0.5 else "#f87171"
    color2 = "#4ade80" if (1 - p) >= 0.5 else "#f87171"
    fav_glow = f"0 0 30px rgba({','.join(str(int(color1[i:i+2], 16)) for i in (1,3,5))}, 0.15)"

    st.markdown(
        f'<div style="display:flex; align-items:center; justify-content:center; gap:20px; margin:20px 0;">'
        # Team 1
        f'<div class="vp-metric" style="flex:1; border-top:3px solid {color1};">'
        f'<div class="big-prob" style="color:{color1}; margin:4px 0;">{p*100:.1f}%</div>'
        f'<div style="text-align:center; font-size:18px; font-weight:700;">{s1_txt}{n1}</div>'
        f'</div>'
        # VS
        f'<div style="text-align:center; padding:0 8px;">'
        f'<div style="font-size:12px; color:#444; font-weight:700; letter-spacing:2px;">VS</div>'
        f'</div>'
        # Team 2
        f'<div class="vp-metric" style="flex:1; border-top:3px solid {color2};">'
        f'<div class="big-prob" style="color:{color2}; margin:4px 0;">{(1-p)*100:.1f}%</div>'
        f'<div style="text-align:center; font-size:18px; font-weight:700;">{n2}{s2_txt}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Probability bar
    st.markdown(
        f'<div style="display:flex; height:8px; border-radius:4px; overflow:hidden; margin:8px 0 16px;">'
        f'<div style="width:{p*100}%; background:linear-gradient(90deg, {color1}, {color1}88);"></div>'
        f'<div style="width:{(1-p)*100}%; background:linear-gradient(90deg, {color2}88, {color2});"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Projected Betting Lines ──
    st.markdown("---")
    st.subheader("Projected Betting Lines")
    st.caption("Projected lines from our ensemble model. For entertainment purposes only.")

    lines = compute_betting_lines(stats, preds, t1, t2)
    n1, n2 = tname(teams, t1), tname(teams, t2)
    fav = t1 if p >= 0.5 else t2
    fav_name = n1 if fav == t1 else n2
    dog_name = n2 if fav == t1 else n1
    spread_display = lines["spread"]

    # Main lines in columns
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("**Moneyline**")
        ml1_color = "#4ade80" if p >= 0.5 else "#f87171"
        ml2_color = "#4ade80" if p < 0.5 else "#f87171"
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:12px; margin:4px 0;">'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #2a2d34;">'
            f'<span>{n1}</span><span style="font-weight:700; color:{ml1_color};">{lines["t1_ml"]}</span></div>'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0;">'
            f'<span>{n2}</span><span style="font-weight:700; color:{ml2_color};">{lines["t2_ml"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with lc2:
        st.markdown("**Spread**")
        if spread_display <= 0:
            s1_spread = f"{spread_display:+.1f}"
            s2_spread = f"{-spread_display:+.1f}"
        else:
            s1_spread = f"+{spread_display:.1f}"
            s2_spread = f"{-spread_display:.1f}"
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:12px; margin:4px 0;">'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #2a2d34;">'
            f'<span>{n1}</span><span style="font-weight:700; color:#60a5fa;">{s1_spread}</span></div>'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0;">'
            f'<span>{n2}</span><span style="font-weight:700; color:#60a5fa;">{s2_spread}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with lc3:
        st.markdown("**Game Total (O/U)**")
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:12px; margin:4px 0;">'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #2a2d34;">'
            f'<span>Over</span><span style="font-weight:700; color:#fbbf24;">{lines["total"]:.1f}</span></div>'
            f'<div style="display:flex; justify-content:space-between; padding:6px 0;">'
            f'<span>Under</span><span style="font-weight:700; color:#fbbf24;">{lines["total"]:.1f}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Projected score
    st.markdown(
        f'<div style="text-align:center; margin:12px 0; font-size:20px; font-weight:700; color:#e2e8f0;">'
        f'Projected Score: {n1} {lines["t1_pts"]:.0f} — {n2} {lines["t2_pts"]:.0f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Team Props
    st.markdown("**Team Props**")
    tp1, tp2 = st.columns(2)

    for col, tid, team_name, props in [
        (tp1, t1, n1, lines["t1_props"]),
        (tp2, t2, n2, lines["t2_props"]),
    ]:
        with col:
            if props:
                prop_rows = ""
                for prop_name, val in props.items():
                    prop_rows += (
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:5px 0; border-bottom:1px solid #2a2d34;">'
                        f'<span style="color:#aaa;">{prop_name}</span>'
                        f'<span style="font-weight:600; color:#e2e8f0;">{val}</span></div>'
                    )
                st.markdown(
                    f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:12px;">'
                    f'<div style="font-weight:700; margin-bottom:8px; color:#60a5fa;">{team_name}</div>'
                    f'{prop_rows}</div>',
                    unsafe_allow_html=True,
                )

    # Stat comparison
    st.markdown("---")
    st.subheader("Stat Comparison")

    if t1 in stats.index and t2 in stats.index:
        s1_stats = stats.loc[t1]
        s2_stats = stats.loc[t2]

        compare_stats = [
            ("Elo", int(elo.get(t1, 1500)), int(elo.get(t2, 1500)), True),
            ("Record", f"{int(s1_stats.Wins)}-{int(s1_stats.Games-s1_stats.Wins)}",
             f"{int(s2_stats.Wins)}-{int(s2_stats.Games-s2_stats.Wins)}", None),
            ("PPG", round(s1_stats.PPG, 1), round(s2_stats.PPG, 1), True),
            ("Opp PPG", round(s1_stats.OppPPG, 1), round(s2_stats.OppPPG, 1), False),
            ("Margin", round(s1_stats.Margin, 1), round(s2_stats.Margin, 1), True),
            ("Off Eff", s1_stats.OffEff, s2_stats.OffEff, True),
            ("Def Eff", s1_stats.DefEff, s2_stats.DefEff, False),
            ("Net Eff", s1_stats.NetEff, s2_stats.NetEff, True),
            ("eFG%", s1_stats.EffFGPct, s2_stats.EffFGPct, True),
            ("FG%", s1_stats.FGPct, s2_stats.FGPct, True),
            ("3P%", s1_stats.FG3Pct, s2_stats.FG3Pct, True),
            ("FT%", s1_stats.FTPct, s2_stats.FTPct, True),
            ("RPG", round(s1_stats.RPG, 1), round(s2_stats.RPG, 1), True),
            ("APG", round(s1_stats.APG, 1), round(s2_stats.APG, 1), True),
            ("TOPG", round(s1_stats.TOPG, 1), round(s2_stats.TOPG, 1), False),
            ("SPG", round(s1_stats.SPG, 1), round(s2_stats.SPG, 1), True),
            ("BPG", round(s1_stats.BPG, 1), round(s2_stats.BPG, 1), True),
            ("Tempo", s1_stats.Tempo, s2_stats.Tempo, None),
        ]

        rows_html = ""
        for label, v1, v2, higher_better in compare_stats:
            if higher_better is not None and isinstance(v1, (int, float)):
                if higher_better:
                    c1_cls = "stat-good" if v1 > v2 else "stat-bad" if v1 < v2 else "stat-neutral"
                    c2_cls = "stat-good" if v2 > v1 else "stat-bad" if v2 < v1 else "stat-neutral"
                    # advantage dot
                    dot1 = '<span style="color:#4ade80; margin-left:6px;">&#9679;</span>' if v1 > v2 else ''
                    dot2 = '<span style="color:#4ade80; margin-right:6px;">&#9679;</span>' if v2 > v1 else ''
                else:
                    c1_cls = "stat-good" if v1 < v2 else "stat-bad" if v1 > v2 else "stat-neutral"
                    c2_cls = "stat-good" if v2 < v1 else "stat-bad" if v2 > v1 else "stat-neutral"
                    dot1 = '<span style="color:#4ade80; margin-left:6px;">&#9679;</span>' if v1 < v2 else ''
                    dot2 = '<span style="color:#4ade80; margin-right:6px;">&#9679;</span>' if v2 < v1 else ''
            else:
                c1_cls = c2_cls = "stat-neutral"
                dot1 = dot2 = ''

            rows_html += (
                f'<tr>'
                f'<td style="text-align:right; padding:6px 14px; font-variant-numeric:tabular-nums;" '
                f'class="{c1_cls}"><span style="font-weight:600;">{v1}</span>{dot1}</td>'
                f'<td style="text-align:center; padding:6px 14px; color:#FF6B35; font-weight:700; '
                f'font-size:11px; letter-spacing:0.5px; text-transform:uppercase;">{label}</td>'
                f'<td style="text-align:left; padding:6px 14px; font-variant-numeric:tabular-nums;" '
                f'class="{c2_cls}">{dot2}<span style="font-weight:600;">{v2}</span></td>'
                f'</tr>'
            )

        st.markdown(
            f'<div style="border-radius:10px; border:1px solid #2a2d36; overflow:hidden; max-width:560px; margin:auto;">'
            f'<table class="vp-table" style="margin:0;">'
            f'<thead><tr>'
            f'<th style="text-align:right; width:40%;">{tname(teams, t1)}</th>'
            f'<th style="text-align:center; width:20%; color:#FF6B35;">STAT</th>'
            f'<th style="text-align:left; width:40%;">{tname(teams, t2)}</th>'
            f'</tr></thead><tbody>'
            f'{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ── Coach Comparison (Men's only) ──
    if prefix == "M" and coach_info:
        st.markdown("---")
        st.subheader("Coach Comparison")
        c1_info = coach_info.get(t1)
        c2_info = coach_info.get(t2)

        cc1, cc2 = st.columns(2)
        for col, tid, info in [(cc1, t1, c1_info), (cc2, t2, c2_info)]:
            with col:
                if info:
                    t_record = (f"{info['tourney_wins']}-{info['tourney_games'] - info['tourney_wins']} "
                                f"({info['tourney_pct']}%)" if info['tourney_games'] > 0
                                else "No tournament games")
                    st.markdown(
                        f'<div class="vp-card">'
                        f'<div style="font-weight:700; color:#FF6B35; font-size:11px; letter-spacing:1px; '
                        f'text-transform:uppercase; margin-bottom:8px;">{tname(teams, tid)}</div>'
                        f'<div style="font-size:18px; font-weight:700; margin-bottom:10px;">{info["name"]}</div>'
                        f'<div style="display:flex; gap:20px;">'
                        f'<div><div style="color:#888; font-size:10px; letter-spacing:0.5px;">TENURE</div>'
                        f'<div style="font-weight:700; font-size:16px;">{info["tenure"]} yr{"s" if info["tenure"] != 1 else ""}</div></div>'
                        f'<div><div style="color:#888; font-size:10px; letter-spacing:0.5px;">NCAA TOURNEY</div>'
                        f'<div style="font-weight:700; font-size:16px;">{t_record}</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="vp-card">'
                        f'<div style="font-weight:700; color:#FF6B35; font-size:11px; letter-spacing:1px; '
                        f'text-transform:uppercase; margin-bottom:8px;">{tname(teams, tid)}</div>'
                        f'<div style="color:#666;">Coach data not available</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Similar Opponents ──
    st.markdown("---")
    st.subheader("Similar Opponents Analysis")
    st.caption(
        "We find the 5 teams most statistically similar to each opponent, "
        "then check how the other team fared against those similar teams this season."
    )

    for team_a, team_b in [(t1, t2), (t2, t1)]:
        st.markdown(f"**How did {tname(teams, team_a)} do vs teams similar to {tname(teams, team_b)}?**")

        similar = neighbors_map.get(team_b, [])
        if not similar:
            st.info(f"No KNN data available for {tname(teams, team_b)}")
            continue

        sim_rows = []
        total_w, total_g, total_margin = 0, 0, 0
        for neighbor_id, dist in similar:
            results = game_results.get((team_a, neighbor_id), [])
            wins = sum(1 for w, _ in results if w == 1)
            games = len(results)
            avg_margin = round(sum(m for _, m in results) / games, 1) if games > 0 else 0
            total_w += wins
            total_g += games
            total_margin += sum(m for _, m in results)
            sim_rows.append({
                "Similar Team": tname(teams, neighbor_id),
                "Distance": dist,
                "Games": games,
                "Record": f"{wins}-{games - wins}" if games > 0 else "Did not play",
                "Avg Margin": f"{avg_margin:+.1f}" if games > 0 else "—",
            })

        sim_df = pd.DataFrame(sim_rows)
        _styled_df(sim_df)

        if total_g > 0:
            avg_m = round(total_margin / total_g, 1)
            st.markdown(
                f"**Overall vs similar teams:** {total_w}-{total_g - total_w} "
                f"(Avg margin: {avg_m:+.1f})"
            )
        else:
            st.markdown("*No games played against similar teams this season.*")
        st.markdown("")

    # ── Head-to-Head History ──
    st.markdown("---")
    st.subheader("All-Time Head-to-Head History")

    low, high = min(t1, t2), max(t1, t2)
    h2h_rec = h2h_history.get((low, high))
    if h2h_rec and h2h_rec["low_wins"] + h2h_rec["high_wins"] > 0:
        t1_wins = h2h_rec["low_wins"] if t1 == low else h2h_rec["high_wins"]
        t2_wins = h2h_rec["high_wins"] if t1 == low else h2h_rec["low_wins"]
        total = t1_wins + t2_wins

        st.markdown(
            f"**{tname(teams, t1)}** leads **{t1_wins}-{t2_wins}** "
            f"({total} games)" if t1_wins >= t2_wins else
            f"**{tname(teams, t2)}** leads **{t2_wins}-{t1_wins}** "
            f"({total} games)"
        )

        # Show last 5 games
        recent = sorted(h2h_rec["games"], key=lambda g: g["season"], reverse=True)[:5]
        if recent:
            st.markdown("**Recent matchups:**")
            for g in recent:
                winner_name = tname(teams, g["winner"])
                st.markdown(f"- {g['season']}: **{winner_name}** won {g['score']}")
    else:
        st.info("These teams have no recorded head-to-head history.")

    # ── Seed Matchup History ──
    s1 = team_seeds.get(t1)
    s2 = team_seeds.get(t2)
    if s1 is not None and s2 is not None:
        st.markdown("---")
        st.subheader("Seed Matchup History")

        low_s, high_s = min(s1, s2), max(s1, s2)
        matchup_rec = seed_history.get((low_s, high_s))
        if matchup_rec and matchup_rec[1] > 0:
            low_wins, total = matchup_rec
            low_pct = round(low_wins / total * 100, 1)
            high_pct = round(100 - low_pct, 1)
            upsets = total - low_wins

            st.markdown(
                f"In NCAA Tournament history, **{low_s}-seeds** have beaten "
                f"**{high_s}-seeds** **{low_pct}%** of the time "
                f"({low_wins}-{total - low_wins} in {total} games)."
            )

            if high_s - low_s >= 4 and upsets > 0:
                st.markdown(
                    f"There have been **{upsets} upset{'s' if upsets != 1 else ''}** "
                    f"by {high_s}-seeds in this matchup."
                )

            # Progress bar for visual
            st.progress(low_pct / 100)
            st.caption(f"{low_s}-seed wins: {low_pct}% | {high_s}-seed wins: {high_pct}%")
        else:
            st.info(f"No historical tournament data for {low_s}-seed vs {high_s}-seed matchup.")

    # ── Elo vs Ensemble ──
    st.markdown("---")
    st.subheader("Elo vs Ensemble Model")
    elo1 = elo.get(t1, 1500)
    elo2 = elo.get(t2, 1500)
    elo_diff = elo1 - elo2
    elo_wp = 1 / (1 + 10 ** (-elo_diff / 400))
    diff_pp = (p - elo_wp) * 100

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(
            f'<div class="vp-metric">'
            f'<div class="label">ELO ONLY</div>'
            f'<div class="value" style="color:#60a5fa;">{elo_wp*100:.1f}%</div>'
            f'<div class="sub">{int(elo1)} vs {int(elo2)}</div>'
            f'</div>', unsafe_allow_html=True)
    with ec2:
        delta_color = "#4ade80" if diff_pp > 0 else "#f87171" if diff_pp < 0 else "#888"
        st.markdown(
            f'<div class="vp-metric">'
            f'<div class="label">FULL ENSEMBLE</div>'
            f'<div class="value" style="color:#FF6B35;">{p*100:.1f}%</div>'
            f'<div class="sub" style="color:{delta_color}; font-weight:600;">{diff_pp:+.1f}pp vs Elo</div>'
            f'</div>', unsafe_allow_html=True)


# ──────────────────────────── Page: Tournament Odds ────────────────────────────

def page_odds(prefix, teams, seeds_df, slots_df, preds):
    st.markdown('<div class="vp-section">CHAMPIONSHIP ODDS</div>', unsafe_allow_html=True)

    with st.spinner("Running 10,000 tournament simulations..."):
        probs, round_labels = monte_carlo_odds(seeds_df, slots_df, preds)

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    elo = compute_elo(prefix)

    rows = []
    for tid, p_list in probs.items():
        rows.append({
            "Team": tname(teams, tid),
            "Seed": team_seeds.get(tid, 99),
            "Elo": int(elo.get(tid, 1500)),
            "R32": f"{p_list[1]*100:.1f}%",
            "S16": f"{p_list[2]*100:.1f}%",
            "E8": f"{p_list[3]*100:.1f}%",
            "FF": f"{p_list[4]*100:.1f}%",
            "CG": f"{p_list[5]*100:.1f}%",
            "Champ": f"{p_list[6]*100:.1f}%",
            "_champ_pct": p_list[6],
        })

    df = pd.DataFrame(rows).sort_values("_champ_pct", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "#"

    # ── Championship Futures / Betting Odds ──
    st.subheader("Championship Futures")
    st.caption("Odds generated from 10,000 simulated tournaments. For entertainment purposes only.")

    futures_rows = []
    for _, row in df.iterrows():
        cp = row["_champ_pct"]
        if cp > 0:
            ml = prob_to_moneyline(cp)
        else:
            ml = "+99999"
        futures_rows.append({
            "Team": row["Team"],
            "Seed": row["Seed"],
            "Champ %": f"{cp*100:.1f}%",
            "Futures Odds": ml,
            "Implied $100 Payout": f"${int(round(100 / cp))}" if cp > 0 else "—",
            "_cp": cp,
        })

    futures_df = pd.DataFrame(futures_rows).sort_values("_cp", ascending=False).reset_index(drop=True)
    futures_df.index = futures_df.index + 1
    futures_df.index.name = "#"

    # Top 20 futures board styled like a sportsbook
    top_futures = futures_df.head(20)
    board_html = '<div style="max-width:700px; margin:auto; border-radius:10px; border:1px solid #2a2d36; overflow:hidden;">'
    # Header row
    board_html += (
        '<div style="display:flex; justify-content:space-between; align-items:center; '
        'padding:10px 16px; background:#12141a; border-bottom:2px solid #FF6B35;">'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px;">TEAM</span>'
        '<div style="display:flex; gap:24px;">'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:60px; text-align:right;">PROB</span>'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:80px; text-align:right;">ODDS</span>'
        '</div></div>'
    )
    for idx, (_, fr) in enumerate(top_futures.iterrows()):
        rank = idx + 1
        seed_txt = f'<span class="seed-badge" style="margin-right:8px;">{int(fr["Seed"])}</span>'
        odds_color = "#FF6B35" if fr["_cp"] >= 0.10 else "#4ade80" if fr["_cp"] >= 0.03 else "#e2e8f0"
        bg = 'rgba(255,107,53,0.04)' if rank <= 4 else '#14161c'
        board_html += (
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:10px 16px; border-bottom:1px solid #1e2028; background:{bg}; '
            f'transition:background 0.15s;" onmouseover="this.style.background=\'rgba(255,107,53,0.08)\'" '
            f'onmouseout="this.style.background=\'{bg}\'">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="color:#FF6B35; font-weight:800; font-size:13px; min-width:20px;">{rank}</span>'
            f'{seed_txt}'
            f'<span style="font-weight:600;">{fr["Team"]}</span></div>'
            f'<div style="display:flex; gap:24px; align-items:center;">'
            f'<span style="color:#aaa; font-size:13px; font-variant-numeric:tabular-nums; min-width:60px; text-align:right;">{fr["Champ %"]}</span>'
            f'<span style="font-weight:700; font-size:16px; color:{odds_color}; min-width:80px; text-align:right; '
            f'font-variant-numeric:tabular-nums;">'
            f'{fr["Futures Odds"]}</span>'
            f'</div></div>'
        )
    board_html += '</div>'
    st.markdown(board_html, unsafe_allow_html=True)

    st.markdown("")

    # Final Four & Championship futures
    st.subheader("Final Four Futures")
    ff_rows = []
    for _, row in df.iterrows():
        ffp = row["_champ_pct"]  # _champ_pct is index 6, we need FF which is index 4
        # Get the original probs for this team
        tid_matches = [tid for tid, p_list in probs.items() if tname(teams, tid) == row["Team"]]
        if tid_matches:
            tid = tid_matches[0]
            ff_prob = probs[tid][4]  # FF probability
            if ff_prob > 0:
                ff_ml = prob_to_moneyline(ff_prob)
            else:
                ff_ml = "+99999"
            ff_rows.append({
                "Team": row["Team"],
                "Seed": row["Seed"],
                "Final Four %": f"{ff_prob*100:.1f}%",
                "FF Odds": ff_ml,
                "_ff": ff_prob,
            })

    ff_df = pd.DataFrame(ff_rows).sort_values("_ff", ascending=False).reset_index(drop=True)
    ff_df.index = ff_df.index + 1
    ff_df.index.name = "#"
    _styled_df(ff_df.drop(columns=["_ff"]).head(30))

    # Top contenders bar chart
    st.markdown("---")
    top = df.head(16).copy()
    top["Champ %"] = top["_champ_pct"] * 100

    st.subheader("Top Championship Contenders")
    chart_data = top.set_index("Team")["Champ %"]
    st.bar_chart(chart_data, height=350)

    # Full table
    st.subheader("Full Round-by-Round Advancement Odds")
    _styled_df(df.drop(columns=["_champ_pct"]), max_height=600)


# ──────────────────────────── Page: Bracket ────────────────────────────

def page_bracket(prefix, teams, seeds_df, slots_df, preds):
    st.markdown('<div class="vp-section">PREDICTED BRACKET</div>', unsafe_allow_html=True)

    sim_results, slot_winners = simulate_bracket(seeds_df, slots_df, preds, deterministic=True)

    team_seed_map = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))

    # Get 1-seed names for region labels
    seed_to_team = dict(zip(seeds_df.Seed, seeds_df.TeamID))
    region_labels = {}
    for region in ["W", "X", "Y", "Z"]:
        one_seed_tid = seed_to_team.get(f"{region}01")
        if one_seed_tid:
            region_labels[region] = f"{tname(teams, one_seed_tid)} Region ({region})"
        else:
            region_labels[region] = f"Region {region}"

    # ── Compare bracket picks vs actual results from ESPN ──
    espn_data = _load_cached_espn()
    odds_map_bk, name_to_tid_bk = _build_odds_name_map(teams, prefix)
    actual_winners = set()  # team IDs that actually won
    actual_losers = set()   # team IDs that actually lost
    actual_matchups = {}    # (winner_tid, loser_tid) -> score string

    if espn_data and espn_data.get("games"):
        for g in espn_data["games"]:
            if g.get("status") != "STATUS_FINAL":
                continue
            home_score = int(g.get("home_score", 0))
            away_score = int(g.get("away_score", 0))
            w_name = g["home_team"] if home_score > away_score else g["away_team"]
            l_name = g["away_team"] if home_score > away_score else g["home_team"]
            w_score = max(home_score, away_score)
            l_score = min(home_score, away_score)

            w_tid = _resolve_odds_team(w_name, odds_map_bk, name_to_tid_bk)
            l_tid = _resolve_odds_team(l_name, odds_map_bk, name_to_tid_bk)
            if w_tid:
                actual_winners.add(w_tid)
            if l_tid:
                actual_losers.add(l_tid)
            if w_tid and l_tid:
                actual_matchups[(w_tid, l_tid)] = f"{w_score}-{l_score}"

    # Find bracket misses: games where model predicted the wrong winner
    bracket_misses = []
    bracket_hits = 0
    for slot, r in sim_results.items():
        t1, t2, winner = r.get("t1"), r.get("t2"), r.get("winner")
        if t1 is None or t2 is None:
            continue
        loser = t2 if winner == t1 else t1
        # Check if this exact matchup has a result (check both orderings)
        if (winner, loser) in actual_matchups:
            bracket_hits += 1
        elif (loser, winner) in actual_matchups:
            # Model predicted wrong winner
            bracket_misses.append({
                "slot": slot,
                "predicted": tname(teams, winner),
                "predicted_tid": winner,
                "predicted_seed": team_seed_map.get(winner, ""),
                "actual": tname(teams, loser),
                "actual_tid": loser,
                "actual_seed": team_seed_map.get(loser, ""),
                "score": actual_matchups[(loser, winner)],
                "prob": r.get("prob", 0.5),
            })

    total_graded = bracket_hits + len(bracket_misses)

    # ── Record Banner ──
    if total_graded > 0:
        rec_color = "#4ade80" if bracket_hits > len(bracket_misses) else "#f87171"
        st.markdown(
            f'<div class="vp-card" style="border-left:4px solid {rec_color}; text-align:center;">'
            f'<div style="font-size:12px; color:#888; letter-spacing:1px; font-weight:600; margin-bottom:4px;">'
            f'2026 TOURNAMENT BRACKET RECORD</div>'
            f'<div style="font-size:32px; font-weight:900; color:{rec_color};">'
            f'{bracket_hits}-{len(bracket_misses)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Champion banner
    champ_result = sim_results.get("R6CH", {})
    champ = champ_result.get("winner")
    if champ:
        s = team_seed_map.get(champ, "")
        seed_txt = f" ({s} seed)" if s else ""
        # Check if champ is already eliminated
        champ_elim = champ in actual_losers
        elim_txt = ' <span style="color:#f87171; font-size:14px;">ELIMINATED</span>' if champ_elim else ""
        st.markdown(
            f'<div class="champ-banner">\U0001f3c6 Predicted Champion: '
            f'{tname(teams, champ)}{seed_txt}{elim_txt}</div>',
            unsafe_allow_html=True,
        )

    # Final Four
    st.subheader("Final Four")
    st.markdown(final_four_html(sim_results, teams, team_seed_map), unsafe_allow_html=True)

    st.markdown("---")

    # ── Missed Picks ──
    if bracket_misses:
        st.markdown('<div class="vp-section">BRACKET MISSES</div>', unsafe_allow_html=True)
        st.caption(f"Games where our model predicted the wrong winner ({len(bracket_misses)} miss{'es' if len(bracket_misses) != 1 else ''}).")
        for miss in bracket_misses:
            pred_seed = f"({miss['predicted_seed']}) " if miss['predicted_seed'] else ""
            actual_seed = f"({miss['actual_seed']}) " if miss['actual_seed'] else ""
            st.markdown(
                f'<div class="vp-card" style="border-left:4px solid #f87171;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div>'
                f'<div style="font-size:13px; margin-bottom:4px;">'
                f'<span style="color:#f87171; font-weight:700;">Model picked:</span> '
                f'<span style="color:#888; text-decoration:line-through;">{pred_seed}{miss["predicted"]}</span>'
                f' <span style="color:#888;">({miss["prob"]*100:.0f}% confidence)</span></div>'
                f'<div style="font-size:13px;">'
                f'<span style="color:#4ade80; font-weight:700;">Actual winner:</span> '
                f'<span style="color:#FAFAFA; font-weight:600;">{actual_seed}{miss["actual"]}</span>'
                f' <span style="color:#888;">{miss["score"]}</span></div>'
                f'</div>'
                f'<span style="background:#f87171; color:#000; font-weight:700; padding:3px 10px; '
                f'border-radius:4px; font-size:11px; letter-spacing:0.5px;">MISS</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    # Regional brackets
    for region in ["W", "X", "Y", "Z"]:
        st.subheader(region_labels[region])
        html = region_bracket_html(region, sim_results, teams, team_seed_map)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("")

    # ── Projected Betting Lines for All Round 1 Games ──
    st.markdown("---")
    st.subheader("Projected Betting Lines — Round of 64")
    st.caption("Projected lines for every first-round matchup. For entertainment purposes only.")

    stats = compute_season_stats(prefix)

    for region in ["W", "X", "Y", "Z"]:
        st.markdown(f"**{region_labels[region]}**")
        r1_slots = [f"R1{region}{n}" for n in [1, 8, 5, 4, 6, 3, 7, 2]]
        game_rows = []
        for slot in r1_slots:
            r = sim_results.get(slot, {})
            gt1, gt2 = r.get("t1"), r.get("t2")
            if gt1 is None or gt2 is None:
                continue
            gl = compute_betting_lines(stats, preds, gt1, gt2)
            gs1 = team_seed_map.get(gt1, "")
            gs2 = team_seed_map.get(gt2, "")
            seed1_txt = f"({gs1}) " if gs1 else ""
            seed2_txt = f"({gs2}) " if gs2 else ""
            game_rows.append({
                "Matchup": f"{seed1_txt}{tname(teams, gt1)} vs {seed2_txt}{tname(teams, gt2)}",
                "Spread": f"{gl['spread']:+.1f}",
                "ML (T1)": gl["t1_ml"],
                "ML (T2)": gl["t2_ml"],
                "Total": f"{gl['total']:.1f}",
                "Proj Score": f"{gl['t1_pts']:.0f}-{gl['t2_pts']:.0f}",
            })

        if game_rows:
            _styled_table(game_rows)
        st.markdown("")

    # ── Championship Odds (Monte Carlo) ──
    st.markdown("---")
    st.subheader("Championship Odds")
    elim_count = len(actual_losers) if actual_losers else 0
    if elim_count:
        st.caption(f"Based on 10,000 simulated tournaments. {elim_count} eliminated teams removed from projections.")
    else:
        st.caption("Based on 10,000 simulated tournaments.")

    with st.spinner("Running 10,000 tournament simulations..."):
        probs, round_labels = monte_carlo_odds(
            seeds_df, slots_df, preds,
            eliminated=frozenset(actual_losers) if actual_losers else None,
        )

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    elo = compute_elo(prefix)

    odds_rows = []
    for tid, p_list in probs.items():
        is_elim = tid in actual_losers
        odds_rows.append({
            "Team": tname(teams, tid),
            "Seed": team_seeds.get(tid, 99),
            "Elo": int(elo.get(tid, 1500)),
            "R32": f"{p_list[1]*100:.1f}%",
            "S16": f"{p_list[2]*100:.1f}%",
            "E8": f"{p_list[3]*100:.1f}%",
            "FF": f"{p_list[4]*100:.1f}%",
            "CG": f"{p_list[5]*100:.1f}%",
            "Champ": f"{p_list[6]*100:.1f}%",
            "_champ_pct": p_list[6],
            "_eliminated": is_elim,
        })

    odds_df = pd.DataFrame(odds_rows).sort_values("_champ_pct", ascending=False).reset_index(drop=True)
    odds_df.index = odds_df.index + 1
    odds_df.index.name = "#"

    # Top 20 futures board styled like a sportsbook (exclude eliminated teams)
    futures_rows = []
    for _, row in odds_df.iterrows():
        if row.get("_eliminated", False):
            continue
        cp = row["_champ_pct"]
        if cp > 0:
            ml = prob_to_moneyline(cp)
        else:
            ml = "+99999"
        futures_rows.append({
            "Team": row["Team"],
            "Seed": row["Seed"],
            "Champ %": f"{cp*100:.1f}%",
            "Futures Odds": ml,
            "_cp": cp,
        })

    futures_df = pd.DataFrame(futures_rows).sort_values("_cp", ascending=False).reset_index(drop=True)
    top_futures = futures_df.head(20)
    board_html = '<div style="max-width:700px; margin:auto;">'
    for _, fr in top_futures.iterrows():
        seed_txt = f'<span style="color:#888; margin-right:6px;">({int(fr["Seed"])})</span>'
        odds_color = "#4ade80" if fr["_cp"] >= 0.10 else "#fbbf24" if fr["_cp"] >= 0.03 else "#e2e8f0"
        board_html += (
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:8px 16px; border-bottom:1px solid #2a2d34; background:#1a1d24; margin:1px 0;">'
            f'<span>{seed_txt}{fr["Team"]}</span>'
            f'<div style="display:flex; gap:24px; align-items:center;">'
            f'<span style="color:#aaa; font-size:13px;">{fr["Champ %"]}</span>'
            f'<span style="font-weight:700; font-size:16px; color:{odds_color}; min-width:80px; text-align:right;">'
            f'{fr["Futures Odds"]}</span>'
            f'</div></div>'
        )
    board_html += '</div>'
    st.markdown(board_html, unsafe_allow_html=True)

    st.markdown("")
    st.subheader("Full Round-by-Round Advancement Odds")
    active_df = odds_df[~odds_df["_eliminated"]].drop(columns=["_champ_pct", "_eliminated"])
    _styled_df(active_df, max_height=600)


# ──────────────────────────── Page: Model Backtest ────────────────────────────

@st.cache_data
def run_backtest(prefix):
    """Out-of-sample backtest: predict every tournament game using only pre-tournament data."""
    reg = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonCompactResults.csv"))
    trn = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneyCompactResults.csv"))
    det = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}RegularSeasonDetailedResults.csv"))
    seeds_all = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}NCAATourneySeeds.csv"))

    # ── Step 1: Build Elo through history, snapshot BEFORE each tournament ──
    all_games = pd.concat([reg, trn]).sort_values(["Season", "DayNum"]).reset_index(drop=True)
    K, HOME_ADV, DECAY = 20, 100, 0.75
    elo = {}
    elo_snapshots = {}
    prev_season = None

    for i in range(len(all_games)):
        season = int(all_games.iloc[i].Season)
        day = int(all_games.iloc[i].DayNum)

        if season != prev_season:
            if prev_season is not None:
                for tid in elo:
                    elo[tid] = 1500 + DECAY * (elo[tid] - 1500)
            prev_season = season

        # Snapshot before tournament games start (DayNum >= 134)
        if season not in elo_snapshots and day >= 134:
            elo_snapshots[season] = dict(elo)

        w_id = int(all_games.iloc[i].WTeamID)
        l_id = int(all_games.iloc[i].LTeamID)
        w_elo = elo.get(w_id, 1500)
        l_elo = elo.get(l_id, 1500)
        has_loc = "WLoc" in all_games.columns
        loc = str(all_games.iloc[i].WLoc) if has_loc else "N"
        w_adj = w_elo + (HOME_ADV if loc == "H" else 0)
        l_adj = l_elo + (HOME_ADV if loc == "A" else 0)

        exp_w = 1.0 / (1.0 + 10 ** ((l_adj - w_adj) / 400))
        mov = int(all_games.iloc[i].WScore) - int(all_games.iloc[i].LScore)
        diff = abs(w_adj - l_adj)
        damp = 2.2 / (diff * 0.001 + 2.2)
        k_adj = K * math.log(max(mov, 1) + 1) * 0.7 * damp
        elo[w_id] = w_elo + k_adj * (1 - exp_w)
        elo[l_id] = l_elo - k_adj * (1 - exp_w)

    if prev_season and prev_season not in elo_snapshots:
        elo_snapshots[prev_season] = dict(elo)

    # ── Step 2: Per-season stats for totals projections ──
    season_stats = {}
    for season in det.Season.unique():
        sdf = det[det.Season == season]
        records = []
        for _, r in sdf.iterrows():
            for side, opp in [("W", "L"), ("L", "W")]:
                records.append({
                    "TeamID": int(r[f"{side}TeamID"]),
                    "Score": r[f"{side}Score"], "OppScore": r[f"{opp}Score"],
                    "FGA": r[f"{side}FGA"], "ORB": r[f"{side}OR"],
                    "TO": r[f"{side}TO"], "FTA": r[f"{side}FTA"],
                    "OppFGA": r[f"{opp}FGA"], "OppOR": r[f"{opp}OR"],
                    "OppTO": r[f"{opp}TO"], "OppFTA": r[f"{opp}FTA"],
                })
        tg = pd.DataFrame(records)
        ss = tg.groupby("TeamID").agg(
            PPG=("Score", "mean"), OppPPG=("OppScore", "mean"),
            FGA=("FGA", "mean"), ORB=("ORB", "mean"),
            TOPG=("TO", "mean"), FTA=("FTA", "mean"),
            OppFGA=("OppFGA", "mean"), OppOR=("OppOR", "mean"),
            OppTO=("OppTO", "mean"), OppFTA=("OppFTA", "mean"),
        )
        ss["Poss"] = ss["FGA"] - ss["ORB"] + ss["TOPG"] + 0.475 * ss["FTA"]
        ss["OppPoss"] = ss["OppFGA"] - ss["OppOR"] + ss["OppTO"] + 0.475 * ss["OppFTA"]
        ss["OffEff"] = ss["PPG"] / ss["Poss"] * 100
        ss["DefEff"] = ss["OppPPG"] / ss["OppPoss"] * 100
        ss["Tempo"] = (ss["Poss"] + ss["OppPoss"]) / 2
        season_stats[int(season)] = ss

    # Seed map for all seasons
    seed_map = {}
    for _, r in seeds_all.iterrows():
        seed_map[(r.Season, r.TeamID)] = int(r.Seed[1:3])

    # ── Step 3: Backtest every tournament game ──
    bt_start = 2010 if prefix == "M" else 2015
    results = []

    for _, r in trn.iterrows():
        season = int(r.Season)
        if season < bt_start or season == 2020 or season >= SEASON:
            continue

        snap = elo_snapshots.get(season)
        ss = season_stats.get(season)
        if snap is None or ss is None:
            continue

        w_id, l_id = int(r.WTeamID), int(r.LTeamID)
        actual_margin = int(r.WScore) - int(r.LScore)
        actual_total = int(r.WScore) + int(r.LScore)

        # Convention: T1 = lower ID
        t1, t2 = min(w_id, l_id), max(w_id, l_id)
        e1 = snap.get(t1, 1500)
        e2 = snap.get(t2, 1500)
        pred_prob = 1 / (1 + 10 ** (-(e1 - e2) / 400))  # prob T1 wins

        # Generate lines
        spread = prob_to_spread(pred_prob)  # neg = T1 favored
        ml_t1 = prob_to_moneyline(pred_prob)
        ml_t2 = prob_to_moneyline(1 - pred_prob)

        # Project total
        if t1 in ss.index and t2 in ss.index:
            s1, s2 = ss.loc[t1], ss.loc[t2]
            avg_tempo = (s1.Tempo + s2.Tempo) / 2
            proj_total = round(((s1.OffEff + s2.DefEff) / 2 * avg_tempo / 100 +
                                (s2.OffEff + s1.DefEff) / 2 * avg_tempo / 100) * 2) / 2
        else:
            proj_total = 140.0

        # Who is our favorite?
        if pred_prob >= 0.5:
            fav, fav_prob, fav_ml = t1, pred_prob, ml_t1
        else:
            fav, fav_prob, fav_ml = t2, 1 - pred_prob, ml_t2

        # ── Moneyline result ──
        ml_correct = (fav == w_id)
        fav_odds = int(fav_ml.replace("+", ""))
        if ml_correct:
            ml_profit = 100 * 100 / abs(fav_odds) if fav_odds < 0 else 100 * fav_odds / 100
        else:
            ml_profit = -100.0

        # ── Spread result (bet on favorite to cover) ──
        actual_t1_margin = actual_margin if w_id == t1 else -actual_margin
        # spread is from T1's perspective: neg = T1 favored
        # If we bet fav: fav is T1 if spread < 0, else T2
        if fav == t1:
            ats_result = actual_t1_margin + spread  # spread is negative, so fav must win by more
            ats_correct = ats_result > 0
        else:
            ats_result = -actual_t1_margin - spread  # T2 is fav
            ats_correct = ats_result > 0
        ats_push = abs(ats_result) < 0.5
        ats_profit = (100 * 100 / 110) if ats_correct else (-110.0 if not ats_push else 0.0)

        # ── Total result (bet over if proj > 140, under otherwise — or just track accuracy) ──
        ou_over = actual_total > proj_total
        ou_under = actual_total < proj_total
        ou_push = abs(actual_total - proj_total) < 0.5
        total_diff = actual_total - proj_total

        s1_seed = seed_map.get((season, t1))
        s2_seed = seed_map.get((season, t2))

        results.append({
            "Season": season, "T1": t1, "T2": t2,
            "Winner": w_id, "W_Score": int(r.WScore), "L_Score": int(r.LScore),
            "Actual_Margin": actual_margin, "Actual_Total": actual_total,
            "Fav": fav, "Fav_Prob": round(fav_prob, 3),
            "Spread": spread, "Proj_Total": proj_total,
            "ML_Correct": ml_correct, "ML_Profit": round(ml_profit, 2),
            "ATS_Correct": ats_correct, "ATS_Push": ats_push, "ATS_Profit": round(ats_profit, 2),
            "OU_Over": ou_over, "OU_Push": ou_push, "Total_Diff": round(total_diff, 1),
            "Fav_Seed": seed_map.get((season, fav)),
            "Dog_Seed": seed_map.get((season, w_id if fav != w_id else l_id)),
        })

    return pd.DataFrame(results)


def _render_summary_cards(bt):
    """Render the 4 summary metric cards for a backtest dataframe."""
    total_games = len(bt)
    if total_games == 0:
        st.info("No games match your filter.")
        return

    ml_wins = int(bt.ML_Correct.sum())
    ml_pct = ml_wins / total_games * 100
    ml_total_profit = bt.ML_Profit.sum()
    ml_roi = ml_total_profit / (total_games * 100) * 100

    ats_valid = bt[~bt.ATS_Push]
    ats_wins = int(ats_valid.ATS_Correct.sum()) if len(ats_valid) > 0 else 0
    ats_pct = ats_wins / len(ats_valid) * 100 if len(ats_valid) > 0 else 0
    ats_total_profit = bt.ATS_Profit.sum()
    ats_roi = ats_total_profit / (total_games * 110) * 100

    ou_valid = bt[~bt.OU_Push]
    ou_over_ct = int(ou_valid.OU_Over.sum()) if len(ou_valid) > 0 else 0
    ou_under_ct = len(ou_valid) - ou_over_ct
    ou_pct = ou_over_ct / len(ou_valid) * 100 if len(ou_valid) > 0 else 50
    mae_total = bt.Total_Diff.abs().mean()

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        ml_color = "#4ade80" if ml_roi > 0 else "#f87171"
        profit_str = f'{"+" if ml_total_profit >= 0 else ""}{ml_total_profit:.0f}u'
        st.markdown(
            f'<div class="vp-metric" style="border-top:3px solid {ml_color};">'
            f'<div class="label">MONEYLINE</div>'
            f'<div class="value" style="color:{ml_color};">{ml_wins}-{total_games - ml_wins}</div>'
            f'<div class="sub">{ml_pct:.1f}% Win Rate</div>'
            f'<div style="font-size:14px; font-weight:700; color:{ml_color}; margin-top:6px;">'
            f'{profit_str} &middot; {ml_roi:+.1f}% ROI</div>'
            f'</div>', unsafe_allow_html=True)
    with kc2:
        ats_color = "#4ade80" if ats_pct > 52.4 else "#f87171"
        ats_profit_str = f'{"+" if ats_total_profit >= 0 else ""}{ats_total_profit:.0f}u'
        st.markdown(
            f'<div class="vp-metric" style="border-top:3px solid {ats_color};">'
            f'<div class="label">AGAINST SPREAD</div>'
            f'<div class="value" style="color:{ats_color};">{ats_wins}-{len(ats_valid) - ats_wins}</div>'
            f'<div class="sub">{ats_pct:.1f}% Cover Rate</div>'
            f'<div style="font-size:14px; font-weight:700; color:{ats_color}; margin-top:6px;">'
            f'{ats_profit_str} &middot; {ats_roi:+.1f}% ROI</div>'
            f'</div>', unsafe_allow_html=True)
    with kc3:
        st.markdown(
            f'<div class="vp-metric" style="border-top:3px solid #60a5fa;">'
            f'<div class="label">TOTALS (O/U)</div>'
            f'<div class="value" style="color:#60a5fa;">'
            f'{ou_over_ct}O / {ou_under_ct}U</div>'
            f'<div class="sub">Over hits: {ou_pct:.1f}%</div>'
            f'<div style="font-size:14px; font-weight:700; color:#fbbf24; margin-top:6px;">'
            f'MAE: {mae_total:.1f} pts</div>'
            f'</div>', unsafe_allow_html=True)
    with kc4:
        brier = ((bt.Fav_Prob - bt.ML_Correct.astype(float)) ** 2).mean()
        st.markdown(
            f'<div class="vp-metric" style="border-top:3px solid #c084fc;">'
            f'<div class="label">MODEL QUALITY</div>'
            f'<div class="value" style="color:#c084fc;">{brier:.3f}</div>'
            f'<div class="sub">Brier Score</div>'
            f'<div style="font-size:14px; font-weight:700; color:#c084fc; margin-top:6px;">'
            f'{total_games} games tested</div>'
            f'</div>', unsafe_allow_html=True)



def _load_cached_odds():
    """Load pre-fetched odds from cached_odds.json (updated by GitHub Actions)."""
    import json
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_odds.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_cached_espn():
    """Load pre-fetched ESPN scores from cached_espn.json (updated by GitHub Actions)."""
    import json
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_espn.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def page_backtest(prefix, teams):
    st.markdown('<div class="vp-section">MODEL BACKTEST</div>', unsafe_allow_html=True)
    st.caption(
        "How would our model have performed betting past NCAA tournaments? "
        "Every prediction uses only data available before that tournament — no hindsight."
    )

    with st.spinner("Running historical backtest..."):
        bt_all = run_backtest(prefix)

    if bt_all.empty:
        st.warning("No backtest data available.")
        return

    all_seasons = sorted(bt_all.Season.unique())

    # ── Filter Controls ──
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        selected_seasons = st.multiselect(
            "Filter by Tournament Year",
            options=all_seasons,
            default=all_seasons,
            help="Select specific tournaments to analyze",
        )
    with fc2:
        confidence_filter = st.slider("Min Confidence %", 50, 95, 50, step=5)

    bt = bt_all[bt_all.Season.isin(selected_seasons) & (bt_all.Fav_Prob * 100 >= confidence_filter)]

    if bt.empty:
        st.warning("No games match your filters.")
        return

    total_games = len(bt)
    seasons = sorted(bt.Season.unique())
    st.markdown(f"**{total_games} tournament games** across **{len(seasons)} seasons** "
                f"({seasons[0]}–{seasons[-1]})")

    # ── Summary Cards ──
    st.markdown("---")
    st.subheader("Overall Performance")
    _render_summary_cards(bt)

    # ── Totals Deep Dive ──
    st.markdown("---")
    st.subheader("Totals Performance Deep Dive")

    ou_valid = bt[~bt.OU_Push]
    if len(ou_valid) > 0:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            avg_proj = bt.Proj_Total.mean()
            avg_actual = bt.Actual_Total.mean()
            bias = avg_proj - avg_actual
            bias_color = "#f87171" if abs(bias) > 3 else "#4ade80"
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid {bias_color};">'
                f'<div class="label">PROJECTION BIAS</div>'
                f'<div class="value" style="color:{bias_color};">{bias:+.1f}</div>'
                f'<div class="sub">Avg Proj: {avg_proj:.1f} | Actual: {avg_actual:.1f}</div>'
                f'</div>', unsafe_allow_html=True)
        with tc2:
            rmse = (bt.Total_Diff ** 2).mean() ** 0.5
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid #60a5fa;">'
                f'<div class="label">RMSE</div>'
                f'<div class="value" style="color:#60a5fa;">{rmse:.1f}</div>'
                f'<div class="sub">Root Mean Squared Error</div>'
                f'</div>', unsafe_allow_html=True)
        with tc3:
            within_5 = (bt.Total_Diff.abs() <= 5).sum()
            within_10 = (bt.Total_Diff.abs() <= 10).sum()
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid #fbbf24;">'
                f'<div class="label">ACCURACY</div>'
                f'<div class="value" style="color:#fbbf24;">'
                f'{within_5/total_games*100:.0f}%</div>'
                f'<div class="sub">Within 5 pts: {within_5} | '
                f'Within 10: {within_10} ({within_10/total_games*100:.0f}%)</div>'
                f'</div>', unsafe_allow_html=True)

        # Totals by projected range
        st.markdown("**Performance by Projected Total Range**")
        total_tiers = [
            ("Low (< 125)", 0, 125), ("Mid-Low (125-135)", 125, 135),
            ("Mid (135-145)", 135, 145), ("Mid-High (145-155)", 145, 155),
            ("High (155+)", 155, 999),
        ]
        total_tier_rows = []
        for label, lo, hi in total_tiers:
            tb = bt[(bt.Proj_Total >= lo) & (bt.Proj_Total < hi)]
            if len(tb) == 0:
                continue
            tv = tb[~tb.OU_Push]
            over_ct = int(tv.OU_Over.sum()) if len(tv) > 0 else 0
            total_tier_rows.append({
                "Range": label, "Games": len(tb),
                "Avg Projected": f"{tb.Proj_Total.mean():.1f}",
                "Avg Actual": f"{tb.Actual_Total.mean():.1f}",
                "Bias": f"{(tb.Proj_Total.mean() - tb.Actual_Total.mean()):+.1f}",
                "Over/Under": f"{over_ct}/{len(tv) - over_ct}" if len(tv) > 0 else "—",
                "MAE": f"{tb.Total_Diff.abs().mean():.1f}",
            })
        if total_tier_rows:
            _styled_table(total_tier_rows)

        # Distribution chart
        st.markdown("**Projection Error Distribution**")
        hist_data = bt[["Total_Diff"]].rename(columns={"Total_Diff": "Error (Actual - Projected)"})
        st.bar_chart(
            hist_data["Error (Actual - Projected)"].value_counts().sort_index(),
            height=250,
        )

    # ── Season-by-Season Breakdown ──
    st.markdown("---")
    st.subheader("Season-by-Season Breakdown")

    season_rows = []
    for season in seasons:
        sbt = bt[bt.Season == season]
        sg = len(sbt)
        s_ml = int(sbt.ML_Correct.sum())
        s_ml_profit = sbt.ML_Profit.sum()
        s_ats_valid = sbt[~sbt.ATS_Push]
        s_ats = int(s_ats_valid.ATS_Correct.sum()) if len(s_ats_valid) > 0 else 0
        s_ats_profit = sbt.ATS_Profit.sum()
        s_mae = sbt.Total_Diff.abs().mean()
        s_ou_valid = sbt[~sbt.OU_Push]
        s_ou_over = int(s_ou_valid.OU_Over.sum()) if len(s_ou_valid) > 0 else 0
        season_rows.append({
            "Season": season, "Games": sg,
            "ML Record": f"{s_ml}-{sg - s_ml}",
            "ML %": f"{s_ml / sg * 100:.1f}%",
            "ML Profit": f"{'+'if s_ml_profit >= 0 else ''}{s_ml_profit:.0f}u",
            "ATS Record": f"{s_ats}-{len(s_ats_valid) - s_ats}" if len(s_ats_valid) > 0 else "—",
            "ATS Profit": f"{'+'if s_ats_profit >= 0 else ''}{s_ats_profit:.0f}u",
            "O/U": f"{s_ou_over}O/{len(s_ou_valid) - s_ou_over}U" if len(s_ou_valid) > 0 else "—",
            "Total MAE": f"{s_mae:.1f}",
        })

    _styled_table(season_rows)

    # ── Cumulative Profit Chart ──
    st.subheader("Cumulative Profit (Units)")
    bt_sorted = bt.sort_values(["Season"]).reset_index(drop=True)
    bt_sorted["Cum_ML_Profit"] = bt_sorted.ML_Profit.cumsum()
    bt_sorted["Cum_ATS_Profit"] = bt_sorted.ATS_Profit.cumsum()
    bt_sorted["Game_Num"] = range(1, len(bt_sorted) + 1)

    chart_df = bt_sorted[["Game_Num", "Cum_ML_Profit", "Cum_ATS_Profit"]].set_index("Game_Num")
    chart_df.columns = ["Moneyline", "Spread"]
    st.line_chart(chart_df, height=350)

    # ── Performance by Confidence Tier ──
    st.markdown("---")
    st.subheader("Performance by Confidence Tier")

    tiers = [
        ("Coin Flip (50-55%)", 0.50, 0.55), ("Lean (55-65%)", 0.55, 0.65),
        ("Confident (65-75%)", 0.65, 0.75), ("Strong (75-85%)", 0.75, 0.85),
        ("Lock (85%+)", 0.85, 1.01),
    ]
    tier_rows = []
    for label, lo, hi in tiers:
        tb = bt[(bt.Fav_Prob >= lo) & (bt.Fav_Prob < hi)]
        if len(tb) == 0:
            continue
        tw = int(tb.ML_Correct.sum())
        tier_rows.append({
            "Tier": label, "Games": len(tb),
            "ML Record": f"{tw}-{len(tb) - tw}",
            "ML Win %": f"{tw / len(tb) * 100:.1f}%",
            "ML Profit": f"{'+'if tb.ML_Profit.sum() >= 0 else ''}{tb.ML_Profit.sum():.0f}u",
            "Avg Spread": f"{tb.Spread.mean():.1f}",
            "ATS Win %": f"{tb[~tb.ATS_Push].ATS_Correct.mean() * 100:.1f}%" if len(tb[~tb.ATS_Push]) > 0 else "—",
            "Total MAE": f"{tb.Total_Diff.abs().mean():.1f}",
        })
    _styled_table(tier_rows)

    # ── Upset Tracker ──
    st.markdown("---")
    st.subheader("Biggest Upsets & Misses")

    upsets = bt[bt.ML_Correct == False].sort_values("Fav_Prob", ascending=False).head(15)
    upset_rows = []
    for _, u in upsets.iterrows():
        fav_name = tname(teams, u.Fav)
        winner_name = tname(teams, u.Winner)
        upset_rows.append({
            "Season": int(u.Season),
            "Our Pick": f"({int(u.Fav_Seed)}) {fav_name}" if pd.notna(u.Fav_Seed) else fav_name,
            "Confidence": f"{u.Fav_Prob * 100:.1f}%",
            "Actual Winner": f"({int(u.Dog_Seed)}) {winner_name}" if pd.notna(u.Dog_Seed) else winner_name,
            "Score": f"{int(u.W_Score)}-{int(u.L_Score)}",
            "Spread": f"{u.Spread:+.1f}",
        })
    if upset_rows:
        _styled_table(upset_rows)

    # ── Calibration ──
    st.markdown("---")
    st.subheader("Model Calibration")
    st.caption("If the model says 70%, the favorite should win ~70% of the time.")

    cal_data = []
    for lo_edge in range(50, 96, 5):
        hi_edge = lo_edge + 5
        bucket = bt[(bt.Fav_Prob * 100 >= lo_edge) & (bt.Fav_Prob * 100 < hi_edge)]
        if len(bucket) >= 3:
            actual_pct = bucket.ML_Correct.mean() * 100
            predicted_pct = (lo_edge + hi_edge) / 2
            cal_data.append({
                "Predicted": f"{lo_edge}-{hi_edge}%",
                "Actual Win %": round(actual_pct, 1),
                "Expected": predicted_pct,
                "Games": len(bucket),
                "Diff": round(actual_pct - predicted_pct, 1),
            })
    if cal_data:
        cal_df = pd.DataFrame(cal_data)
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            st.bar_chart(cal_df.set_index("Predicted")[["Actual Win %", "Expected"]], height=300)
        with cc2:
            _styled_df(cal_df)

    # ── Live Results Tracker ──
    live_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_results.json")
    if os.path.exists(live_path):
        import json
        with open(live_path) as f:
            live = json.load(f)

        if live.get("games"):
            st.markdown("---")
            st.subheader("2026 Tournament Results Tracker")
            st.markdown(
                f'<div style="background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.3); '
                f'border-radius:8px; padding:10px 16px; font-size:13px; color:#4ade80; font-weight:600;">'
                f'Tracking {len(live["games"])} completed games</div>',
                unsafe_allow_html=True,
            )
            live_rows = []
            ml_w, ml_l, ats_w, ats_l, ou_o, ou_u = 0, 0, 0, 0, 0, 0
            stats = compute_season_stats(prefix)
            for g in live["games"]:
                gt1, gt2 = min(g["team1"], g["team2"]), max(g["team1"], g["team2"])
                winner = g["winner"]
                w_score, l_score = g["w_score"], g["l_score"]
                actual_total = w_score + l_score

                p = get_pred(preds, gt1, gt2)
                fav = gt1 if p >= 0.5 else gt2
                fav_prob = p if fav == gt1 else 1 - p
                spread = prob_to_spread(p)
                proj_total = project_game_total(stats, gt1, gt2)

                ml_hit = (fav == winner)
                actual_margin = w_score - l_score
                actual_t1_m = actual_margin if winner == gt1 else -actual_margin
                ats_hit = (actual_t1_m + spread > 0) if fav == gt1 else (-actual_t1_m - spread > 0)
                is_over = actual_total > proj_total

                ml_w += ml_hit; ml_l += not ml_hit
                ats_w += ats_hit; ats_l += not ats_hit
                ou_o += is_over; ou_u += not is_over

                live_rows.append({
                    "Matchup": f"{tname(teams, gt1)} vs {tname(teams, gt2)}",
                    "Pred": f"{fav_prob*100:.0f}% {tname(teams, fav)}",
                    "Spread": f"{spread:+.1f}",
                    "Total": f"{proj_total:.1f}",
                    "Result": f"{tname(teams, winner)} {w_score}-{l_score}",
                    "ML": "HIT" if ml_hit else "MISS",
                    "ATS": "HIT" if ats_hit else "MISS",
                    "O/U": f"{'OVER' if is_over else 'UNDER'} ({actual_total})",
                })

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                c = "#4ade80" if ml_w > ml_l else "#f87171"
                st.markdown(
                    f'<div class="vp-metric" style="border-top:3px solid {c};">'
                    f'<div class="label">MONEYLINE</div>'
                    f'<div class="value" style="color:{c};">{ml_w}-{ml_l}</div></div>',
                    unsafe_allow_html=True)
            with lc2:
                c = "#4ade80" if ats_w > ats_l else "#f87171"
                st.markdown(
                    f'<div class="vp-metric" style="border-top:3px solid {c};">'
                    f'<div class="label">SPREAD (ATS)</div>'
                    f'<div class="value" style="color:{c};">{ats_w}-{ats_l}</div></div>',
                    unsafe_allow_html=True)
            with lc3:
                st.markdown(
                    f'<div class="vp-metric" style="border-top:3px solid #fbbf24;">'
                    f'<div class="label">TOTALS (O/U)</div>'
                    f'<div class="value" style="color:#fbbf24;">{ou_o}O-{ou_u}U</div></div>',
                    unsafe_allow_html=True)

            _styled_table(live_rows)


# ──────────────────────────── Page: Betting Picks ────────────────────────────

def _implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def _ev(model_prob, american_odds):
    """Expected value of a $100 bet given model probability and American odds."""
    if american_odds < 0:
        profit = 100 * 100 / abs(american_odds)
    else:
        profit = 100 * american_odds / 100
    return model_prob * profit - (1 - model_prob) * 100


def _kelly(model_prob, american_odds):
    """Kelly criterion fraction for optimal bet sizing."""
    if american_odds < 0:
        b = 100 / abs(american_odds)
    else:
        b = american_odds / 100
    q = 1 - model_prob
    f = (model_prob * b - q) / b
    return max(f, 0)


def _edge_rating(ev, kelly_frac, backtest_pct):
    """Composite edge score 0-100 combining EV, Kelly, and historical accuracy."""
    ev_score = min(max(ev / 20, 0), 1)  # +20 EV = max score
    kelly_score = min(kelly_frac / 0.15, 1)  # 15% Kelly = max
    bt_score = min(max((backtest_pct - 50) / 25, 0), 1)  # 75% historical = max
    return int((ev_score * 0.4 + kelly_score * 0.3 + bt_score * 0.3) * 100)


def _edge_label(score):
    if score >= 70:
        return "STRONG", "#4ade80"
    elif score >= 45:
        return "MODERATE", "#fbbf24"
    elif score >= 25:
        return "SLIGHT", "#60a5fa"
    else:
        return "SKIP", "#666"


def _build_odds_name_map(teams, prefix="M"):
    """Build a mapping from odds API team names to our team IDs.
    Odds API uses 'Duke Blue Devils', our data uses 'Duke'.
    Filter to only men's (1xxx) or women's (3xxx) teams based on prefix."""
    # Filter teams to the correct gender to avoid M/W ID collisions
    if prefix == "M":
        teams = {tid: name for tid, name in teams.items() if tid < 3000}
    else:
        teams = {tid: name for tid, name in teams.items() if tid >= 3000}
    # Explicit overrides for tricky names (odds API -> our data)
    OVERRIDES = {
        "hawai'i rainbow warriors": "Hawaii",
        "hawaii rainbow warriors": "Hawaii",
        "st. john's red storm": "St John's",
        "saint john's red storm": "St John's",
        "miami hurricanes": "Miami FL",
        "miami (oh) redhawks": "Miami OH",
        "miami (fl) hurricanes": "Miami FL",
        "queens university royals": "Queens NC",
        "liu sharks": "LIU Brooklyn",
        "suny albany great danes": "SUNY Albany",
        "unc wilmington seahawks": "UNC Wilmington",
        "unc greensboro spartans": "UNC Greensboro",
        "unc asheville bulldogs": "UNC Asheville",
        "gw revolutionaries": "G Washington",
        "george washington revolutionaries": "G Washington",
        "george washington colonials": "G Washington",
        "saint mary's gaels": "St Mary's CA",
        "saint joseph's hawks": "St Joseph's PA",
        "sam houston state bearkats": "Sam Houston St",
        "sam houston bearkats": "Sam Houston St",
        "houston christian huskies": "Houston Chr",
        "cal baptist lancers": "Cal Baptist",
        "prairie view panthers": "Prairie View",
        "penn quakers": "Pennsylvania",
        "pennsylvania quakers": "Pennsylvania",
        "connecticut huskies": "UConn",
        "uconn huskies": "UConn",
        "smu mustangs": "SMU",
        "lsu tigers": "LSU",
        "tcu horned frogs": "TCU",
        "ucf knights": "UCF",
        "ole miss rebels": "Ole Miss",
        "pitt panthers": "Pittsburgh",
        "pittsburgh panthers": "Pittsburgh",
        "vcu rams": "VCU",
        "utep miners": "UTEP",
        "unlv rebels": "UNLV",
        "usc trojans": "USC",
        "north dakota st bison": "N Dakota St",
        "north dakota state bison": "N Dakota St",
        "south dakota st jackrabbits": "S Dakota St",
        "south dakota state jackrabbits": "S Dakota St",
        "east washington eagles": "E Washington",
        "eastern washington eagles": "E Washington",
        "mcneese cowboys": "McNeese St",
        "mcneese state cowboys": "McNeese St",
        "seattle redhawks": "Seattle",
        "kennesaw st owls": "Kennesaw",
        "kennesaw state owls": "Kennesaw",
        "illinois st redbirds": "Illinois St",
        "illinois state redbirds": "Illinois St",
        "utah state aggies": "Utah St",
        "michigan st spartans": "Michigan St",
        "michigan state spartans": "Michigan St",
        "oklahoma st cowboys": "Oklahoma St",
        "oklahoma state cowboys": "Oklahoma St",
        "wichita st shockers": "Wichita St",
        "wichita state shockers": "Wichita St",
        "tennessee st tigers": "Tennessee St",
        "tennessee state tigers": "Tennessee St",
        "wright st raiders": "Wright St",
        "wright state raiders": "Wright St",
        "saint louis billikens": "Saint Louis",
        "texas a&m aggies": "Texas A&M",
        "byu cougars": "BYU",
        "arkansas pine bluff golden lions": "Ark Pine Bluff",
        "loy marymount lions": "Loy Marymount",
        "loyola marymount lions": "Loy Marymount",
        "mt st mary's mountaineers": "Mt St Mary's",
        "mount st mary's mountaineers": "Mt St Mary's",
        "cal golden bears": "California",
        "california golden bears": "California",
        "iowa hawkeyes": "Iowa",
        "iowa state cyclones": "Iowa St",
        "new mexico lobos": "New Mexico",
        "new mexico st aggies": "New Mexico St",
        "northern iowa panthers": "Northern Iowa",
        "dayton flyers": "Dayton",
        "liberty flames": "Liberty",
        "furman paladins": "Furman",
        "hofstra pride": "Hofstra",
        "idaho vandals": "Idaho",
        "akron zips": "Akron",
        "santa clara broncos": "Santa Clara",
        "wake forest demon deacons": "Wake Forest",
        "nebraska cornhuskers": "Nebraska",
        "tulsa golden hurricane": "Tulsa",
        "missouri tigers": "Missouri",
        "texas tech red raiders": "Texas Tech",
        "texas longhorns": "Texas",
        "nevada wolf pack": "Nevada",
    }

    # Build reverse map: our_name -> tid
    name_to_tid = {}
    for tid, name in teams.items():
        name_to_tid[name.lower()] = tid

    # Build odds_name -> tid
    odds_map = {}
    for odds_name_lower, our_name in OVERRIDES.items():
        tid = name_to_tid.get(our_name.lower())
        if tid:
            odds_map[odds_name_lower] = tid

    # For each team, also register patterns: "teamname *" matches
    for tid, our_name in teams.items():
        ln = our_name.lower()
        odds_map[ln] = tid  # exact match

    return odds_map, name_to_tid


def _resolve_odds_team(odds_name, odds_map, name_to_tid):
    """Resolve an odds API team name to a team ID."""
    ln = odds_name.lower()

    # Direct match (including overrides)
    if ln in odds_map:
        return odds_map[ln]

    # Try stripping last word (mascot) progressively: "Duke Blue Devils" -> "Duke Blue" -> "Duke"
    words = ln.split()
    for i in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:i])
        if prefix in name_to_tid:
            return name_to_tid[prefix]

    # Try "St" / "St." normalization: "Oklahoma St Cowboys" -> "Oklahoma St"
    for i in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:i])
        # Try adding "St" if it's not there
        prefix_st = prefix.replace("state", "st").replace("saint", "st")
        if prefix_st in name_to_tid:
            return name_to_tid[prefix_st]
        # Try "St." -> "St"
        prefix_nodot = prefix.replace("st.", "st")
        if prefix_nodot in name_to_tid:
            return name_to_tid[prefix_nodot]

    # Substring: if our team name is IN the odds name
    for name, tid in name_to_tid.items():
        if len(name) >= 4 and name in ln:
            return tid

    return None


def _match_odds_teams(teams, stats, odds_cache, prefix="M"):
    """Match odds API team names to our team IDs and extract Vegas lines."""
    odds_map, name_to_tid = _build_odds_name_map(teams, prefix)
    matched = []

    for game in odds_cache.get("games", []):
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        home_tid = _resolve_odds_team(home, odds_map, name_to_tid)
        away_tid = _resolve_odds_team(away, odds_map, name_to_tid)

        if not home_tid or not away_tid or home_tid not in stats.index or away_tid not in stats.index:
            continue

        # Merge markets from all bookmakers (use first available for each market type)
        markets = {}
        bk_title = bookmakers[0].get("title", "")
        for bk in bookmakers:
            for m in bk.get("markets", []):
                if m["key"] not in markets:
                    markets[m["key"]] = m

        v_ml1, v_ml2, v_spread, v_total = None, None, None, None

        t1, t2 = min(home_tid, away_tid), max(home_tid, away_tid)
        home_is_t1 = (home_tid == t1)

        if "h2h" in markets:
            for o in markets["h2h"]["outcomes"]:
                if o["name"] == home:
                    if home_is_t1:
                        v_ml1 = o["price"]
                    else:
                        v_ml2 = o["price"]
                elif o["name"] == away:
                    if home_is_t1:
                        v_ml2 = o["price"]
                    else:
                        v_ml1 = o["price"]

        if "spreads" in markets:
            for o in markets["spreads"]["outcomes"]:
                if o["name"] == home:
                    pt = o.get("point", 0)
                    v_spread = pt if home_is_t1 else -pt
                    break

        if "totals" in markets:
            for o in markets["totals"]["outcomes"]:
                if o["name"] == "Over":
                    v_total = o.get("point")
                    break

        matched.append({
            "t1": t1, "t2": t2,
            "home": home, "away": away, "bk_title": bk_title,
            "v_ml1": v_ml1, "v_ml2": v_ml2,
            "v_spread": v_spread, "v_total": v_total,
            "commence_time": game.get("commence_time", ""),
        })

    return matched


def page_picks(prefix, teams, seeds_df, preds):
    st.markdown('<div class="vp-section">TODAY\'S PICKS</div>', unsafe_allow_html=True)
    st.caption("Live scores, Vegas odds, and our model's best bets.")

    # ── Load data ──
    espn_data = _load_cached_espn()
    odds_data = _load_cached_odds()
    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)
    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))

    # Load backtest for historical accuracy by confidence tier
    bt = run_backtest(prefix)
    bt_tiers = {}
    for lo in range(50, 100, 5):
        hi = lo + 5
        tier = bt[(bt.Fav_Prob * 100 >= lo) & (bt.Fav_Prob * 100 < hi)]
        if len(tier) >= 5:
            bt_tiers[(lo, hi)] = {
                "ml_pct": tier.ML_Correct.mean() * 100,
                "ats_pct": tier[~tier.ATS_Push].ATS_Correct.mean() * 100 if len(tier[~tier.ATS_Push]) > 0 else 50,
                "games": len(tier),
            }

    def get_bt_pct(prob, bet_type="ml"):
        pct_val = prob * 100
        for (lo, hi), data in bt_tiers.items():
            if lo <= pct_val < hi:
                return data[f"{bet_type}_pct"]
        return 50.0

    # ── Pre-compute odds matching for use in FINAL grading ──
    odds_map_lookup, name_to_tid_lookup = _build_odds_name_map(teams, prefix)

    # ── Live Scores ──
    if espn_data and espn_data.get("games"):
        espn_games = espn_data["games"]
        fetched_at = espn_data.get("fetched_at", "")
        if fetched_at:
            st.caption(f"Scores last updated: {fetched_at[:16].replace('T', ' ')} UTC")

        live_games = [g for g in espn_games if g["status"] == "STATUS_IN_PROGRESS"]
        final_games = [g for g in espn_games if g["status"] == "STATUS_FINAL"]

        if live_games:
            st.markdown('<div class="vp-section" style="margin-top:8px;">LIVE</div>', unsafe_allow_html=True)
            for g in live_games:
                st.markdown(
                    f'<div class="vp-card" style="border-left:3px solid #fbbf24; display:flex; '
                    f'justify-content:space-between; align-items:center;">'
                    f'<div style="font-weight:600;">'
                    f'<span style="color:#aaa;">{g["away_team"]}</span> '
                    f'<span style="font-size:18px; font-weight:800;">{g["away_score"]}</span>'
                    f'<span style="color:#555; margin:0 8px;">@</span>'
                    f'<span style="color:#aaa;">{g["home_team"]}</span> '
                    f'<span style="font-size:18px; font-weight:800;">{g["home_score"]}</span></div>'
                    f'<span style="color:#000; background:#fbbf24; font-weight:700; font-size:10px; '
                    f'padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">{g["status_detail"]}</span>'
                    f'</div>', unsafe_allow_html=True)

        if final_games:
            st.markdown('<div class="vp-section" style="margin-top:8px;">FINAL — MODEL PICKS</div>', unsafe_allow_html=True)
            for g in final_games:
                away_score = int(g["away_score"])
                home_score = int(g["home_score"])
                a_bold = 'font-weight:800; color:#FAFAFA;' if away_score > home_score else ''
                h_bold = 'font-weight:800; color:#FAFAFA;' if home_score > away_score else ''

                # Try to resolve ESPN teams to model IDs for pick display
                home_tid = _resolve_odds_team(g["home_team"], odds_map_lookup, name_to_tid_lookup)
                away_tid = _resolve_odds_team(g["away_team"], odds_map_lookup, name_to_tid_lookup)
                pick_html = ""
                if home_tid and away_tid and home_tid in stats.index and away_tid in stats.index:
                    t1g, t2g = min(home_tid, away_tid), max(home_tid, away_tid)
                    pg = get_pred(preds, t1g, t2g)
                    lines_g = compute_betting_lines(stats, preds, t1g, t2g)
                    fav_g = t1g if pg >= 0.5 else t2g
                    fav_prob_g = pg if fav_g == t1g else 1 - pg
                    fav_name_g = tname(teams, fav_g)
                    m_spread_g = lines_g["spread"]
                    m_total_g = lines_g["total"]

                    # Determine actual margin (t1 perspective)
                    t1_score = home_score if home_tid == t1g else away_score
                    t2_score = away_score if home_tid == t1g else home_score
                    actual_total = t1_score + t2_score

                    # ML result
                    winner_tid = home_tid if home_score > away_score else away_tid
                    ml_correct = (fav_g == winner_tid)
                    ml_icon = "✓" if ml_correct else "✗"
                    ml_color = "#4ade80" if ml_correct else "#f87171"

                    pick_html = (
                        f'<div style="display:flex; gap:10px; margin-top:4px; flex-wrap:wrap;">'
                        f'<span style="font-size:11px; font-weight:600; color:{ml_color};">'
                        f'{ml_icon} ML: {fav_name_g} ({fav_prob_g*100:.0f}%)</span>'
                        f'<span style="font-size:11px; font-weight:600; color:#aaa;">'
                        f'Spread: {m_spread_g:+.1f}</span>'
                        f'<span style="font-size:11px; font-weight:600; color:#aaa;">'
                        f'Total: {m_total_g:.0f} (Actual: {actual_total})</span>'
                        f'</div>'
                    )

                st.markdown(
                    f'<div class="vp-card" style="border-left:3px solid #4ade80;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div style="font-weight:500;">'
                    f'<span style="color:#aaa; {a_bold}">{g["away_team"]}</span> '
                    f'<span style="font-size:18px; {a_bold}">{g["away_score"]}</span>'
                    f'<span style="color:#555; margin:0 8px;">@</span>'
                    f'<span style="color:#aaa; {h_bold}">{g["home_team"]}</span> '
                    f'<span style="font-size:18px; {h_bold}">{g["home_score"]}</span></div>'
                    f'<span style="color:#4ade80; font-weight:700; font-size:10px; letter-spacing:1px;">FINAL</span>'
                    f'</div>'
                    f'{pick_html}'
                    f'</div>', unsafe_allow_html=True)

    # ── Build ESPN broadcast + score lookup ──
    espn_broadcast = {}
    espn_scores = {}  # keyed by lowercased team fragments for flexible matching
    if espn_data and espn_data.get("games"):
        for g in espn_data["games"]:
            key = g.get("name", "").lower()
            espn_broadcast[key] = {
                "broadcast": g.get("broadcast", ""),
                "start_time": g.get("start_time", ""),
            }
            # Store scores for grading picks on final games
            espn_scores[key] = {
                "status": g.get("status", ""),
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "home_score": int(g.get("home_score", 0)),
                "away_score": int(g.get("away_score", 0)),
            }

    # ── Model Picks vs Vegas ──
    st.markdown("---")
    st.subheader("Model Picks")
    st.caption(
        "Our model vs Vegas for every upcoming game, sorted by tip-off time."
    )

    # Try to auto-load live odds first
    has_live_odds = odds_data and odds_data.get("games")
    odds_matched = []
    if has_live_odds:
        odds_matched = _match_odds_teams(teams, stats, odds_data, prefix)

    if odds_matched:
        # We have live odds — default to those, but allow switching
        data_source = st.radio(
            "Odds source",
            ["Live Vegas Odds", "Enter Odds Manually", "Tournament Bracket"],
            horizontal=True,
        )
    else:
        data_source = st.radio(
            "Odds source",
            ["Tournament Bracket", "Enter Odds Manually"],
            horizontal=True,
        )

    picks = []

    if data_source == "Live Vegas Odds" and odds_matched:
        fetched_at = odds_data.get("fetched_at", "")
        st.success(f"Comparing model vs Vegas for {len(odds_matched)} games"
                   + (f" (odds updated {fetched_at[:16].replace('T', ' ')} UTC)" if fetched_at else ""))

        for gm in odds_matched:
            t1, t2 = gm["t1"], gm["t2"]
            p = get_pred(preds, t1, t2)
            lines = compute_betting_lines(stats, preds, t1, t2)

            picks.append({
                "t1": t1, "t2": t2,
                "model_prob_t1": p,
                "model_spread": lines["spread"],
                "model_total": lines["total"],
                "model_ml_t1": lines["t1_ml"],
                "model_ml_t2": lines["t2_ml"],
                "model_t1_pts": lines["t1_pts"],
                "model_t2_pts": lines["t2_pts"],
                "vegas_ml_t1": gm["v_ml1"], "vegas_ml_t2": gm["v_ml2"],
                "vegas_spread": gm["v_spread"], "vegas_total": gm["v_total"],
                "region": "",
                "commence_time": gm.get("commence_time", ""),
                "home": gm.get("home", ""), "away": gm.get("away", ""),
            })

    elif data_source == "Tournament Bracket":
        m_slots_df, w_slots_df = load_slots()
        slots_df = m_slots_df if prefix == "M" else w_slots_df
        sim_results, _ = simulate_bracket(seeds_df, slots_df, preds, deterministic=True)

        st.info("Showing model projections for all Round 1 games. "
                "Switch to 'Live Vegas Odds' to compare against real lines.")

        for region in ["W", "X", "Y", "Z"]:
            for n in [1, 8, 5, 4, 6, 3, 7, 2]:
                slot = f"R1{region}{n}"
                r = sim_results.get(slot, {})
                gt1, gt2 = r.get("t1"), r.get("t2")
                if gt1 is None or gt2 is None:
                    continue

                p = get_pred(preds, gt1, gt2)
                lines = compute_betting_lines(stats, preds, gt1, gt2)

                picks.append({
                    "t1": gt1, "t2": gt2,
                    "model_prob_t1": p,
                    "model_spread": lines["spread"],
                    "model_total": lines["total"],
                    "model_ml_t1": lines["t1_ml"],
                    "model_ml_t2": lines["t2_ml"],
                    "model_t1_pts": lines["t1_pts"],
                    "model_t2_pts": lines["t2_pts"],
                    "vegas_ml_t1": None, "vegas_ml_t2": None,
                    "vegas_spread": None, "vegas_total": None,
                    "region": region,
                    "home": "", "away": "",
                })

    elif data_source == "Enter Odds Manually":
        st.markdown("**Enter a matchup with Vegas odds:**")
        mc1, mc2 = st.columns(2)

        id_range = range(1101, 1482) if prefix == "M" else range(3101, 3482)
        available = [tid for tid in id_range if tid in stats.index]
        team_options = sorted(available, key=lambda t: elo.get(t, 1500), reverse=True)
        labels = {t: tname(teams, t) for t in team_options}

        with mc1:
            mt1 = st.selectbox("Team 1", team_options, format_func=lambda t: labels[t], index=0, key="pick_t1")
        with mc2:
            mt2 = st.selectbox("Team 2", team_options, format_func=lambda t: labels[t], index=1, key="pick_t2")

        if mt1 != mt2:
            vc1, vc2, vc3, vc4 = st.columns(4)
            with vc1:
                v_ml1 = st.number_input(f"Vegas ML ({labels[mt1]})", value=-200, step=5, key="vml1")
            with vc2:
                v_ml2 = st.number_input(f"Vegas ML ({labels[mt2]})", value=170, step=5, key="vml2")
            with vc3:
                v_spread = st.number_input("Vegas Spread (T1)", value=-5.0, step=0.5, key="vspread")
            with vc4:
                v_total = st.number_input("Vegas Total (O/U)", value=140.0, step=0.5, key="vtotal")

            p = get_pred(preds, mt1, mt2)
            lines = compute_betting_lines(stats, preds, mt1, mt2)
            picks.append({
                "t1": mt1, "t2": mt2,
                "model_prob_t1": p,
                "model_spread": lines["spread"],
                "model_total": lines["total"],
                "model_ml_t1": lines["t1_ml"],
                "model_ml_t2": lines["t2_ml"],
                "model_t1_pts": lines["t1_pts"],
                "model_t2_pts": lines["t2_pts"],
                "vegas_ml_t1": v_ml1, "vegas_ml_t2": v_ml2,
                "vegas_spread": v_spread, "vegas_total": v_total,
                "region": "",
                "home": "", "away": "",
            })

    # ── Render Picks ──
    if not picks:
        st.info("No games found. Check back when odds are posted or use 'Tournament Bracket' mode.")
        return

    # ── Build game cards with all three bet types per game ──
    all_bets = []  # for summary table
    game_cards = []  # for game-by-game display

    for pick in picks:
        t1, t2 = pick["t1"], pick["t2"]
        n1, n2 = tname(teams, t1), tname(teams, t2)
        s1 = team_seeds.get(t1, "")
        s2 = team_seeds.get(t2, "")
        s1t = f"({s1}) " if s1 else ""
        s2t = f"({s2}) " if s2 else ""
        p = pick["model_prob_t1"]
        fav = t1 if p >= 0.5 else t2
        fav_name = n1 if fav == t1 else n2
        dog_name = n2 if fav == t1 else n1
        fav_prob = p if fav == t1 else 1 - p
        dog_prob = 1 - fav_prob
        bt_ml_pct = get_bt_pct(fav_prob, "ml")
        bt_ats_pct = get_bt_pct(fav_prob, "ats")

        has_vegas = pick["vegas_ml_t1"] is not None
        m_spread = pick["model_spread"]
        m_total = pick["model_total"]

        # Get game time from odds API commence_time
        commence_time = pick.get("commence_time", "")
        game_time_display = ""
        broadcast_display = ""
        if commence_time:
            try:
                ct = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                et = ct - timedelta(hours=4)  # Convert UTC to EDT
                game_time_display = et.strftime("%I:%M %p ET").lstrip("0")
            except Exception:
                game_time_display = ""

        # Try to find broadcast + final score from ESPN data
        game_final = False
        final_home_score = 0
        final_away_score = 0
        final_home_team = ""
        for ekey, edata in espn_broadcast.items():
            if (n1.lower() in ekey and n2.lower() in ekey):
                broadcast_display = edata.get("broadcast", "")
                if not game_time_display and edata.get("start_time"):
                    try:
                        ct = datetime.fromisoformat(edata["start_time"].replace("Z", "+00:00"))
                        et = ct - timedelta(hours=4)
                        game_time_display = et.strftime("%I:%M %p ET").lstrip("0")
                    except Exception:
                        pass
                break

        # Check if game is final for grading
        for ekey, sc in espn_scores.items():
            if n1.lower() in ekey and n2.lower() in ekey:
                if sc["status"] == "STATUS_FINAL":
                    game_final = True
                    final_home_score = sc["home_score"]
                    final_away_score = sc["away_score"]
                    final_home_team = sc["home_team"]
                break

        # Determine actual scores mapped to t1/t2
        t1_final_score = 0
        t2_final_score = 0
        if game_final:
            # Figure out which ESPN team maps to t1 vs t2
            # Try multiple matching strategies; skip grading if uncertain
            home_name = pick.get("home", "")
            mapped = False

            # Strategy 1: Match via odds API home/away names stored in pick
            if home_name:
                if n1.lower() in home_name.lower():
                    t1_final_score = final_home_score
                    t2_final_score = final_away_score
                    mapped = True
                elif n2.lower() in home_name.lower():
                    t1_final_score = final_away_score
                    t2_final_score = final_home_score
                    mapped = True

            # Strategy 2: Match via ESPN home_team name
            if not mapped and final_home_team:
                if n1.lower() in final_home_team.lower():
                    t1_final_score = final_home_score
                    t2_final_score = final_away_score
                    mapped = True
                elif n2.lower() in final_home_team.lower():
                    t1_final_score = final_away_score
                    t2_final_score = final_home_score
                    mapped = True

            # If we couldn't confidently map, don't grade this game
            if not mapped:
                game_final = False

        card = {
            "game": f"{s1t}{n1} vs {s2t}{n2}",
            "ml": None, "spread": None, "total": None,
            "best_rating": 0,
            "commence_time": commence_time,
            "game_time_display": game_time_display,
            "broadcast_display": broadcast_display,
            "game_final": game_final,
            "final_score": f"{t1_final_score}-{t2_final_score}" if game_final else "",
            "t1_final": t1_final_score,
            "t2_final": t2_final_score,
        }

        # ── Moneyline Pick ──
        if has_vegas:
            v_ml_fav = pick["vegas_ml_t1"] if fav == t1 else pick["vegas_ml_t2"]
            v_ml_dog = pick["vegas_ml_t2"] if fav == t1 else pick["vegas_ml_t1"]
            vegas_implied = _implied_prob(v_ml_fav)
            ml_ev = _ev(fav_prob, v_ml_fav)
            ml_kelly = _kelly(fav_prob, v_ml_fav)
            ml_edge = fav_prob - vegas_implied
            ml_score = _edge_rating(max(ml_ev, 0), ml_kelly, bt_ml_pct)
            ml_label, ml_color = _edge_label(ml_score)

            # Also check if dog has value
            dog_ev = _ev(dog_prob, v_ml_dog)
            if dog_ev > ml_ev and dog_ev > 0:
                dog_implied = _implied_prob(v_ml_dog)
                dog_kelly = _kelly(dog_prob, v_ml_dog)
                dog_edge = dog_prob - dog_implied
                dog_score = _edge_rating(dog_ev, dog_kelly, 50)
                dog_label, dog_color = _edge_label(dog_score)
                card["ml"] = {
                    "pick": f"{dog_name} ({'+' if v_ml_dog > 0 else ''}{v_ml_dog})",
                    "model": f"{dog_prob*100:.1f}%", "vegas": f"{dog_implied*100:.1f}%",
                    "edge": f"{dog_edge*100:+.1f}%", "ev": f"${dog_ev:+.1f}",
                    "kelly": f"{dog_kelly*100:.1f}%", "score": dog_score,
                    "label": dog_label, "color": dog_color, "bt": "—",
                }
            else:
                card["ml"] = {
                    "pick": f"{fav_name} ({'+' if v_ml_fav > 0 else ''}{v_ml_fav})",
                    "model": f"{fav_prob*100:.1f}%", "vegas": f"{vegas_implied*100:.1f}%",
                    "edge": f"{ml_edge*100:+.1f}%", "ev": f"${ml_ev:+.1f}",
                    "kelly": f"{ml_kelly*100:.1f}%", "score": ml_score,
                    "label": ml_label, "color": ml_color, "bt": f"{bt_ml_pct:.0f}%",
                }
        else:
            # No Vegas — show model projection
            model_ml = pick["model_ml_t1"] if fav == t1 else pick["model_ml_t2"]
            card["ml"] = {
                "pick": f"{fav_name} ({model_ml})",
                "model": f"{fav_prob*100:.1f}%", "vegas": "—",
                "edge": "—", "ev": "—", "kelly": "—", "score": 0,
                "label": "NO LINE", "color": "#666", "bt": f"{bt_ml_pct:.0f}%",
            }

        # ── Spread Pick ──
        if has_vegas and pick["vegas_spread"] is not None:
            v_spread = pick["vegas_spread"]
            spread_diff = v_spread - m_spread
            spread_ev = abs(spread_diff) * 4.0
            spread_kelly = min(abs(spread_diff) / 20, 0.15)
            spread_score = _edge_rating(spread_ev if abs(spread_diff) >= 1.0 else 0, spread_kelly, bt_ats_pct)
            spread_label, spread_color = _edge_label(spread_score)

            if m_spread < v_spread:
                pick_text = f"{fav_name} {v_spread:+.1f}"
            else:
                pick_text = f"{dog_name} {-v_spread:+.1f}"

            card["spread"] = {
                "pick": pick_text,
                "model": f"{m_spread:+.1f}", "vegas": f"{v_spread:+.1f}",
                "edge": f"{abs(spread_diff):.1f} pts",
                "ev": f"${spread_ev:+.1f}" if abs(spread_diff) >= 1.0 else "—",
                "kelly": f"{spread_kelly*100:.1f}%" if abs(spread_diff) >= 1.0 else "—",
                "score": spread_score if abs(spread_diff) >= 1.0 else 0,
                "label": spread_label if abs(spread_diff) >= 1.0 else "EVEN",
                "color": spread_color if abs(spread_diff) >= 1.0 else "#666",
                "bt": f"{bt_ats_pct:.0f}%",
            }
        else:
            # No Vegas spread — show model only
            card["spread"] = {
                "pick": f"{fav_name} {m_spread:+.1f}",
                "model": f"{m_spread:+.1f}", "vegas": "—",
                "edge": "—", "ev": "—", "kelly": "—", "score": 0,
                "label": "NO LINE", "color": "#666", "bt": f"{bt_ats_pct:.0f}%",
            }

        # ── Over/Under Pick ──
        if has_vegas and pick["vegas_total"] is not None:
            v_total = pick["vegas_total"]
            total_diff = m_total - v_total
            total_ev = abs(total_diff) * 3.0
            total_kelly = min(abs(total_diff) / 25, 0.12)
            total_score = _edge_rating(total_ev if abs(total_diff) >= 2.0 else 0, total_kelly, 55)
            total_label, total_color = _edge_label(total_score)

            ou_pick = "OVER" if m_total > v_total else "UNDER"

            card["total"] = {
                "pick": f"{ou_pick} {v_total}",
                "model": f"{m_total:.1f}", "vegas": f"{v_total:.1f}",
                "edge": f"{abs(total_diff):.1f} pts",
                "ev": f"${total_ev:+.1f}" if abs(total_diff) >= 2.0 else "—",
                "kelly": f"{total_kelly*100:.1f}%" if abs(total_diff) >= 2.0 else "—",
                "score": total_score if abs(total_diff) >= 2.0 else 0,
                "label": total_label if abs(total_diff) >= 2.0 else "EVEN",
                "color": total_color if abs(total_diff) >= 2.0 else "#666",
                "bt": "—",
            }
        else:
            card["total"] = {
                "pick": f"Proj {m_total:.1f}",
                "model": f"{m_total:.1f}", "vegas": "—",
                "edge": "—", "ev": "—", "kelly": "—", "score": 0,
                "label": "NO LINE", "color": "#666", "bt": "—",
            }

        card["best_rating"] = max(
            card["ml"]["score"], card["spread"]["score"], card["total"]["score"]
        )

        # ── Grade picks for final games ──
        if game_final and has_vegas:
            actual_margin = t1_final_score - t2_final_score  # t1 margin
            actual_total = t1_final_score + t2_final_score
            fav_won = (fav == t1 and t1_final_score > t2_final_score) or \
                      (fav == t2 and t2_final_score > t1_final_score)

            # Grade ML — did the model's ML pick win?
            ml_pick_data = card["ml"]
            # Check if model picked the dog instead
            if dog_name in ml_pick_data["pick"]:
                ml_correct = not fav_won  # model picked dog, dog needs to win
            else:
                ml_correct = fav_won
            card["ml"]["result"] = "WIN" if ml_correct else "LOSS"

            # Grade ATS — did the picked side cover?
            if pick["vegas_spread"] is not None:
                v_spread = pick["vegas_spread"]
                # t1 spread: t1_score + v_spread vs t2_score
                t1_covers = (t1_final_score + v_spread) > t2_final_score
                # Model picked t1 side if model spread < vegas spread
                model_took_t1 = m_spread < v_spread
                if model_took_t1:
                    ats_correct = t1_covers
                else:
                    ats_correct = not t1_covers
                # Push check
                if abs((t1_final_score + v_spread) - t2_final_score) < 0.01:
                    card["spread"]["result"] = "PUSH"
                else:
                    card["spread"]["result"] = "WIN" if ats_correct else "LOSS"

            # Grade O/U
            if pick["vegas_total"] is not None:
                v_total = pick["vegas_total"]
                model_took_over = m_total > v_total
                if abs(actual_total - v_total) < 0.01:
                    card["total"]["result"] = "PUSH"
                elif model_took_over:
                    card["total"]["result"] = "WIN" if actual_total > v_total else "LOSS"
                else:
                    card["total"]["result"] = "WIN" if actual_total < v_total else "LOSS"

        game_cards.append(card)

        # Also build flat list for board table
        for btype, data in [("Moneyline", card["ml"]), ("Spread", card["spread"]), ("O/U", card["total"])]:
            all_bets.append({
                "Game": card["game"], "Bet Type": btype,
                "Pick": data["pick"], "Model": data["model"], "Vegas": data["vegas"],
                "Edge": data["edge"], "EV ($100)": data["ev"],
                "Kelly %": data["kelly"], "Signal": data["label"],
                "Rating": data["score"],
                "_color": data["color"],
            })

    # Sort games by tip-off time (next game first), then by edge as tiebreaker
    game_cards.sort(key=lambda x: (x["commence_time"] or "9999", -x["best_rating"]))

    # ── Today's Record Summary ──
    graded = [c for c in game_cards if c.get("game_final")]
    if graded:
        ml_w = sum(1 for c in graded if c["ml"].get("result") == "WIN")
        ml_l = sum(1 for c in graded if c["ml"].get("result") == "LOSS")
        ats_w = sum(1 for c in graded if c["spread"].get("result") == "WIN")
        ats_l = sum(1 for c in graded if c["spread"].get("result") == "LOSS")
        ou_w = sum(1 for c in graded if c["total"].get("result") == "WIN")
        ou_l = sum(1 for c in graded if c["total"].get("result") == "LOSS")
        kc1, kc2, kc3 = st.columns(3)
        ml_c = "#4ade80" if ml_w > ml_l else "#f87171" if ml_l > ml_w else "#fbbf24"
        ats_c = "#4ade80" if ats_w > ats_l else "#f87171" if ats_l > ats_w else "#fbbf24"
        ou_c = "#4ade80" if ou_w > ou_l else "#f87171" if ou_l > ou_w else "#fbbf24"
        with kc1:
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid {ml_c};">'
                f'<div class="label">TODAY\'S ML</div>'
                f'<div class="value" style="color:{ml_c};">{ml_w}-{ml_l}</div>'
                f'</div>', unsafe_allow_html=True)
        with kc2:
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid {ats_c};">'
                f'<div class="label">TODAY\'S ATS</div>'
                f'<div class="value" style="color:{ats_c};">{ats_w}-{ats_l}</div>'
                f'</div>', unsafe_allow_html=True)
        with kc3:
            st.markdown(
                f'<div class="vp-metric" style="border-top:3px solid {ou_c};">'
                f'<div class="label">TODAY\'S O/U</div>'
                f'<div class="value" style="color:{ou_c};">{ou_w}-{ou_l}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Render Game Cards ──
    for card in game_cards:
        best = card["best_rating"]
        border_color = "#4ade80" if best >= 70 else "#fbbf24" if best >= 45 else "#60a5fa" if best >= 25 else "#444"

        # Build game info line (time + broadcast + final score)
        info_parts = []
        is_final = card.get("game_final", False)
        if is_final:
            info_parts.append(f'<span style="color:#4ade80; font-weight:700;">FINAL: {card["t1_final"]}-{card["t2_final"]}</span>')
        if card.get("game_time_display"):
            info_parts.append(f'<span style="color:#fbbf24; font-weight:600;">{card["game_time_display"]}</span>')
        if card.get("broadcast_display"):
            info_parts.append(f'<span style="color:#60a5fa; font-weight:600;">{card["broadcast_display"]}</span>')
        info_line = ""
        if info_parts:
            info_line = (
                f'<div style="font-size:12px; margin-top:2px; display:flex; gap:12px; align-items:center;">'
                + " &middot; ".join(info_parts)
                + '</div>'
            )

        # For final games, show record instead of edge
        if is_final:
            results = [card["ml"].get("result"), card["spread"].get("result"), card["total"].get("result")]
            wins = sum(1 for r in results if r == "WIN")
            losses = sum(1 for r in results if r == "LOSS")
            pushes = sum(1 for r in results if r == "PUSH")
            record_str = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
            record_color = "#4ade80" if wins > losses else "#f87171" if losses > wins else "#fbbf24"
            badge_html = (f'<span style="background:{record_color}; color:#000; font-weight:700; padding:3px 10px; '
                          f'border-radius:4px; font-size:11px; letter-spacing:0.5px;">{record_str}</span>')
        else:
            badge_html = (f'<span style="background:{border_color}; color:#000; font-weight:700; padding:3px 10px; '
                          f'border-radius:4px; font-size:11px; letter-spacing:0.5px;">EDGE: {best}</span>')

        st.markdown(
            f'<div class="vp-bet-card" style="border-left:4px solid {border_color}; border-color:{border_color};">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
            f'<div>'
            f'<div style="font-weight:700; font-size:16px; letter-spacing:-0.3px;">{card["game"]}</div>'
            f'{info_line}'
            f'</div>'
            f'{badge_html}'
            f'</div>'
            f'<div style="display:flex; gap:12px; flex-wrap:wrap;">',
            unsafe_allow_html=True,
        )

        # Three bet type columns
        for btype, data in [("MONEYLINE", card["ml"]), ("SPREAD", card["spread"]), ("O/U", card["total"])]:
            sc = data["score"]
            col = data["color"]
            lbl = data["label"]
            edge_display = data["edge"]

            # Result badge for graded games
            result = data.get("result", "")
            if result == "WIN":
                result_html = ('<div style="margin-top:6px;"><span style="background:#4ade80; color:#000; '
                               'font-weight:800; padding:3px 10px; border-radius:4px; font-size:11px; '
                               'letter-spacing:1px;">✓ WIN</span></div>')
            elif result == "LOSS":
                result_html = ('<div style="margin-top:6px;"><span style="background:#f87171; color:#000; '
                               'font-weight:800; padding:3px 10px; border-radius:4px; font-size:11px; '
                               'letter-spacing:1px;">✗ LOSS</span></div>')
            elif result == "PUSH":
                result_html = ('<div style="margin-top:6px;"><span style="background:#fbbf24; color:#000; '
                               'font-weight:800; padding:3px 10px; border-radius:4px; font-size:11px; '
                               'letter-spacing:1px;">— PUSH</span></div>')
            else:
                result_html = ""

            st.markdown(
                f'<div class="vp-bet-type">'
                f'<div style="font-size:10px; color:#FF6B35; font-weight:700; letter-spacing:1px; margin-bottom:6px;">{btype}</div>'
                f'<div style="font-size:16px; font-weight:700; color:{col}; margin-bottom:8px;">{data["pick"]}</div>'
                f'<div style="font-size:12px; color:#aaa; margin-bottom:3px;">Model: <span style="color:#e2e8f0; font-weight:600;">{data["model"]}</span></div>'
                f'<div style="font-size:12px; color:#aaa; margin-bottom:3px;">Vegas: <span style="color:#e2e8f0; font-weight:600;">{data["vegas"]}</span></div>'
                f'<div style="font-size:12px; color:#aaa; margin-bottom:8px;">Edge: <span style="color:#e2e8f0; font-weight:600;">{edge_display}</span> &nbsp; EV: <span style="color:#e2e8f0; font-weight:600;">{data["ev"]}</span></div>'
                f'<div>'
                f'<span style="background:{col}; color:#000; font-weight:700; padding:3px 10px; '
                f'border-radius:4px; font-size:11px; letter-spacing:0.5px;">{lbl} ({sc})</span></div>'
                f'{result_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Full Bet Board (table) ──
    st.markdown("---")
    st.subheader("Full Bet Board")

    all_bets.sort(key=lambda x: x["Rating"], reverse=True)
    board_rows = [{k: v for k, v in b.items() if not k.startswith("_")} for b in all_bets]
    if board_rows:
        _styled_table(board_rows)

    # ── Legend ──
    with st.expander("How to read this page"):
        st.markdown("""
**Every game gets three picks:** Moneyline (who wins), Spread (margin of victory), and O/U (total points).

**Edge** — How much our model disagrees with Vegas. Bigger = stronger play.

**EV ($100)** — Expected profit on a $100 bet. Positive = profitable long-term.

**Kelly %** — Recommended bet size as % of bankroll. Use half or quarter Kelly for safety.

**Rating** (0-100) — Overall strength combining EV, Kelly, and backtest accuracy.

| Signal | Rating | Action |
|--------|--------|--------|
| **STRONG** | 70+ | High-confidence play |
| **MODERATE** | 45-69 | Standard bet |
| **SLIGHT** | 25-44 | Small position |
| **SKIP/EVEN** | <25 | Pass or no edge |
""")


# ──────────────────────────── Page: About ────────────────────────────

def page_about():
    st.markdown(
        "<div style='text-align:center; margin:40px 0 20px;'>"
        "<span style='font-size:48px; font-weight:900; letter-spacing:-2px;'>"
        "<span style='color:#FF6B35;'>VIL</span><span style='color:#FAFAFA;'>POM</span></span>"
        "</div>"
        "<div style='text-align:center; color:#888; font-size:12px; letter-spacing:3px; font-weight:600; margin-bottom:40px;'>"
        "NCAA TOURNAMENT ANALYTICS & BETTING INTELLIGENCE</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="vp-metric">'
            '<div class="label">MODELS</div>'
            '<div class="value" style="color:#FF6B35;">3</div>'
            '<div class="sub">XGBoost + CatBoost + LightGBM</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="vp-metric">'
            '<div class="label">TRAINING DATA</div>'
            '<div class="value" style="color:#4ade80;">20+ YRS</div>'
            '<div class="sub">Tournament games since 2003</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="vp-metric">'
            '<div class="label">SIMULATIONS</div>'
            '<div class="value" style="color:#60a5fa;">10K</div>'
            '<div class="sub">Monte Carlo per bracket</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown('<div class="vp-section">THE ENGINE</div>', unsafe_allow_html=True)
    st.markdown("""
Vilpom's ensemble model processes **50+ features per team** including efficiency ratings,
strength of schedule, coaching track records, Elo trajectories, and shooting profiles.
Three gradient-boosted models independently evaluate every matchup, then consensus
probabilities drive spreads, totals, and moneylines.
""")

    st.markdown('<div class="vp-section">WHAT YOU GET</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#FF6B35; margin-bottom:6px;">Power Rankings</div>'
            '<div style="color:#aaa; font-size:13px;">Elo ratings, net efficiency, and tier classifications '
            'updated through the full regular season.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#4ade80; margin-bottom:6px;">Live Betting Picks</div>'
            '<div style="color:#aaa; font-size:13px;">Model vs. Vegas comparisons with edge ratings, EV, '
            'and Kelly sizing on every available market.</div></div>',
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#60a5fa; margin-bottom:6px;">Head-to-Head Analysis</div>'
            '<div style="color:#aaa; font-size:13px;">Stat comparisons, projected lines, similar opponent '
            'analysis, and coaching matchup data.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#fbbf24; margin-bottom:6px;">Bracket & Futures</div>'
            '<div style="color:#aaa; font-size:13px;">Full bracket predictions, championship futures odds, '
            'and round-by-round advancement probabilities.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="vp-section">GLOSSARY</div>', unsafe_allow_html=True)
    st.markdown(
        '<table class="vp-table"><thead><tr>'
        '<th>TERM</th><th>DEFINITION</th></tr></thead><tbody>'
        '<tr><td style="font-weight:600; color:#FF6B35;">Spread</td>'
        '<td>Projected margin of victory. Negative = favored.</td></tr>'
        '<tr><td style="font-weight:600; color:#FF6B35;">Moneyline</td>'
        '<td>Odds to win outright. Negative = favorite, positive = underdog.</td></tr>'
        '<tr><td style="font-weight:600; color:#FF6B35;">Over/Under</td>'
        '<td>Projected combined score of both teams.</td></tr>'
        '<tr><td style="font-weight:600; color:#FF6B35;">Edge Rating</td>'
        '<td>Confidence score 0-100. Higher = stronger play.</td></tr>'
        '<tr><td style="font-weight:600; color:#FF6B35;">EV</td>'
        '<td>Expected profit on a $100 bet. Positive = long-term profitable.</td></tr>'
        '<tr><td style="font-weight:600; color:#FF6B35;">Kelly %</td>'
        '<td>Optimal bet size as % of bankroll based on edge.</td></tr>'
        '</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:12px; padding:20px;'>"
        "For entertainment purposes only. Past performance does not guarantee future results. "
        "Not professional financial advice. Please gamble responsibly.</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────── Main ────────────────────────────

teams = load_teams()
preds = load_predictions()
seeds = load_seeds()
m_slots, w_slots = load_slots()
conferences = load_conferences()

# Sidebar
st.sidebar.markdown(
    "<div style='text-align:center; margin-bottom:8px;'>"
    "<span style='font-size:28px; font-weight:900; letter-spacing:-1px;'>"
    "<span style='color:#FF6B35;'>VIL</span><span style='color:#FAFAFA;'>POM</span></span>"
    "</div>"
    "<div style='text-align:center; color:#666; font-size:10px; letter-spacing:2px; font-weight:600;'>"
    "ANALYTICS & INTELLIGENCE</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], horizontal=True)
prefix = "M" if gender == "Men's" else "W"


gender_seeds = seeds[seeds.TeamID < 3000] if prefix == "M" else seeds[seeds.TeamID >= 3000]
gender_slots = m_slots if prefix == "M" else w_slots

page = st.sidebar.radio(
    "Navigate",
    ["\U0001f4b0 Betting Picks", "\U0001f93c Head-to-Head", "\U0001f4ca Rankings",
     "\U0001f3c6 Bracket", "\U0001f9ea Backtest", "\u2139\ufe0f About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:#444; font-size:10px; letter-spacing:1px;'>"
    "VILPOM &copy; 2026<br>For entertainment only. Not financial advice.</div>",
    unsafe_allow_html=True,
)

if "Betting Picks" in page:
    page_picks(prefix, teams, gender_seeds, preds)
elif "Head-to-Head" in page:
    coach_info = load_coach_data() if prefix == "M" else {}
    knn_data = build_knn_lookup(prefix)
    h2h_history = build_h2h_history(prefix)
    seed_history = build_seed_history(prefix)
    page_h2h(prefix, teams, gender_seeds, preds, coach_info, knn_data, h2h_history, seed_history)
elif "Rankings" in page:
    page_rankings(prefix, teams, gender_seeds, conferences)
elif "Bracket" in page:
    page_bracket(prefix, teams, gender_seeds, gender_slots, preds)
elif "Backtest" in page:
    page_backtest(prefix, teams)
elif "About" in page:
    page_about()
