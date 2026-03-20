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
        "<div style='text-align:center; margin-top:100px;'>"
        "<div style='font-size:64px; font-weight:900; letter-spacing:-3px; margin-bottom:6px;'>"
        "<span style='color:#41B6E6;'>VIL</span><span style='color:#FAFAFA;'>POM</span></div>"
        "<div style='width:50px; height:2px; background:linear-gradient(90deg, #41B6E6, #6dcbf2); "
        "margin:12px auto;'></div>"
        "<div style='color:#555; font-size:10px; letter-spacing:4px; font-weight:700;'>"
        "NCAA ANALYTICS & BETTING INTELLIGENCE</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    # CSS to disable copy/paste/select on the password field
    st.markdown(
        """<style>
        input[type="password"] {
            -webkit-user-select: none; -moz-user-select: none;
            -ms-user-select: none; user-select: none;
            background: rgba(18, 20, 26, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important; padding: 14px !important;
            font-size: 14px !important; letter-spacing: 3px !important;
            backdrop-filter: blur(8px) !important;
            transition: all 0.25s ease !important;
        }
        input[type="password"]:focus {
            border-color: rgba(65, 182, 230, 0.5) !important;
            box-shadow: 0 0 0 1px rgba(65, 182, 230, 0.3), 0 4px 16px rgba(65, 182, 230, 0.1) !important;
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
            "<div style='text-align:center; color:#333; font-size:9px; letter-spacing:2px; "
            "font-weight:700; margin-top:24px;'>"
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
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* ── Fix broken Material Icons rendering as text ── */
    .material-icons,
    [data-testid="stExpander"] summary svg,
    [class*="material-icons"] {
        font-family: 'Material Icons' !important;
        -webkit-font-feature-settings: 'liga';
        font-feature-settings: 'liga';
        font-size: 24px !important;
        overflow: hidden;
        max-width: 24px;
        display: inline-block !important;
    }
    /* Fallback: if font still fails, hide the text and use CSS arrows */
    @font-face {
        font-family: 'Material Icons Detect';
        src: url('https://fonts.gstatic.com/s/materialicons/v140/flUhRq6tzZclQEJ-Vdg-IuiaDsNc.woff2') format('woff2');
    }
    [data-testid="stExpander"] details > summary > span:first-of-type {
        font-size: 0 !important; width: 24px; height: 24px;
        display: inline-flex !important; align-items: center; justify-content: center;
        overflow: hidden;
    }
    [data-testid="stExpander"] details > summary > span:first-of-type::after {
        content: "▾"; font-size: 14px; color: #888;
    }
    [data-testid="stExpander"] details[open] > summary > span:first-of-type::after {
        content: "▴";
    }

    /* ── Foundation ── */
    .block-container { padding-top: 0.5rem; max-width: 1200px; }
    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0a0c10;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] .stRadio label { font-weight: 500; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 1px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        padding: 7px 12px !important; border-radius: 6px !important;
        font-size: 13px !important; border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
        background: rgba(255, 255, 255, 0.03) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label[aria-checked="true"] {
        background: rgba(65, 182, 230, 0.08) !important;
        border-color: rgba(65, 182, 230, 0.2) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #2a2a2a; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; }

    /* ── Card ── */
    .vp-card {
        background: #18191f; border: 1px solid #2a2a2a; border-radius: 8px;
        padding: 14px 18px; margin: 6px 0;
    }

    /* ── Section headers ── */
    .vp-section {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        color: #41B6E6; text-transform: uppercase; margin-bottom: 10px;
        padding-bottom: 6px; border-bottom: 1px solid #2a2a2a;
    }

    /* ── Page header ── */
    .vp-page-header { padding: 8px 0 10px; margin-bottom: 4px; }
    .vp-page-header h1 {
        font-size: 22px !important; font-weight: 800 !important;
        color: #FAFAFA !important; margin: 0 0 2px !important; line-height: 1.3 !important;
    }
    .vp-page-header .subtitle { font-size: 12px; color: #555; font-weight: 500; }

    /* ── Subheaders ── */
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 14px !important; font-weight: 700 !important;
        color: #e2e8f0 !important;
        border-bottom: 1px solid #2a2a2a; padding-bottom: 8px; margin-top: 8px;
    }
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 17px !important; font-weight: 800 !important; color: #FAFAFA !important;
    }

    /* ── Captions ── */
    [data-testid="stCaptionContainer"] p { font-size: 11px !important; color: #555 !important; }

    /* ── Inputs ── */
    [data-testid="stSelectbox"] label, [data-testid="stTextInput"] label,
    [data-testid="stCheckbox"] label, [data-testid="stSlider"] label,
    [data-testid="stMultiSelect"] label {
        font-size: 11px !important; font-weight: 700 !important;
        letter-spacing: 0.5px; text-transform: uppercase; color: #666 !important;
    }

    /* ── Dividers ── */
    hr { border-color: #2a2a2a !important; margin: 16px 0 !important; }

    /* ── Radio ── */
    [data-testid="stRadio"] > div > label { font-size: 13px !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #41B6E6 !important; color: #fff !important; font-weight: 700 !important;
        border: none !important; border-radius: 6px !important;
        padding: 8px 24px !important;
    }
    .stButton > button:hover { background: #2d95c4 !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #41B6E6 !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #2a2a2a !important; border-radius: 8px !important;
        background: #18191f;
    }
    [data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 13px !important; }

    /* ── Table ── */
    .vp-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px; }
    .vp-table thead th {
        background: #131418; color: #777; font-size: 9px; font-weight: 700;
        letter-spacing: 0.8px; text-transform: uppercase; padding: 7px 8px;
        border-bottom: 2px solid #41B6E6; position: sticky; top: 0; z-index: 1;
        text-align: left; white-space: nowrap;
    }
    .vp-table tbody tr:hover { background: rgba(255, 255, 255, 0.02); }
    .vp-table tbody td {
        padding: 5px 8px; border-bottom: 1px solid #2a2a2a;
        font-size: 12px; font-variant-numeric: tabular-nums;
    }
    .vp-table .rank-cell { font-weight: 700; color: #41B6E6; font-size: 12px; width: 28px; text-align: center; }
    .vp-table .team-cell { font-weight: 600; color: #FAFAFA; white-space: nowrap; }
    .vp-table .seed-badge {
        display: inline-block; background: #41B6E6; color: #fff; font-weight: 700;
        font-size: 10px; padding: 1px 6px; border-radius: 3px; min-width: 18px; text-align: center;
    }
    .vp-table .stat-good { color: #4ade80; }
    .vp-table .stat-bad { color: #f87171; }
    .vp-table .stat-neutral { color: #888; }

    /* ── Tier badges ── */
    .tier-elite { color: #fff; background: #41B6E6; padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 10px; }
    .tier-strong { color: #000; background: #4ade80; padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 10px; }
    .tier-solid { color: #000; background: #60a5fa; padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 10px; }
    .tier-avg { color: #000; background: #666; padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 10px; }

    /* ── Bracket (NCAA Official Style on dark bg) ── */
    .bracket { display: flex; gap: 8px; min-height: 540px; overflow-x: auto; padding: 4px 0; }
    .round { display: flex; flex-direction: column; justify-content: space-around; min-width: 175px; }
    .round-title {
        text-align: center; font-size: 10px; color: #aaa;
        margin-bottom: 6px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    }
    .matchup {
        border: 1px solid #333; border-radius: 3px; overflow: hidden;
        margin: 2px 0; background: #18191f; position: relative;
    }
    .matchup:hover { border-color: #555; }
    .matchup.game-miss { border-color: #f87171; }
    .matchup.game-hit { border-color: #2d3a2d; }
    .team-slot {
        padding: 5px 8px; font-size: 12px; display: flex;
        align-items: center; gap: 0;
        border-bottom: 1px solid #2a2a2a; color: #999;
    }
    .team-slot:last-child { border-bottom: none; }
    .team-slot.winner { background: rgba(255, 255, 255, 0.04); }
    .team-slot.winner .bk-name { color: #FAFAFA; font-weight: 700; }
    .team-slot.winner .bk-seed { color: #FAFAFA; }
    .team-slot.actual-winner { background: rgba(74, 222, 128, 0.06); }
    .team-slot.actual-winner .bk-name { color: #4ade80; font-weight: 700; }
    .team-slot.actual-winner .bk-seed { color: #4ade80; }
    .team-slot.actual-loser .bk-name { color: #555; }
    .team-slot.upset-loser .bk-name { color: #555; text-decoration: line-through; text-decoration-color: #f8717166; }
    .bk-seed {
        color: #888; font-size: 10px; font-weight: 700; min-width: 18px;
        text-align: right; margin-right: 6px; font-variant-numeric: tabular-nums;
    }
    .bk-name {
        flex: 1; font-weight: 600; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; font-size: 12px;
    }
    .bk-score {
        font-size: 13px; font-weight: 800; font-variant-numeric: tabular-nums;
        margin-left: auto; padding-left: 8px; color: #555; min-width: 24px; text-align: right;
    }
    .bk-score.win { color: #FAFAFA; }
    .matchup-badge {
        position: absolute; top: -1px; right: -1px; font-size: 8px; font-weight: 800;
        padding: 1px 5px; border-radius: 0 3px 0 3px; letter-spacing: 0.5px; z-index: 1;
    }
    .matchup-badge.miss { background: #f87171; color: #000; }
    .matchup-badge.hit { background: #2d3a2d; color: #4ade80; }
    .prob-tag { color: #555; font-size: 10px; font-variant-numeric: tabular-nums; margin-left: auto; padding-left: 6px; }
    .big-prob { font-size: 36px; font-weight: 800; text-align: center; line-height: 1; margin: 4px 0; letter-spacing: -1px; }
    .stat-bar { height: 4px; border-radius: 2px; margin: 2px 0; }
    .ff-bracket { display: flex; gap: 12px; min-height: 160px; align-items: center; }
    .ff-round { display: flex; flex-direction: column; justify-content: space-around; min-width: 180px; min-height: 150px; }
    .champ-banner {
        text-align: center; font-size: 22px; font-weight: 800;
        padding: 16px; border: 1px solid #41B6E6; border-radius: 8px;
        background: rgba(65, 182, 230, 0.06); color: #41B6E6; margin: 8px 0;
    }

    /* ── Metric cards ── */
    .vp-metric {
        background: #18191f; border: 1px solid #2a2a2a; border-radius: 8px;
        padding: 12px; text-align: center;
    }
    .vp-metric .label { color: #666; font-size: 9px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
    .vp-metric .value { font-size: 24px; font-weight: 800; margin: 4px 0; letter-spacing: -0.5px; }
    .vp-metric .sub { color: #666; font-size: 11px; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Bet cards ── */
    .vp-bet-card {
        background: #18191f; border: 1px solid #2a2a2a; border-radius: 8px;
        padding: 14px 18px; margin: 8px 0;
    }
    .vp-bet-type {
        background: #131418; border-radius: 6px; padding: 10px 14px;
        flex: 1; min-width: 160px; border: 1px solid #2a2a2a;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

    /* ── Progress bar ── */
    .stProgress > div > div { background: #2a2a2a !important; border-radius: 4px !important; }
    .stProgress > div > div > div { background: #41B6E6 !important; border-radius: 4px !important; }

    /* ── Score Bug (TV Broadcast Style) ── */
    .score-bug {
        background: #18191f; border: 1px solid #2a2a2a; border-radius: 4px;
        overflow: hidden; margin: 4px 0;
    }
    .score-bug .team-row {
        display: flex; align-items: center; padding: 6px 10px;
        border-bottom: 1px solid #2a2a2a; gap: 6px;
    }
    .score-bug .team-row:last-child { border-bottom: none; }
    .score-bug .team-row .color-bar {
        width: 4px; height: 100%; min-height: 28px; border-radius: 1px;
        flex-shrink: 0;
    }
    .score-bug .seed { color: #888; font-size: 11px; font-weight: 700;
        min-width: 18px; text-align: right; font-variant-numeric: tabular-nums; }
    .score-bug .team-name { flex: 1; font-weight: 600; font-size: 13px;
        color: #FAFAFA; letter-spacing: 0.3px; }
    .score-bug .team-name.loser { color: #555; }
    .score-bug .score { font-size: 18px; font-weight: 800; color: #FAFAFA;
        font-variant-numeric: tabular-nums; min-width: 32px; text-align: right; }
    .score-bug .score.loser { color: #555; }
    .score-bug .game-info {
        display: flex; align-items: center; gap: 8px; padding: 4px 10px;
        background: #131418; font-size: 10px; color: #666;
        border-top: 1px solid #2a2a2a;
    }

    /* ── Mobile ── */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem 0.8rem !important; }
        .vp-page-header h1 { font-size: 18px !important; }
        .big-prob { font-size: 28px !important; }
        .bracket { gap: 4px; }
        .round { min-width: 140px; }
        .team-slot { font-size: 10px; padding: 3px 6px; }
        .bk-seed { font-size: 8px; min-width: 14px; }
        .team-slot img { width: 14px !important; height: 14px !important; }
        .vp-bet-card { padding: 10px 12px; }
        .vp-bet-type { min-width: 120px; padding: 8px 10px; }
        .vp-metric .value { font-size: 20px; }
        .ff-bracket { flex-direction: column; align-items: stretch; }
        .ff-round { min-width: auto; }
        .vp-table thead th { font-size: 8px; padding: 6px 4px; }
        .vp-table tbody td { font-size: 11px; padding: 4px; }
        .score-bug .team-name { font-size: 11px; }
        .score-bug .score { font-size: 15px; }
    }

    /* ── Badges ── */
    .vp-badge {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 700;
    }
    .vp-badge-live { background: rgba(65, 182, 230, 0.15); color: #41B6E6; }
    .vp-badge-final { background: rgba(74, 222, 128, 0.1); color: #4ade80; }
    .vp-badge-win { background: rgba(74, 222, 128, 0.15); color: #4ade80; }
    .vp-badge-loss { background: rgba(248, 113, 113, 0.15); color: #f87171; }

    /* ── Divider ── */
    .vp-divider { height: 1px; margin: 16px 0; background: #2a2a2a; }
</style>
""", unsafe_allow_html=True)


def _styled_table(rows, columns=None, max_height=None, espn_map=None):
    """Render a list of dicts as a premium styled HTML table.
    If rows contain '_tid' and espn_map is provided, 'Team' column gets logo + color bar."""
    if not rows:
        return
    if columns is None:
        columns = list(rows[0].keys())
    # Filter out internal columns starting with _
    columns = [c for c in columns if not c.startswith("_")]
    has_team_enrichment = espn_map is not None and rows and "_tid" in rows[0]
    h = max_height or 0
    wrapper_style = f'max-height:{h}px; overflow-y:auto; ' if h else ''
    html = f'<div style="{wrapper_style}border-radius:10px; border:1px solid #333;">'
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
            # Enrich "Team" column with logo + color bar
            if c == "Team" and has_team_enrichment:
                tid = r.get("_tid")
                tc = _team_color(tid) if tid else '#666'
                logo = _team_logo_img(tid, espn_map, size=16) if tid else ''
                html += (
                    f'<td class="team-cell" style="border-left:3px solid {tc}; padding-left:8px;">'
                    f'{logo}{val}</td>'
                )
            else:
                if sval.startswith('+') or 'HIT' in sval:
                    style = ' style="color:#4ade80; font-weight:600;"'
                elif 'MISS' in sval:
                    style = ' style="color:#f87171; font-weight:600;"'
                html += f'<td{style}>{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)


def _styled_df(df, max_height=None, espn_map=None):
    """Convert a pandas DataFrame to a premium styled HTML table."""
    rows = df.to_dict('records')
    columns = list(df.columns)
    _styled_table(rows, columns, max_height, espn_map=espn_map)


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


_CONF_DISPLAY = {
    "acc": "ACC", "big_ten": "B10", "big_twelve": "B12", "sec": "SEC",
    "big_east": "BE", "pac_twelve": "P12", "aac": "AAC", "mwc": "MWC",
    "wcc": "WCC", "mvc": "MVC", "a_ten": "A10", "colonial": "CAA",
    "cusa": "CUSA", "horizon": "Horz", "ivy": "Ivy", "maac": "MAAC",
    "mac": "MAC", "meac": "MEAC", "ovc": "OVC", "patriot": "Pat",
    "southern": "SoCon", "sun_belt": "SBelt", "swac": "SWAC",
    "wac": "WAC", "big_sky": "BSky", "big_south": "BSth",
    "big_west": "BWst", "america_east": "AE", "atlantic_sun": "ASun",
    "northeast": "NE", "southland": "Slnd", "summit": "Sum",
    "ind": "Ind", "mountain_west": "MWC", "west_coast": "WCC",
}


def conf_display(abbrev):
    """Convert raw conference abbreviation to display name."""
    return _CONF_DISPLAY.get(abbrev, abbrev.upper() if abbrev else "")


# ──────────────────────────── Massey Composite Rankings ────────────────────────────

@st.cache_data
def load_massey_ranks():
    """Load Massey ordinals and compute composite rank per team for current season.
    Uses latest available ranking day per system (before tournament, day <= 133)."""
    path = os.path.join(DATA_DIR, "MMasseyOrdinals.csv")
    if not os.path.exists(path):
        return {}
    massey = pd.read_csv(path)
    massey = massey[(massey.Season == SEASON) & (massey.RankingDayNum <= 133)]
    if massey.empty:
        return {}
    # Use only the latest ranking day per system
    last_day = massey.groupby("SystemName")["RankingDayNum"].transform("max")
    massey = massey[massey.RankingDayNum == last_day]
    agg = massey.groupby("TeamID").agg(
        MasseyRank=("OrdinalRank", "mean"),
        MasseyMedian=("OrdinalRank", "median"),
        MasseyBest=("OrdinalRank", "min"),
        MasseyWorst=("OrdinalRank", "max"),
        MasseyStd=("OrdinalRank", "std"),
        MasseySystems=("SystemName", "nunique"),
    )
    agg["MasseyStd"] = agg["MasseyStd"].fillna(0)
    return agg.to_dict("index")


# ──────────────────────────── Conference Strength ────────────────────────────

@st.cache_data
def load_conf_strength(prefix):
    """Compute average win percentage per conference for current season."""
    stats = compute_season_stats(prefix)
    tc = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}TeamConferences.csv"))
    tc = tc[tc.Season == SEASON]
    merged = tc.merge(stats[["WinPct"]].reset_index(), on="TeamID", how="left")
    conf_agg = merged.groupby("ConfAbbrev").agg(
        AvgWinPct=("WinPct", "mean"),
    ).reset_index()
    # Map team -> conf strength
    result = tc.merge(conf_agg, on="ConfAbbrev", how="left")
    team_conf_str = dict(zip(result.TeamID, result.AvgWinPct.round(3)))
    conf_vals = dict(zip(conf_agg.ConfAbbrev, conf_agg.AvgWinPct.round(3)))
    return team_conf_str, conf_vals


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
                "TeamID": int(r[f"{side}TeamID"]), "OppID": int(r[f"{opp_side}TeamID"]),
                "Win": win,
                "DayNum": int(r["DayNum"]),
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
    stats["FGPct"] = (stats["FGM"] / stats["FGA"].clip(lower=1) * 100).round(1)
    stats["FG3Pct"] = (stats["FGM3"] / stats["FGA3"].clip(lower=1) * 100).round(1)
    stats["FTPct"] = (stats["FTM"] / stats["FTA"].clip(lower=1) * 100).round(1)
    stats["RPG"] = stats["ORB"] + stats["DRB"]
    stats["EffFGPct"] = ((stats["FGM"] + 0.5 * stats["FGM3"]) / stats["FGA"].clip(lower=1) * 100).round(1)

    stats["Poss"] = (stats["FGA"] - stats["ORB"] + stats["TOPG"] + 0.475 * stats["FTA"]).clip(lower=1)
    stats["OppPoss"] = (stats["OppFGA"] - stats["OppOR"] + stats["OppTO"] + 0.475 * stats["OppFTA"]).clip(lower=1)
    stats["OffEff"] = (stats["PPG"] / stats["Poss"] * 100).round(1)
    stats["DefEff"] = (stats["OppPPG"] / stats["OppPoss"] * 100).round(1)
    stats["NetEff"] = (stats["OffEff"] - stats["DefEff"]).round(1)
    stats["Tempo"] = ((stats["Poss"] + stats["OppPoss"]) / 2).round(1)

    # Last 10 games momentum
    tg_sorted = tg.sort_values("DayNum")
    last10 = tg_sorted.groupby("TeamID").tail(10)
    l10_agg = last10.groupby("TeamID").agg(
        L10_Wins=("Win", "sum"),
        L10_Games=("Win", "count"),
        L10_Margin=("Score", "mean"),
        L10_OppScore=("OppScore", "mean"),
    )
    l10_agg["L10_Margin"] = (l10_agg["L10_Margin"] - l10_agg["L10_OppScore"]).round(1)
    l10_agg["L10_Losses"] = l10_agg["L10_Games"] - l10_agg["L10_Wins"]
    stats = stats.join(l10_agg[["L10_Wins", "L10_Losses", "L10_Margin"]])

    # Strength of Schedule: average opponent win percentage
    opp_wp = tg.merge(stats[["WinPct"]].rename(columns={"WinPct": "OppWinPct"}),
                       left_on="OppID", right_index=True, how="left")
    sos = opp_wp.groupby("TeamID")["OppWinPct"].mean().round(3)
    sos.name = "SOS"
    stats = stats.join(sos)

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
        "Total Rebounds": round((st.RPG + (st.RPG * (so.RPG / max(stats["RPG"].mean(), 1)))) / 2, 1),
        "Assists": round((st.APG + (st.APG * (so.OppPPG / max(stats["OppPPG"].mean(), 1)))) / 2, 1),
        "3-Pointers Made": round(st.FGM3, 1),
        "Steals": round((st.SPG + (st.SPG * (so.TOPG / max(stats["TOPG"].mean(), 0.1)))) / 2, 1),
        "Blocks": round(st.BPG, 1),
        "Turnovers": round((st.TOPG + (st.TOPG * (so.SPG / max(stats["SPG"].mean(), 0.1)))) / 2, 1),
    }
    return props


def compute_betting_lines(stats, preds, t1, t2):
    """Compute full betting line package for a matchup."""
    p = get_pred(preds, t1, t2)
    spread = prob_to_spread(p)
    total = project_game_total(stats, t1, t2)
    # Derive individual scores from total + spread so they're consistent:
    # t1_pts - t2_pts == -spread (spread negative = t1 favored = t1 scores more)
    # t1_pts + t2_pts == total
    t1_pts = round((total - spread) / 2 * 2) / 2  # round to 0.5
    t2_pts = round((total + spread) / 2 * 2) / 2

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


def simulate_bracket(seeds_df, slots_df, preds, deterministic=True, actual_winners=None):
    """Simulate bracket. If actual_winners is provided (dict: frozenset({tid1,tid2}) -> winner_tid),
    use actual results for completed games and model predictions for the rest."""
    actual_winners = actual_winners or {}
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

        # Use actual result if available
        matchup_key = frozenset({t1, t2})
        if matchup_key in actual_winners:
            winner = actual_winners[matchup_key]
        elif deterministic:
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

def _build_espn_id_map(teams, prefix="M"):
    """Map our team IDs to ESPN team IDs for logo display."""
    # Hardcoded mapping for 2026 tournament teams (Kaggle ID -> ESPN ID)
    KAGGLE_TO_ESPN = {
        1104: "333", 1103: "2006", 1112: "12", 1116: "8", 1140: "252",
        1155: "228", 1163: "41", 1181: "150", 1196: "57", 1202: "231",
        1208: "61", 1211: "2250", 1218: "62", 1219: "2272", 1220: "2275",
        1222: "248", 1224: "47", 1225: "70", 1228: "356", 1234: "2294",
        1235: "66", 1242: "2305", 1244: "338", 1246: "96", 1250: "2329",
        1254: "112358", 1257: "97", 1270: "2377", 1274: "2390", 1275: "193",
        1276: "130", 1277: "127", 1281: "142", 1295: "2449", 1301: "152",
        1304: "158", 1314: "153", 1320: "2460", 1326: "194", 1335: "219",
        1341: "2504", 1345: "2509", 1365: "2541", 1373: "2561", 1374: "2567",
        1378: "58", 1385: "2599", 1387: "139", 1388: "2608", 1395: "2628",
        1397: "2633", 1398: "2634", 1400: "251", 1401: "245", 1403: "2641",
        1407: "2653", 1416: "2116", 1417: "26", 1420: "2378", 1429: "328",
        1433: "2670", 1435: "238", 1437: "222", 1438: "258", 1458: "275",
        1460: "2750", 1465: "2856", 1474: "2511",
    }
    # Also try to augment from ESPN cached data for any teams not in the hardcoded map
    espn_data = _load_cached_espn()
    if espn_data:
        odds_map, name_to_tid = _build_odds_name_map(teams, prefix)
        for g in espn_data.get("games", []):
            for name_key, id_key in [("home_team", "home_id"), ("away_team", "away_id")]:
                espn_name = g.get(name_key, "")
                espn_id = g.get(id_key, "")
                if espn_name and espn_id:
                    tid = _resolve_odds_team(espn_name, odds_map, name_to_tid)
                    if tid and tid not in KAGGLE_TO_ESPN:
                        KAGGLE_TO_ESPN[tid] = espn_id
    return KAGGLE_TO_ESPN


def _team_logo_img(tid, espn_map, size=18):
    """Return an <img> tag for a team's ESPN logo, or empty string if unavailable."""
    espn_id = espn_map.get(tid)
    if not espn_id:
        return ""
    return (
        f'<img src="https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/{espn_id}.png'
        f'&h={size*2}&w={size*2}" width="{size}" height="{size}" '
        f'style="vertical-align:middle; margin-right:4px; border-radius:2px;" '
        f'onerror="this.style.display=\'none\'">'
    )


TEAM_COLORS = {
    1104: "#9E1B32", 1103: "#041E42", 1112: "#CC0033", 1116: "#9D2235",
    1140: "#002E5D", 1155: "#F56600", 1163: "#000E2F", 1181: "#003087",
    1196: "#0021A5", 1202: "#582C83", 1208: "#BA0C2F", 1211: "#002967",
    1218: "#024731", 1219: "#330072", 1220: "#00529C", 1222: "#C8102E",
    1224: "#003A63", 1225: "#B5A36A", 1228: "#13294B", 1234: "#FFCD00",
    1235: "#C8102E", 1242: "#0051BA", 1244: "#FDBB30", 1246: "#0033A0",
    1250: "#653819", 1254: "#00508F", 1257: "#AD0000", 1270: "#00527C",
    1274: "#F47321", 1275: "#B61E2E", 1276: "#00274C", 1277: "#18453B",
    1281: "#F1B82D", 1295: "#006A31", 1301: "#CC0000", 1304: "#E41C38",
    1314: "#7BAFD4", 1320: "#4B116F", 1326: "#BB0000", 1335: "#011F5B",
    1341: "#4F2D7F", 1345: "#CEB888", 1365: "#862633", 1373: "#006747",
    1374: "#CC0035", 1378: "#006747", 1385: "#BA0C2F", 1387: "#003DA5",
    1388: "#003DA5", 1395: "#4D1979", 1397: "#FF8200", 1398: "#00539F",
    1400: "#BF5700", 1401: "#500000", 1403: "#CC0000", 1407: "#8B2332",
    1416: "#BA9B37", 1417: "#2D68C4", 1420: "#000000", 1429: "#00263A",
    1433: "#F8B800", 1435: "#866D4B", 1437: "#003366", 1438: "#232D4B",
    1458: "#C5050C", 1460: "#006A4E", 1465: "#002554", 1474: "#002D6C",
}


def _team_color(tid):
    """Return team's primary color hex, or a neutral gray fallback."""
    return TEAM_COLORS.get(tid, "#666")


def matchup_html(t1, t2, t1_prob, winner, teams, team_seed_map, game_state=None, espn_map=None):
    """Render a bracket matchup card with team logos and seeds."""
    espn_map = espn_map or {}
    s1 = team_seed_map.get(t1, "")
    s2 = team_seed_map.get(t2, "")
    n1 = tname(teams, t1)[:16] if t1 else "TBD"
    n2 = tname(teams, t2)[:16] if t2 else "TBD"
    s1_tag = f'<span class="bk-seed">{s1}</span>' if s1 else ""
    s2_tag = f'<span class="bk-seed">{s2}</span>' if s2 else ""
    logo1 = _team_logo_img(t1, espn_map) if t1 else ""
    logo2 = _team_logo_img(t2, espn_map) if t2 else ""

    is_miss = isinstance(game_state, dict) and game_state.get("type") == "miss"
    is_hit = isinstance(game_state, dict) and game_state.get("type") == "hit"
    is_completed = is_miss or is_hit
    score = game_state.get("score", "") if isinstance(game_state, dict) else ""
    actual_winner = game_state.get("actual_winner") if isinstance(game_state, dict) else None

    # Card-level class
    card_cls = "game-miss" if is_miss else ("game-hit" if is_hit else "")
    badge = ""
    if is_miss:
        badge = '<span class="matchup-badge miss">MISS</span>'
    elif is_hit:
        badge = '<span class="matchup-badge hit">\u2713</span>'

    if is_completed and actual_winner:
        if t1 == actual_winner:
            c1 = "actual-winner"
            c2 = "upset-loser" if is_miss else "actual-loser"
        else:
            c2 = "actual-winner"
            c1 = "upset-loser" if is_miss else "actual-loser"
        if score:
            w_score, l_score = score.split("-")
            sc1 = f'<span class="bk-score win">{w_score}</span>' if t1 == actual_winner else f'<span class="bk-score">{l_score}</span>'
            sc2 = f'<span class="bk-score win">{w_score}</span>' if t2 == actual_winner else f'<span class="bk-score">{l_score}</span>'
        else:
            sc1 = ""
            sc2 = ""
    else:
        c1 = "winner" if t1 == winner else ""
        c2 = "winner" if t2 == winner else ""
        p1 = f"{t1_prob*100:.0f}%" if t1 and t2 else ""
        p2 = f"{(1-t1_prob)*100:.0f}%" if t1 and t2 else ""
        sc1 = f'<span class="prob-tag">{p1}</span>'
        sc2 = f'<span class="prob-tag">{p2}</span>'

    # Team color bars — grey out losers/projected losers
    tc1 = _team_color(t1) if t1 else '#333'
    tc2 = _team_color(t2) if t2 else '#333'
    if is_completed and actual_winner:
        bar1 = tc1 if t1 == actual_winner else '#333'
        bar2 = tc2 if t2 == actual_winner else '#333'
    else:
        bar1 = tc1 if t1 == winner else '#333'
        bar2 = tc2 if t2 == winner else '#333'
    bar_style = 'width:3px; min-height:22px; border-radius:1px; margin-right:5px; flex-shrink:0;'

    return f"""<div class="matchup {card_cls}">{badge}
  <div class="team-slot {c1}"><div style="{bar_style} background:{bar1};"></div>{s1_tag}{logo1}<span class="bk-name">{n1}</span>{sc1}</div>
  <div class="team-slot {c2}"><div style="{bar_style} background:{bar2};"></div>{s2_tag}{logo2}<span class="bk-name">{n2}</span>{sc2}</div>
</div>"""


def _build_slot_game_states(sim_results, preds, actual_matchups, actual_winners_map):
    """Build a dict: slot -> game_state for bracket rendering.
    Uses sim_results (actual-winners simulation) for correct team matchups,
    and preds to determine what the model would have picked."""
    states = {}
    for slot, r in sim_results.items():
        t1, t2 = r.get("t1"), r.get("t2")
        if t1 is None or t2 is None:
            continue
        matchup_key = frozenset({t1, t2})
        actual_winner = actual_winners_map.get(matchup_key)
        if not actual_winner:
            continue  # game not yet played
        actual_loser = t2 if actual_winner == t1 else t1
        # Determine what the model would have predicted for this matchup
        p = get_pred(preds, t1, t2)
        model_pick = t1 if p >= 0.5 else t2
        score = actual_matchups.get((actual_winner, actual_loser), "")
        if model_pick == actual_winner:
            states[slot] = {"type": "hit", "actual_winner": actual_winner, "score": score}
        else:
            states[slot] = {"type": "miss", "actual_winner": actual_winner, "score": score}
    return states


def region_bracket_html(region, sim_results, teams, team_seed_map, slot_states=None, espn_map=None):
    slot_states = slot_states or {}
    espn_map = espn_map or {}
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
                game_state=slot_states.get(slot), espn_map=espn_map,
            )
        html += "</div>"
    html += "</div>"
    return html


def final_four_html(sim_results, teams, team_seed_map, slot_states=None, espn_map=None):
    slot_states = slot_states or {}
    espn_map = espn_map or {}
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
            game_state=slot_states.get(slot), espn_map=espn_map,
        )
    html += "</div>"

    # Championship
    html += '<div class="ff-round">'
    html += '<div class="round-title">Championship</div>'
    r = sim_results.get(ch_slot, {})
    html += matchup_html(
        r.get("t1"), r.get("t2"), r.get("t1_prob", 0.5),
        r.get("winner"), teams, team_seed_map,
        game_state=slot_states.get(ch_slot), espn_map=espn_map,
    )
    html += "</div>"

    # Champion
    champ = sim_results.get(ch_slot, {}).get("winner")
    if champ:
        s = team_seed_map.get(champ, "")
        seed_txt = f" ({s})" if s else ""
        logo = _team_logo_img(champ, espn_map, size=28)
        html += f'<div class="ff-round"><div class="champ-banner">'
        html += f'{logo}{tname(teams, champ)}{seed_txt}'
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


def page_rankings(prefix, teams, seeds_df, conferences, massey_ranks):
    gender_label = "Men's" if prefix == "M" else "Women's"
    has_massey = prefix == "M" and bool(massey_ranks)
    st.markdown(
        f'<div class="vp-page-header">'
        f'<h1>{gender_label} Power Rankings</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)
    team_conf_str, conf_vals_raw = load_conf_strength(prefix)
    # Re-key conf_vals by display name
    conf_vals = {conf_display(k): v for k, v in conf_vals_raw.items()}

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    espn_map = _build_espn_id_map(teams, prefix)

    rows = []
    for tid in stats.index:
        s = stats.loc[tid]
        m = massey_ranks.get(tid, {})
        rows.append({
            "tid": tid,
            "Team": tname(teams, tid),
            "Conf": conf_display(conferences.get(tid, "")),
            "Seed": team_seeds.get(tid, ""),
            "Record": f"{int(s.Wins)}-{int(s.Games - s.Wins)}",
            "Elo": int(elo.get(tid, 1500)),
            "Massey": int(round(m.get("MasseyRank", 999))) if m else "",
            "NetEff": round(s.NetEff, 1),
            "OffEff": round(s.OffEff, 1),
            "DefEff": round(s.DefEff, 1),
            "Margin": round(s.Margin, 1),
            "SOS": round(s.SOS, 3) if hasattr(s, "SOS") and pd.notna(s.SOS) else 0.500,
            "eFG%": round(s.EffFGPct, 1),
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
    html = '<div style="max-height:650px; overflow-y:auto; border-radius:10px; border:1px solid #333;">'
    html += '<table class="vp-table"><thead><tr>'
    html += '<th style="width:36px;">#</th><th>TEAM</th><th>CONF</th><th>SEED</th>'
    html += '<th>RECORD</th><th>TIER</th><th>ELO</th>'
    if has_massey:
        html += '<th title="Composite rank across 60+ computer ranking systems">MASSEY</th>'
    html += '<th>NET EFF</th><th>OFF EFF</th><th>DEF EFF</th>'
    html += '<th>MARGIN</th>'
    html += '<th title="Strength of Schedule: avg opponent win%">SOS</th>'
    html += '<th>eFG%</th><th>3P%</th><th>FT%</th>'
    html += '</tr></thead><tbody>'

    # Compute averages for relative coloring of OffEff/DefEff
    avg_off = sum(r["OffEff"] for r in rows) / max(len(rows), 1)
    avg_def = sum(r["DefEff"] for r in rows) / max(len(rows), 1)

    for i, r in enumerate(rows, 1):
        seed_html = f'<span class="seed-badge">{r["Seed"]}</span>' if r["Seed"] else ""
        tier_html = _tier_badge(r["Elo"], max_elo)
        net_cls = _eff_color(r["NetEff"])
        off_cls = _eff_color(r["OffEff"] - avg_off)
        def_cls = _eff_color(avg_def - r["DefEff"])  # lower DefEff = better
        margin_cls = "stat-good" if r["Margin"] > 0 else "stat-bad" if r["Margin"] < 0 else "stat-neutral"

        html += f'<tr>'
        html += f'<td class="rank-cell">{i}</td>'
        logo = _team_logo_img(r["tid"], espn_map, size=16)
        tc = _team_color(r["tid"])
        html += f'<td class="team-cell" style="border-left:3px solid {tc}; padding-left:8px;">{logo}{r["Team"]}</td>'
        conf_str = conf_vals.get(r["Conf"], 0.5)
        conf_color = "#4ade80" if conf_str >= 0.55 else "#f87171" if conf_str < 0.48 else "#888"
        html += f'<td style="color:{conf_color}; font-weight:600;" title="Conf Avg Win%: {conf_str:.1%}">{r["Conf"]}</td>'
        html += f'<td>{seed_html}</td>'
        html += f'<td>{r["Record"]}</td>'
        html += f'<td>{tier_html}</td>'
        html += f'<td style="font-weight:700;">{r["Elo"]}</td>'
        if has_massey:
            massey_val = r["Massey"]
            if massey_val != "" and massey_val <= 25:
                m_cls = "stat-good"
            elif massey_val != "" and massey_val <= 100:
                m_cls = "stat-neutral"
            elif massey_val != "":
                m_cls = "stat-bad"
            else:
                m_cls = "stat-neutral"
            html += f'<td class="{m_cls}" style="font-weight:600;">{massey_val if massey_val != "" else "—"}</td>'
        html += f'<td class="{net_cls}" style="font-weight:600;">{r["NetEff"]:+.1f}</td>'
        html += f'<td class="{off_cls}">{r["OffEff"]:.1f}</td>'
        html += f'<td class="{def_cls}">{r["DefEff"]:.1f}</td>'
        html += f'<td class="{margin_cls}" style="font-weight:600;">{r["Margin"]:+.1f}</td>'
        sos_val = r["SOS"]
        sos_color = "#4ade80" if sos_val >= 0.530 else "#f87171" if sos_val < 0.480 else "#888"
        html += f'<td style="color:{sos_color}; font-weight:600;">{sos_val:.3f}</td>'
        html += f'<td>{r["eFG%"]:.1f}</td>'
        html += f'<td>{r["3P%"]:.1f}</td>'
        html += f'<td>{r["FT%"]:.1f}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Showing {len(rows)} teams")


# ──────────────────────────── Page: Head-to-Head ────────────────────────────

def page_h2h(prefix, teams, seeds_df, preds, coach_info, knn_data, h2h_history, seed_history):
    st.markdown(
        '<div class="vp-page-header">'
        '<h1>Head-to-Head Matchup</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)
    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    espn_map = _build_espn_id_map(teams, prefix)
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
    s1_txt = f'<span style="color:#41B6E6; font-size:13px; font-weight:700;">({s1_seed})</span> ' if s1_seed else ''
    s2_txt = f' <span style="color:#41B6E6; font-size:13px; font-weight:700;">({s2_seed})</span>' if s2_seed else ''
    color1 = "#4ade80" if p >= 0.5 else "#f87171"
    color2 = "#4ade80" if (1 - p) >= 0.5 else "#f87171"
    fav_glow = f"0 0 30px rgba({','.join(str(int(color1[i:i+2], 16)) for i in (1,3,5))}, 0.15)"

    t1_prob_winning = p >= 0.5
    t2_prob_winning = (1 - p) >= 0.5
    t1_prob_name_style = 'color:#FAFAFA; font-weight:700;' if t1_prob_winning else 'color:#888;'
    t2_prob_name_style = 'color:#FAFAFA; font-weight:700;' if t2_prob_winning else 'color:#888;'
    t1_bar_color = _team_color(t1) if t1_prob_winning else '#333'
    t2_bar_color = _team_color(t2) if t2_prob_winning else '#333'
    st.markdown(
        f'<div style="max-width:480px; margin:20px auto;">'
        f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; text-align:center; margin-bottom:4px;">WIN PROBABILITY</div>'
        f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
        f'<div style="display:flex; align-items:center; padding:10px 14px; border-bottom:1px solid #2a2a2a;">'
        f'<div style="width:4px; height:32px; border-radius:1px; background:{t1_bar_color}; margin-right:8px;"></div>'
        f'{s1_txt}{_team_logo_img(t1, espn_map, size=22)}'
        f'<span style="{t1_prob_name_style} font-size:18px; flex:1;">{n1}</span>'
        f'<span class="big-prob" style="color:{color1}; font-size:28px; font-weight:900; min-width:70px; text-align:right;">{p*100:.1f}%</span>'
        f'</div>'
        f'<div style="display:flex; align-items:center; padding:10px 14px;">'
        f'<div style="width:4px; height:32px; border-radius:1px; background:{t2_bar_color}; margin-right:8px;"></div>'
        f'{s2_txt}{_team_logo_img(t2, espn_map, size=22)}'
        f'<span style="{t2_prob_name_style} font-size:18px; flex:1;">{n2}</span>'
        f'<span class="big-prob" style="color:{color2}; font-size:28px; font-weight:900; min-width:70px; text-align:right;">{(1-p)*100:.1f}%</span>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Probability bar
    st.markdown(
        f'<div style="display:flex; height:6px; border-radius:10px; overflow:hidden; margin:8px 0 20px; '
        f'box-shadow: 0 2px 8px rgba(0,0,0,0.2);">'
        f'<div style="width:{p*100}%; background:linear-gradient(90deg, {color1}, {color1}99); '
        f'border-radius:10px 0 0 10px;"></div>'
        f'<div style="width:{(1-p)*100}%; background:linear-gradient(90deg, {color2}99, {color2}); '
        f'border-radius:0 10px 10px 0;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Projected Betting Lines ──
    st.markdown("---")
    st.subheader("Projected Betting Lines")
    st.caption("Projected lines from our ensemble model.")

    lines = compute_betting_lines(stats, preds, t1, t2)
    n1, n2 = tname(teams, t1), tname(teams, t2)
    fav = t1 if p >= 0.5 else t2
    fav_name = n1 if fav == t1 else n2
    dog_name = n2 if fav == t1 else n1
    spread_display = lines["spread"]

    # Seed tags (used in betting lines + projected score cards)
    s1_tag = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px; min-width:16px; display:inline-block; text-align:right;">{int(s1_seed)}</span>' if s1_seed else ''
    s2_tag = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px; min-width:16px; display:inline-block; text-align:right;">{int(s2_seed)}</span>' if s2_seed else ''

    # Betting lines — bracket-style stacked cards
    ml1_color = "#4ade80" if p >= 0.5 else "#f87171"
    ml2_color = "#4ade80" if p < 0.5 else "#f87171"
    if spread_display <= 0:
        s1_spread = f"{spread_display:+.1f}"
        s2_spread = f"{-spread_display:+.1f}"
    else:
        s1_spread = f"+{spread_display:.1f}"
        s2_spread = f"{-spread_display:.1f}"

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown(
            f'<div style="max-width:400px; margin:4px auto;">'
            f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; text-align:center; margin-bottom:4px;">MONEYLINE</div>'
            f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
            f'<div style="display:flex; align-items:center; padding:8px 12px; border-bottom:1px solid #2a2a2a;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t1)}; margin-right:8px;"></div>'
            f'{s1_tag}{_team_logo_img(t1, espn_map, size=20)}'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">{n1}</span>'
            f'<span style="font-size:18px; font-weight:800; color:{ml1_color}; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{lines["t1_ml"]}</span>'
            f'</div>'
            f'<div style="display:flex; align-items:center; padding:8px 12px;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t2)}; margin-right:8px;"></div>'
            f'{s2_tag}{_team_logo_img(t2, espn_map, size=20)}'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">{n2}</span>'
            f'<span style="font-size:18px; font-weight:800; color:{ml2_color}; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{lines["t2_ml"]}</span>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with lc2:
        st.markdown(
            f'<div style="max-width:400px; margin:4px auto;">'
            f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; text-align:center; margin-bottom:4px;">SPREAD</div>'
            f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
            f'<div style="display:flex; align-items:center; padding:8px 12px; border-bottom:1px solid #2a2a2a;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t1)}; margin-right:8px;"></div>'
            f'{s1_tag}{_team_logo_img(t1, espn_map, size=20)}'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">{n1}</span>'
            f'<span style="font-size:18px; font-weight:800; color:#41B6E6; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{s1_spread}</span>'
            f'</div>'
            f'<div style="display:flex; align-items:center; padding:8px 12px;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t2)}; margin-right:8px;"></div>'
            f'{s2_tag}{_team_logo_img(t2, espn_map, size=20)}'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">{n2}</span>'
            f'<span style="font-size:18px; font-weight:800; color:#41B6E6; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{s2_spread}</span>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with lc3:
        st.markdown(
            f'<div style="max-width:400px; margin:4px auto;">'
            f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; text-align:center; margin-bottom:4px;">GAME TOTAL (O/U)</div>'
            f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
            f'<div style="display:flex; align-items:center; padding:8px 12px; border-bottom:1px solid #2a2a2a;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:#41B6E6; margin-right:8px;"></div>'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">Over</span>'
            f'<span style="font-size:18px; font-weight:800; color:#41B6E6; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{lines["total"]:.1f}</span>'
            f'</div>'
            f'<div style="display:flex; align-items:center; padding:8px 12px;">'
            f'<div style="width:4px; height:28px; border-radius:1px; background:#41B6E6; margin-right:8px;"></div>'
            f'<span style="font-size:14px; flex:1; color:#FAFAFA;">Under</span>'
            f'<span style="font-size:18px; font-weight:800; color:#41B6E6; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{lines["total"]:.1f}</span>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Projected score — NCAA bracket-style stacked card
    t1_proj = lines["t1_pts"]
    t2_proj = lines["t2_pts"]
    t1_winning = t1_proj > t2_proj
    t2_winning = t2_proj > t1_proj
    t1_name_style = 'color:#FAFAFA; font-weight:700;' if t1_winning else 'color:#888;'
    t2_name_style = 'color:#FAFAFA; font-weight:700;' if t2_winning else 'color:#888;'
    t1_score_style = 'color:#FAFAFA; font-weight:800;' if t1_winning else 'color:#555;'
    t2_score_style = 'color:#FAFAFA; font-weight:800;' if t2_winning else 'color:#555;'
    st.markdown(
        f'<div style="max-width:400px; margin:12px auto;">'
        f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; text-align:center; margin-bottom:4px;">PROJECTED SCORE</div>'
        f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
        f'<div style="display:flex; align-items:center; padding:8px 12px; border-bottom:1px solid #2a2a2a;">'
        f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t1)}; margin-right:8px;"></div>'
        f'{s1_tag}{_team_logo_img(t1, espn_map, size=20)}'
        f'<span style="{t1_name_style} font-size:15px; flex:1;">{n1}</span>'
        f'<span style="{t1_score_style} font-size:20px; font-variant-numeric:tabular-nums; min-width:32px; text-align:right;">{t1_proj:.0f}</span>'
        f'</div>'
        f'<div style="display:flex; align-items:center; padding:8px 12px;">'
        f'<div style="width:4px; height:28px; border-radius:1px; background:{_team_color(t2)}; margin-right:8px;"></div>'
        f'{s2_tag}{_team_logo_img(t2, espn_map, size=20)}'
        f'<span style="{t2_name_style} font-size:15px; flex:1;">{n2}</span>'
        f'<span style="{t2_score_style} font-size:20px; font-variant-numeric:tabular-nums; min-width:32px; text-align:right;">{t2_proj:.0f}</span>'
        f'</div>'
        f'</div></div>',
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
                        f'padding:5px 0; border-bottom:1px solid #333;">'
                        f'<span style="color:#aaa;">{prop_name}</span>'
                        f'<span style="font-weight:600; color:#e2e8f0;">{val}</span></div>'
                    )
                st.markdown(
                    f'<div style="background:#2a2a2a; border:1px solid #333; border-radius:8px; padding:12px;">'
                    f'<div style="font-weight:700; margin-bottom:8px; color:{_team_color(tid)};">{_team_logo_img(tid, espn_map, size=16)}{team_name}</div>'
                    f'{prop_rows}</div>',
                    unsafe_allow_html=True,
                )

    # Stat comparison
    st.markdown("---")
    st.subheader("Stat Comparison")

    if t1 in stats.index and t2 in stats.index:
        s1_stats = stats.loc[t1]
        s2_stats = stats.loc[t2]

        # Massey composite ranks (men's only)
        massey_ranks = load_massey_ranks() if prefix == "M" else {}
        m1 = massey_ranks.get(t1, {})
        m2 = massey_ranks.get(t2, {})

        # Conference strength
        h2h_conferences = load_conferences()
        h2h_conf_str, h2h_conf_vals = load_conf_strength(prefix)
        c1_conf = conf_display(h2h_conferences.get(t1, ""))
        c2_conf = conf_display(h2h_conferences.get(t2, ""))
        c1_str = h2h_conf_str.get(t1, 0.5)
        c2_str = h2h_conf_str.get(t2, 0.5)

        compare_stats = [
            ("Elo", int(elo.get(t1, 1500)), int(elo.get(t2, 1500)), True),
        ]
        if m1 and m2:
            compare_stats.append(
                ("Massey Rank", int(round(m1.get("MasseyRank", 999))),
                 int(round(m2.get("MasseyRank", 999))), False),
            )
        compare_stats += [
            ("Conference", c1_conf, c2_conf, None),
            ("Conf Strength", f"{c1_str:.1%}", f"{c2_str:.1%}", None),
            ("Record", f"{int(s1_stats.Wins)}-{int(s1_stats.Games-s1_stats.Wins)}",
             f"{int(s2_stats.Wins)}-{int(s2_stats.Games-s2_stats.Wins)}", None),
            ("SOS", round(s1_stats.SOS, 3) if hasattr(s1_stats, "SOS") and pd.notna(s1_stats.SOS) else 0.500,
             round(s2_stats.SOS, 3) if hasattr(s2_stats, "SOS") and pd.notna(s2_stats.SOS) else 0.500, True),
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
            ("Last 10", f"{int(s1_stats.L10_Wins)}-{int(s1_stats.L10_Losses)}",
             f"{int(s2_stats.L10_Wins)}-{int(s2_stats.L10_Losses)}", None),
            ("L10 Margin", round(s1_stats.L10_Margin, 1), round(s2_stats.L10_Margin, 1), True),
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
                f'<td style="text-align:center; padding:6px 14px; color:#41B6E6; font-weight:700; '
                f'font-size:11px; letter-spacing:0.5px; text-transform:uppercase;">{label}</td>'
                f'<td style="text-align:left; padding:6px 14px; font-variant-numeric:tabular-nums;" '
                f'class="{c2_cls}">{dot2}<span style="font-weight:600;">{v2}</span></td>'
                f'</tr>'
            )

        st.markdown(
            f'<div style="border-radius:10px; border:1px solid #333; overflow:hidden; max-width:560px; margin:auto;">'
            f'<table class="vp-table" style="margin:0;">'
            f'<thead><tr>'
            f'<th style="text-align:right; width:40%;">{_team_logo_img(t1, espn_map, size=18)}{tname(teams, t1)}</th>'
            f'<th style="text-align:center; width:20%; color:#41B6E6;">STAT</th>'
            f'<th style="text-align:left; width:40%;">{_team_logo_img(t2, espn_map, size=18)}{tname(teams, t2)}</th>'
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
                        f'<div style="display:flex; align-items:center; margin-bottom:8px;">'
                        f'<div style="width:3px; height:18px; border-radius:1px; background:{_team_color(tid)}; margin-right:6px;"></div>'
                        f'{_team_logo_img(tid, espn_map, size=16)}'
                        f'<span style="font-weight:700; color:{_team_color(tid)}; font-size:11px; letter-spacing:1px; '
                        f'text-transform:uppercase;">{tname(teams, tid)}</span></div>'
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
                        f'<div style="display:flex; align-items:center; margin-bottom:8px;">'
                        f'<div style="width:3px; height:18px; border-radius:1px; background:{_team_color(tid)}; margin-right:6px;"></div>'
                        f'{_team_logo_img(tid, espn_map, size=16)}'
                        f'<span style="font-weight:700; color:{_team_color(tid)}; font-size:11px; letter-spacing:1px; '
                        f'text-transform:uppercase;">{tname(teams, tid)}</span></div>'
                        f'<div style="color:#666;">Coach data not available</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Similar Opponents ──
    st.markdown("---")
    with st.expander("Similar Opponents Analysis", expanded=True):
        st.caption(
            "We find the 5 teams most statistically similar to each opponent, "
            "then check how the other team fared against those similar teams this season."
        )

        for team_a, team_b in [(t1, t2), (t2, t1)]:
            st.markdown(
                f'<div style="font-weight:700; font-size:14px; margin:12px 0 6px;">How did '
                f'<span style="display:inline-flex; align-items:center; gap:2px;">'
                f'<span style="width:3px; height:14px; border-radius:1px; background:{_team_color(team_a)}; display:inline-block;"></span>'
                f'{_team_logo_img(team_a, espn_map, size=16)}{tname(teams, team_a)}</span>'
                f' do vs teams similar to '
                f'<span style="display:inline-flex; align-items:center; gap:2px;">'
                f'<span style="width:3px; height:14px; border-radius:1px; background:{_team_color(team_b)}; display:inline-block;"></span>'
                f'{_team_logo_img(team_b, espn_map, size=16)}{tname(teams, team_b)}</span>?</div>',
                unsafe_allow_html=True,
            )

            similar = neighbors_map.get(team_b, [])
            if not similar:
                st.info(f"No KNN data available for {tname(teams, team_b)}")
                continue

            sim_rows_data = []
            total_w, total_g, total_margin = 0, 0, 0
            for neighbor_id, dist in similar:
                results = game_results.get((team_a, neighbor_id), [])
                wins = sum(1 for w, _ in results if w == 1)
                games = len(results)
                avg_margin = round(sum(m for _, m in results) / games, 1) if games > 0 else 0
                total_w += wins
                total_g += games
                total_margin += sum(m for _, m in results)
                sim_rows_data.append((neighbor_id, games, wins, avg_margin))

            for nb_id, nb_games, nb_wins, nb_margin in sim_rows_data:
                record_txt = f"{nb_wins}-{nb_games - nb_wins}" if nb_games > 0 else "—"
                margin_txt = f"{nb_margin:+.1f}" if nb_games > 0 else "—"
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">'
                    f'<div style="display:flex; align-items:center; min-width:180px;">'
                    f'<div style="width:3px; height:16px; border-radius:1px; background:{_team_color(nb_id)}; margin-right:5px; flex-shrink:0;"></div>'
                    f'{_team_logo_img(nb_id, espn_map, size=14)}'
                    f'<span style="font-size:13px; color:#FAFAFA;">{tname(teams, nb_id)}</span></div>'
                    f'<span style="font-size:12px; color:#888; min-width:40px;">{record_txt}</span>'
                    f'<span style="font-size:12px; color:#888; min-width:40px;">{margin_txt}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
    with st.expander("All-Time Head-to-Head History", expanded=True):
        low, high = min(t1, t2), max(t1, t2)
        h2h_rec = h2h_history.get((low, high))
        if h2h_rec and h2h_rec["low_wins"] + h2h_rec["high_wins"] > 0:
            t1_wins = h2h_rec["low_wins"] if t1 == low else h2h_rec["high_wins"]
            t2_wins = h2h_rec["high_wins"] if t1 == low else h2h_rec["low_wins"]
            total = t1_wins + t2_wins

            leader = t1 if t1_wins >= t2_wins else t2
            trailer = t2 if leader == t1 else t1
            lead_wins = max(t1_wins, t2_wins)
            trail_wins = min(t1_wins, t2_wins)
            st.markdown(
                f'<div style="max-width:380px; margin:8px 0;">'
                f'<div style="font-size:9px; color:#666; letter-spacing:1.5px; font-weight:700; margin-bottom:4px;">ALL-TIME SERIES ({total} GAMES)</div>'
                f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden;">'
                f'<div style="display:flex; align-items:center; padding:6px 10px; border-bottom:1px solid #2a2a2a;">'
                f'<div style="width:3px; height:20px; border-radius:1px; background:{_team_color(leader)}; margin-right:6px; flex-shrink:0;"></div>'
                f'{_team_logo_img(leader, espn_map, size=16)}'
                f'<span style="font-size:14px; font-weight:700; color:#FAFAFA; flex:1;">{tname(teams, leader)}</span>'
                f'<span style="font-size:16px; font-weight:800; color:#FAFAFA; min-width:28px; text-align:right;">{lead_wins}</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; padding:6px 10px;">'
                f'<div style="width:3px; height:20px; border-radius:1px; background:#333; margin-right:6px; flex-shrink:0;"></div>'
                f'{_team_logo_img(trailer, espn_map, size=16)}'
                f'<span style="font-size:14px; color:#888; flex:1;">{tname(teams, trailer)}</span>'
                f'<span style="font-size:16px; color:#555; min-width:28px; text-align:right;">{trail_wins}</span>'
                f'</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Show last 5 games — bracket-style mini cards
            recent = sorted(h2h_rec["games"], key=lambda g: g["season"], reverse=True)[:5]
            if recent:
                st.markdown("**Recent matchups:**")
                for g in recent:
                    g_winner = g["winner"]
                    g_loser = t2 if g_winner == t1 else t1
                    w_color = _team_color(g_winner)
                    l_color = '#333'
                    score_parts = g["score"].split("-") if "-" in g["score"] else [g["score"], ""]
                    st.markdown(
                        f'<div style="max-width:380px; margin:4px 0;">'
                        f'<div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">'
                        f'<span style="color:#666; font-size:11px; min-width:32px;">{g["season"]}</span>'
                        f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden; flex:1;">'
                        f'<div style="display:flex; align-items:center; padding:4px 8px; border-bottom:1px solid #2a2a2a;">'
                        f'<div style="width:3px; height:18px; border-radius:1px; background:{w_color}; margin-right:6px; flex-shrink:0;"></div>'
                        f'{_team_logo_img(g_winner, espn_map, size=14)}'
                        f'<span style="font-size:13px; font-weight:700; color:#FAFAFA; flex:1;">{tname(teams, g_winner)}</span>'
                        f'<span style="font-size:14px; font-weight:800; color:#FAFAFA; min-width:24px; text-align:right;">{score_parts[0].strip()}</span>'
                        f'</div>'
                        f'<div style="display:flex; align-items:center; padding:4px 8px;">'
                        f'<div style="width:3px; height:18px; border-radius:1px; background:{l_color}; margin-right:6px; flex-shrink:0;"></div>'
                        f'{_team_logo_img(g_loser, espn_map, size=14)}'
                        f'<span style="font-size:13px; color:#888; flex:1;">{tname(teams, g_loser)}</span>'
                        f'<span style="font-size:14px; color:#555; min-width:24px; text-align:right;">{score_parts[1].strip() if len(score_parts) > 1 else ""}</span>'
                        f'</div>'
                        f'</div></div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("These teams have no recorded head-to-head history.")

    # ── Seed Matchup History ──
    s1 = team_seeds.get(t1)
    s2 = team_seeds.get(t2)
    if s1 is not None and s2 is not None:
        st.markdown("---")
        with st.expander("Seed Matchup History", expanded=True):
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



# ──────────────────────────── Page: Tournament Odds ────────────────────────────

def page_odds(prefix, teams, seeds_df, slots_df, preds):
    st.markdown(
        '<div class="vp-page-header">'
        '<h1>Championship Odds</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Running 10,000 tournament simulations..."):
        probs, round_labels = monte_carlo_odds(seeds_df, slots_df, preds)

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    elo = compute_elo(prefix)
    massey_ranks = load_massey_ranks() if prefix == "M" else {}
    has_massey = bool(massey_ranks)
    espn_map = _build_espn_id_map(teams, prefix)

    # Build set of eliminated teams from ESPN final results
    espn_data = _load_cached_espn()
    odds_map_elim, name_to_tid_elim = _build_odds_name_map(teams, prefix)
    eliminated = set()
    if espn_data and espn_data.get("games"):
        for g in espn_data["games"]:
            if g.get("status") != "STATUS_FINAL":
                continue
            hs, aws = int(g.get("home_score", 0)), int(g.get("away_score", 0))
            loser_name = g["away_team"] if hs > aws else g["home_team"]
            loser_tid = _resolve_odds_team(loser_name, odds_map_elim, name_to_tid_elim)
            if loser_tid:
                eliminated.add(loser_tid)

    rows = []
    for tid, p_list in probs.items():
        if tid in eliminated:
            continue
        row_data = {
            "Team": tname(teams, tid),
            "_tid": tid,
            "Seed": team_seeds.get(tid, 99),
            "Elo": int(elo.get(tid, 1500)),
        }
        if has_massey:
            m = massey_ranks.get(tid)
            row_data["Massey"] = round(m["MasseyRank"], 1) if m else "—"
        row_data.update({
            "R32": f"{p_list[1]*100:.1f}%",
            "S16": f"{p_list[2]*100:.1f}%",
            "E8": f"{p_list[3]*100:.1f}%",
            "FF": f"{p_list[4]*100:.1f}%",
            "CG": f"{p_list[5]*100:.1f}%",
            "Champ": f"{p_list[6]*100:.1f}%",
            "_champ_pct": p_list[6],
        })
        rows.append(row_data)

    df = pd.DataFrame(rows).sort_values("_champ_pct", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "#"

    # ── Championship Futures / Betting Odds ──
    st.markdown('<div class="vp-section">CHAMPIONSHIP FUTURES</div>', unsafe_allow_html=True)

    # Build team name -> tid lookup for enrichment
    name_to_tid = {}
    for tid, p_list in probs.items():
        name_to_tid[tname(teams, tid)] = tid

    futures_rows = []
    for _, row in df.iterrows():
        cp = row["_champ_pct"]
        if cp > 0:
            ml = prob_to_moneyline(cp)
        else:
            ml = "+99999"
        tid = name_to_tid.get(row["Team"])
        fr_entry = {
            "Team": row["Team"],
            "_tid": tid,
            "Seed": row["Seed"],
            "Champ %": f"{cp*100:.1f}%",
            "Futures Odds": ml,
            "Implied $100 Payout": f"${int(round(100 / cp))}" if cp > 0 else "—",
            "_cp": cp,
        }
        # Enrich with round probabilities and ratings
        if tid:
            p_list = probs[tid]
            fr_entry["_e8"] = p_list[3]
            fr_entry["_ff"] = p_list[4]
            fr_entry["_elo"] = int(elo.get(tid, 1500))
            if has_massey and tid in massey_ranks:
                fr_entry["_massey"] = round(massey_ranks[tid]["MasseyRank"], 1)
            else:
                fr_entry["_massey"] = None
        futures_rows.append(fr_entry)

    futures_df = pd.DataFrame(futures_rows).sort_values("_cp", ascending=False).reset_index(drop=True)
    futures_df.index = futures_df.index + 1
    futures_df.index.name = "#"

    # Top 20 futures board styled like a sportsbook
    top_futures = futures_df.head(20)
    massey_hdr = '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:48px; text-align:right;">MASSEY</span>' if has_massey else ''
    board_html = '<div style="max-width:900px; margin:auto; border-radius:10px; border:1px solid #333; overflow:hidden;">'
    # Header row
    board_html += (
        '<div style="display:flex; justify-content:space-between; align-items:center; '
        'padding:10px 16px; background:#18191f; border-bottom:2px solid #41B6E6;">'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px;">TEAM</span>'
        '<div style="display:flex; gap:16px;">'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:36px; text-align:right;">ELO</span>'
        f'{massey_hdr}'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:40px; text-align:right;">E8</span>'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:40px; text-align:right;">FF</span>'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:50px; text-align:right;">CHAMP</span>'
        '<span style="color:#888; font-size:10px; font-weight:700; letter-spacing:1px; min-width:80px; text-align:right;">ODDS</span>'
        '</div></div>'
    )
    for idx, (_, fr) in enumerate(top_futures.iterrows()):
        rank = idx + 1
        seed_txt = f'<span class="seed-badge" style="margin-right:8px;">{int(fr["Seed"])}</span>'
        odds_color = "#41B6E6" if fr["_cp"] >= 0.10 else "#4ade80" if fr["_cp"] >= 0.03 else "#e2e8f0"
        bg = 'rgba(65,182,230,0.04)' if rank <= 4 else '#18191f'
        elo_val = fr.get("_elo", 1500)
        elo_color = "#4ade80" if elo_val >= 1650 else "#e2e8f0" if elo_val >= 1550 else "#888"
        e8_pct = fr.get("_e8", 0) * 100
        ff_pct = fr.get("_ff", 0) * 100
        massey_cell = ""
        if has_massey:
            m_val = fr.get("_massey")
            if m_val is not None:
                m_color = "#4ade80" if m_val <= 25 else "#e2e8f0" if m_val <= 75 else "#f87171"
                massey_cell = f'<span style="color:{m_color}; font-size:13px; font-variant-numeric:tabular-nums; min-width:48px; text-align:right;">{m_val:.0f}</span>'
            else:
                massey_cell = '<span style="color:#444; font-size:13px; min-width:48px; text-align:right;">—</span>'
        board_html += (
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:10px 16px; border-bottom:1px solid #2a2a2a; background:{bg}; '
            f'transition:background 0.15s;" onmouseover="this.style.background=\'rgba(65,182,230,0.08)\'" '
            f'onmouseout="this.style.background=\'{bg}\'">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="color:{_team_color(fr.get("_tid"))}; font-weight:800; font-size:13px; min-width:20px;">{rank}</span>'
            f'{seed_txt}'
            f'{_team_logo_img(fr.get("_tid"), espn_map, size=18)}'
            f'<span style="font-weight:600;">{fr["Team"]}</span></div>'
            f'<div style="display:flex; gap:16px; align-items:center;">'
            f'<span style="color:{elo_color}; font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; min-width:36px; text-align:right;">{elo_val}</span>'
            f'{massey_cell}'
            f'<span style="color:#aaa; font-size:13px; font-variant-numeric:tabular-nums; min-width:40px; text-align:right;">{e8_pct:.0f}%</span>'
            f'<span style="color:#aaa; font-size:13px; font-variant-numeric:tabular-nums; min-width:40px; text-align:right;">{ff_pct:.0f}%</span>'
            f'<span style="color:#aaa; font-size:13px; font-variant-numeric:tabular-nums; min-width:50px; text-align:right;">{fr["Champ %"]}</span>'
            f'<span style="font-weight:700; font-size:16px; color:{odds_color}; min-width:80px; text-align:right; '
            f'font-variant-numeric:tabular-nums;">'
            f'{fr["Futures Odds"]}</span>'
            f'</div></div>'
        )
    board_html += '</div>'
    st.markdown(board_html, unsafe_allow_html=True)

    st.markdown("")

    # Full table
    st.subheader("Full Round-by-Round Advancement Odds")
    _styled_df(df.drop(columns=["_champ_pct"]), max_height=600, espn_map=espn_map)


# ──────────────────────────── Page: Bracket ────────────────────────────

def page_bracket(prefix, teams, seeds_df, slots_df, preds):
    st.markdown(
        '<div class="vp-page-header">'
        '<h1>Predicted Bracket</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

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

    # ── Build actual results from ESPN BEFORE simulating bracket ──
    espn_data = _load_cached_espn()
    odds_map_bk, name_to_tid_bk = _build_odds_name_map(teams, prefix)
    actual_winners_set = set()
    actual_losers = set()
    actual_matchups = {}      # (winner_tid, loser_tid) -> score string
    actual_winners_map = {}   # frozenset({tid1,tid2}) -> winner_tid

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
                actual_winners_set.add(w_tid)
            if l_tid:
                actual_losers.add(l_tid)
            if w_tid and l_tid:
                actual_matchups[(w_tid, l_tid)] = f"{w_score}-{l_score}"
                actual_winners_map[frozenset({w_tid, l_tid})] = w_tid

    # Infer play-in results: if a play-in team won an R1 game, they must have won their play-in
    seed_to_team = dict(zip(seeds_df.Seed, seeds_df.TeamID))
    playin_slots = [s for s in seeds_df.Seed.unique() if len(s) > 3 and s[-1] in ("a", "b")]
    playin_pairs = {}  # base (e.g. "Z11") -> {"a": tid, "b": tid}
    for s in playin_slots:
        base = s[:-1]
        suffix = s[-1]
        playin_pairs.setdefault(base, {})[suffix] = seed_to_team.get(s)
    for base, pair in playin_pairs.items():
        a_tid, b_tid = pair.get("a"), pair.get("b")
        if a_tid and b_tid:
            key = frozenset({a_tid, b_tid})
            if key not in actual_winners_map:
                # If one of them won an R1 game, they won the play-in
                if a_tid in actual_winners_set:
                    actual_winners_map[key] = a_tid
                    actual_losers.add(b_tid)
                elif b_tid in actual_winners_set:
                    actual_winners_map[key] = b_tid
                    actual_losers.add(a_tid)

    # Run model-only simulation first to detect misses (no actual results)
    model_results, _ = simulate_bracket(seeds_df, slots_df, preds, deterministic=True)

    # Find bracket misses: compare model predictions vs actual results
    bracket_misses = []
    bracket_hits = 0
    for slot, r in model_results.items():
        t1, t2, winner = r.get("t1"), r.get("t2"), r.get("winner")
        if t1 is None or t2 is None:
            continue
        loser = t2 if winner == t1 else t1
        if (winner, loser) in actual_matchups:
            bracket_hits += 1
        elif (loser, winner) in actual_matchups:
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

    # Now simulate with actual results for display (correct teams advance)
    sim_results, slot_winners = simulate_bracket(
        seeds_df, slots_df, preds, deterministic=True, actual_winners=actual_winners_map
    )

    total_graded = bracket_hits + len(bracket_misses)

    # Build slot-level game states for inline bracket rendering
    slot_states = _build_slot_game_states(sim_results, preds, actual_matchups, actual_winners_map)
    espn_map = _build_espn_id_map(teams, prefix)

    # ── Top bar: Record + Champion side by side ──
    champ_result = sim_results.get("R6CH", {})
    champ = champ_result.get("winner")

    if total_graded > 0 or champ:
        col_rec, col_champ = st.columns([1, 1])
        if total_graded > 0:
            with col_rec:
                rec_color = "#4ade80" if bracket_hits > len(bracket_misses) else "#f87171"
                pct = bracket_hits / total_graded * 100
                miss_note = ""
                if bracket_misses:
                    miss_items = ""
                    for m in bracket_misses:
                        miss_items += (
                            f'<span style="display:inline-flex; align-items:center; gap:2px; margin:2px 6px 2px 0;">'
                            f'<span style="width:2px; height:12px; border-radius:1px; background:{_team_color(m["actual_tid"])}; display:inline-block;"></span>'
                            f'{_team_logo_img(m["actual_tid"], espn_map, size=12)}'
                            f'{m["actual"]}</span>'
                        )
                    miss_note = (
                        f'<div style="font-size:11px; color:#888; margin-top:8px;">'
                        f'<span style="color:#f87171;">Misses:</span> {miss_items}</div>'
                    )
                st.markdown(
                    f'<div class="vp-card" style="border-top:3px solid {rec_color}; text-align:center; padding:20px;">'
                    f'<div style="font-size:9px; color:#666; letter-spacing:2px; font-weight:800; margin-bottom:6px;">'
                    f'BRACKET RECORD</div>'
                    f'<div style="font-size:36px; font-weight:900; color:{rec_color}; letter-spacing:-2px; line-height:1;">'
                    f'{bracket_hits}-{len(bracket_misses)}</div>'
                    f'<div style="font-size:12px; color:#888; margin-top:4px;">{pct:.0f}% accuracy</div>'
                    f'{miss_note}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if champ:
            with col_champ:
                s = team_seed_map.get(champ, "")
                seed_txt = f" ({s})" if s else ""
                champ_elim = champ in actual_losers
                if champ_elim:
                    border_color = "#f87171"
                    elim_txt = '<div style="color:#f87171; font-size:11px; font-weight:700; margin-top:4px; letter-spacing:1px;">ELIMINATED</div>'
                else:
                    border_color = "#41B6E6"
                    elim_txt = ""
                st.markdown(
                    f'<div class="vp-card" style="border-top:3px solid {border_color}; padding:20px;">'
                    f'<div style="font-size:9px; color:#666; letter-spacing:2px; font-weight:800; margin-bottom:10px; text-align:center;">'
                    f'PREDICTED CHAMPION</div>'
                    f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden; max-width:300px; margin:0 auto;">'
                    f'<div style="display:flex; align-items:center; padding:10px 14px;">'
                    f'<div style="width:4px; height:32px; border-radius:1px; background:{_team_color(champ)}; margin-right:8px;"></div>'
                    f'<span style="color:#888; font-weight:700; font-size:13px; margin-right:6px;">{s}</span>'
                    f'{_team_logo_img(champ, espn_map, size=24)}'
                    f'<span style="font-size:20px; font-weight:900; color:{border_color}; flex:1;">{tname(teams, champ)}</span>'
                    f'</div></div>'
                    f'{elim_txt}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Final Four
    st.markdown("")
    st.subheader("Final Four")
    st.markdown(final_four_html(sim_results, teams, team_seed_map, slot_states=slot_states, espn_map=espn_map), unsafe_allow_html=True)

    st.markdown("---")

    # ── Legend ──
    if total_graded > 0:
        st.markdown(
            '<div style="display:flex; gap:16px; align-items:center; margin-bottom:8px; flex-wrap:wrap;">'
            '<span style="font-size:10px; color:#666; letter-spacing:1px; font-weight:700;">LEGEND</span>'
            '<span style="font-size:11px; color:#888;">'
            '<span style="display:inline-block; width:10px; height:10px; background:#2d3a2d; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>'
            'Correct pick</span>'
            '<span style="font-size:11px; color:#888;">'
            '<span style="display:inline-block; width:10px; height:10px; background:#f87171; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>'
            'Missed pick</span>'
            '<span style="font-size:11px; color:#888;">'
            '<span style="display:inline-block; width:10px; height:10px; background:#2a2a2a; border:1px solid #333; border-radius:2px; margin-right:4px; vertical-align:middle;"></span>'
            'Prediction (not yet played)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Regional brackets
    for region in ["W", "X", "Y", "Z"]:
        st.subheader(region_labels[region])
        html = region_bracket_html(region, sim_results, teams, team_seed_map, slot_states=slot_states, espn_map=espn_map)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("")



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
            f'<div style="font-size:14px; font-weight:700; color:#41B6E6; margin-top:6px;">'
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
    espn_map = _build_espn_id_map(teams, prefix)
    st.markdown(
        '<div class="vp-page-header">'
        '<h1>Model Backtest</h1>'
        '</div>',
        unsafe_allow_html=True,
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
    ou_valid = bt[~bt.OU_Push]
    with st.expander("Totals Performance Deep Dive"):
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
                    f'<div class="vp-metric" style="border-top:3px solid #41B6E6;">'
                    f'<div class="label">ACCURACY</div>'
                    f'<div class="value" style="color:#41B6E6;">'
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
    with st.expander("Performance by Confidence Tier"):
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
    if len(upsets) > 0:
        for _, u in upsets.iterrows():
            fav_tid = u.Fav
            winner_tid = u.Winner
            fav_seed = f'{int(u.Fav_Seed)}' if pd.notna(u.Fav_Seed) else ''
            dog_seed = f'{int(u.Dog_Seed)}' if pd.notna(u.Dog_Seed) else ''
            fav_seed_tag = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px;">{fav_seed}</span>' if fav_seed else ''
            dog_seed_tag = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px;">{dog_seed}</span>' if dog_seed else ''
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0;">'
                f'<span style="color:#666; font-size:11px; min-width:32px;">{int(u.Season)}</span>'
                f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden; min-width:240px;">'
                f'<div style="display:flex; align-items:center; padding:4px 8px; border-bottom:1px solid #2a2a2a;">'
                f'<div style="width:3px; height:18px; border-radius:1px; background:{_team_color(winner_tid)}; margin-right:6px; flex-shrink:0;"></div>'
                f'{dog_seed_tag}{_team_logo_img(winner_tid, espn_map, size=14)}'
                f'<span style="font-size:13px; font-weight:700; color:#FAFAFA; flex:1;">{tname(teams, winner_tid)}</span>'
                f'<span style="font-size:14px; font-weight:800; color:#FAFAFA; min-width:24px; text-align:right;">{int(u.W_Score)}</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; padding:4px 8px;">'
                f'<div style="width:3px; height:18px; border-radius:1px; background:#333; margin-right:6px; flex-shrink:0;"></div>'
                f'{fav_seed_tag}{_team_logo_img(fav_tid, espn_map, size=14)}'
                f'<span style="font-size:13px; color:#888; flex:1;">{tname(teams, fav_tid)}</span>'
                f'<span style="font-size:14px; color:#555; min-width:24px; text-align:right;">{int(u.L_Score)}</span>'
                f'</div>'
                f'</div>'
                f'<span style="font-size:11px; color:#888;">Conf: {u.Fav_Prob*100:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Calibration ──
    st.markdown("---")
    with st.expander("Model Calibration"):
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

                loser = gt2 if winner == gt1 else gt1
                live_rows.append({
                    "_winner": winner, "_loser": loser,
                    "_w_score": w_score, "_l_score": l_score,
                    "_fav": fav, "_fav_prob": fav_prob,
                    "_spread": spread, "_proj_total": proj_total,
                    "_actual_total": actual_total,
                    "_ml_hit": ml_hit, "_ats_hit": ats_hit, "_is_over": is_over,
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
                    f'<div class="vp-metric" style="border-top:3px solid #41B6E6;">'
                    f'<div class="label">TOTALS (O/U)</div>'
                    f'<div class="value" style="color:#41B6E6;">{ou_o}O-{ou_u}U</div></div>',
                    unsafe_allow_html=True)

            for lr in live_rows:
                w_tid, l_tid = lr["_winner"], lr["_loser"]
                w_c = _team_color(w_tid)
                ml_badge = f'<span style="background:#4ade80; color:#000; font-weight:700; padding:2px 6px; border-radius:3px; font-size:10px;">HIT</span>' if lr["_ml_hit"] else '<span style="background:#f87171; color:#000; font-weight:700; padding:2px 6px; border-radius:3px; font-size:10px;">MISS</span>'
                ats_badge = f'<span style="background:#4ade80; color:#000; font-weight:700; padding:2px 6px; border-radius:3px; font-size:10px;">HIT</span>' if lr["_ats_hit"] else '<span style="background:#f87171; color:#000; font-weight:700; padding:2px 6px; border-radius:3px; font-size:10px;">MISS</span>'
                ou_label = "OVER" if lr["_is_over"] else "UNDER"
                ou_badge = f'<span style="background:#41B6E6; color:#000; font-weight:700; padding:2px 6px; border-radius:3px; font-size:10px;">{ou_label}</span>'
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0;">'
                    f'<div style="background:#18191f; border:1px solid #333; border-radius:4px; overflow:hidden; min-width:260px;">'
                    f'<div style="display:flex; align-items:center; padding:4px 8px; border-bottom:1px solid #2a2a2a;">'
                    f'<div style="width:3px; height:18px; border-radius:1px; background:{w_c}; margin-right:6px; flex-shrink:0;"></div>'
                    f'{_team_logo_img(w_tid, espn_map, size=14)}'
                    f'<span style="font-size:13px; font-weight:700; color:#FAFAFA; flex:1;">{tname(teams, w_tid)}</span>'
                    f'<span style="font-size:14px; font-weight:800; color:#FAFAFA; min-width:24px; text-align:right;">{lr["_w_score"]}</span>'
                    f'</div>'
                    f'<div style="display:flex; align-items:center; padding:4px 8px;">'
                    f'<div style="width:3px; height:18px; border-radius:1px; background:#333; margin-right:6px; flex-shrink:0;"></div>'
                    f'{_team_logo_img(l_tid, espn_map, size=14)}'
                    f'<span style="font-size:13px; color:#888; flex:1;">{tname(teams, l_tid)}</span>'
                    f'<span style="font-size:14px; color:#555; min-width:24px; text-align:right;">{lr["_l_score"]}</span>'
                    f'</div>'
                    f'</div>'
                    f'<div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">'
                    f'<span style="font-size:10px; color:#888;">ML</span>{ml_badge}'
                    f'<span style="font-size:10px; color:#888;">ATS</span>{ats_badge}'
                    f'<span style="font-size:10px; color:#888;">O/U</span>{ou_badge}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


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
        return "MODERATE", "#41B6E6"
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
        "ohio state buckeyes": "Ohio St",
        "ohio st buckeyes": "Ohio St",
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
    # At each step, also try state->st / saint->st / st.->st normalization
    words = ln.split()
    for i in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:i])
        # Try state/saint/st. normalization first (more specific match)
        prefix_st = prefix.replace("state", "st").replace("saint", "st")
        if prefix_st != prefix and prefix_st in name_to_tid:
            return name_to_tid[prefix_st]
        prefix_nodot = prefix.replace("st.", "st")
        if prefix_nodot != prefix and prefix_nodot in name_to_tid:
            return name_to_tid[prefix_nodot]
        # Then try exact match
        if prefix in name_to_tid:
            return name_to_tid[prefix]

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
    st.markdown(
        '<div class="vp-page-header">'
        '<h1>Betting Picks</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ──
    espn_data = _load_cached_espn()
    odds_data = _load_cached_odds()
    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)
    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))
    espn_map = _build_espn_id_map(teams, prefix)

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

        def _seed_tag(tid):
            """Return a small seed number prefix for NCAA bracket-style display."""
            s = team_seeds.get(tid)
            if s:
                return f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px; min-width:16px; display:inline-block; text-align:right;">{int(s)}</span>'
            return ""

        if live_games:
            st.markdown('<div class="vp-section" style="margin-top:8px;">LIVE NOW</div>', unsafe_allow_html=True)
            live_grid = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
            for g in live_games:
                home_tid = _resolve_odds_team(g["home_team"], odds_map_lookup, name_to_tid_lookup)
                away_tid = _resolve_odds_team(g["away_team"], odds_map_lookup, name_to_tid_lookup)
                away_seed = _seed_tag(away_tid) if away_tid else ""
                home_seed = _seed_tag(home_tid) if home_tid else ""
                pick_icon = ""
                if home_tid and away_tid and home_tid in stats.index and away_tid in stats.index:
                    t1g, t2g = min(home_tid, away_tid), max(home_tid, away_tid)
                    pg = get_pred(preds, t1g, t2g)
                    lines_g = compute_betting_lines(stats, preds, t1g, t2g)
                    fav_g = t1g if pg >= 0.5 else t2g
                    fav_prob_g = pg if fav_g == t1g else 1 - pg
                    fav_name_g = tname(teams, fav_g)
                    pick_icon = (
                        f'<div style="padding:3px 10px; border-top:1px solid #2a2a2a; font-size:10px; color:#41B6E6; font-weight:600; display:flex; align-items:center; gap:3px;">'
                        f'<span style="width:2px; height:12px; border-radius:1px; background:{_team_color(fav_g)};"></span>'
                        f'{_team_logo_img(fav_g, espn_map, size=12)}'
                        f'Pick: {fav_name_g} ({fav_prob_g*100:.0f}%) | Spread: {lines_g["spread"]:+.1f} | Total: {lines_g["total"]:.0f}'
                        f'</div>'
                    )

                broadcast = g.get("broadcast", "")
                status_html = (
                    f'<div style="display:flex; justify-content:space-between; align-items:center; padding:2px 10px; border-bottom:1px solid #2a2a2a; background:#1a1c22;">'
                    f'<span class="vp-badge vp-badge-live" style="font-size:9px;">{g["status_detail"]}</span>'
                    f'<span style="font-size:9px; font-weight:700; color:#555;">{broadcast}</span>'
                    f'</div>'
                )

                away_color = _team_color(away_tid) if away_tid else '#666'
                home_color = _team_color(home_tid) if home_tid else '#666'
                live_grid += (
                    f'<div style="background:#18191f; border:1px solid #41B6E6; border-radius:4px; overflow:hidden;">'
                    f'{status_html}'
                    f'<div style="display:flex; align-items:center; padding:6px 10px; border-bottom:1px solid #2a2a2a;">'
                    f'<div style="width:4px; height:24px; border-radius:1px; background:{away_color}; margin-right:6px; flex-shrink:0;"></div>'
                    f'{away_seed}{_team_logo_img(away_tid, espn_map, size=18)}'
                    f'<span style="font-size:13px; flex:1; color:#ccc;">{g["away_team"]}</span>'
                    f'<span style="font-size:16px; font-weight:900; color:#FAFAFA; font-variant-numeric:tabular-nums; min-width:28px; text-align:right;">{g["away_score"]}</span>'
                    f'</div>'
                    f'<div style="display:flex; align-items:center; padding:6px 10px;">'
                    f'<div style="width:4px; height:24px; border-radius:1px; background:{home_color}; margin-right:6px; flex-shrink:0;"></div>'
                    f'{home_seed}{_team_logo_img(home_tid, espn_map, size=18)}'
                    f'<span style="font-size:13px; flex:1; color:#ccc;">{g["home_team"]}</span>'
                    f'<span style="font-size:16px; font-weight:900; color:#FAFAFA; font-variant-numeric:tabular-nums; min-width:28px; text-align:right;">{g["home_score"]}</span>'
                    f'</div>'
                    f'{pick_icon}'
                    f'</div>'
                )
            live_grid += '</div>'
            st.markdown(live_grid, unsafe_allow_html=True)

        # Final scores are shown in the Model Picks grid below (with W/L grading),
        # so we skip a separate FINAL SCORES section to avoid duplication.

    # ── Build ESPN broadcast + score lookup keyed by resolved team ID pairs ──
    espn_by_ids = {}  # keyed by frozenset({tid1, tid2}) for reliable matching
    if espn_data and espn_data.get("games"):
        for g in espn_data["games"]:
            htid = _resolve_odds_team(g.get("home_team", ""), odds_map_lookup, name_to_tid_lookup)
            atid = _resolve_odds_team(g.get("away_team", ""), odds_map_lookup, name_to_tid_lookup)
            if htid and atid:
                pair = frozenset({htid, atid})
                espn_by_ids[pair] = {
                    "broadcast": g.get("broadcast", ""),
                    "start_time": g.get("start_time", ""),
                    "status": g.get("status", ""),
                    "home_team": g.get("home_team", ""),
                    "away_team": g.get("away_team", ""),
                    "home_score": int(g.get("home_score", 0)),
                    "away_score": int(g.get("away_score", 0)),
                    "home_tid": htid,
                }

    # ── Model Picks vs Vegas ──
    st.markdown("---")
    st.subheader("Model Picks")
    st.caption(
        "Our model vs Vegas for every upcoming game, sorted by tip-off time."
    )

    # Auto-use live Vegas odds when available
    has_live_odds = odds_data and odds_data.get("games")
    odds_matched = []
    if has_live_odds:
        odds_matched = _match_odds_teams(teams, stats, odds_data, prefix)

    picks = []

    if odds_matched:
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

    # ── Render Picks ──
    if not picks:
        st.markdown(
            '<div class="vp-card" style="text-align:center; padding:40px 20px;">'
            '<div style="font-size:16px; margin-bottom:12px; color:#41B6E6; font-weight:800;">VILPOM</div>'
            '<div style="font-size:16px; font-weight:600; color:#FAFAFA; margin-bottom:8px;">'
            'No Games With Odds Right Now</div>'
            '<div style="font-size:13px; color:#888; max-width:400px; margin:0 auto;">'
            'Picks appear here once matchups and Vegas lines are available. '
            'Check back closer to game time, or explore the Bracket page for full tournament predictions.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Build game cards with all three bet types per game ──
    game_cards = []  # for game-by-game display

    for pick in picks:
        t1, t2 = pick["t1"], pick["t2"]
        n1, n2 = tname(teams, t1), tname(teams, t2)
        s1 = team_seeds.get(t1, "")
        s2 = team_seeds.get(t2, "")
        s1t = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px; min-width:16px; display:inline-block; text-align:right;">{int(s1)}</span>' if s1 else ""
        s2t = f'<span style="color:#888; font-weight:700; font-size:11px; margin-right:4px; min-width:16px; display:inline-block; text-align:right;">{int(s2)}</span>' if s2 else ""
        p = pick["model_prob_t1"]
        fav = t1 if p >= 0.5 else t2
        fav_name = n1 if fav == t1 else n2
        dog_name = n2 if fav == t1 else n1
        fav_prob = p if fav == t1 else 1 - p

        # Upset detection: model picks the higher seed number (underdog by seed)
        is_upset = False
        upset_text = ""
        fav_seed = team_seeds.get(fav, "")
        dog_seed_num = team_seeds.get(t2 if fav == t1 else t1, "")
        if fav_seed and dog_seed_num:
            try:
                fs = int(fav_seed)
                ds = int(dog_seed_num)
                if fs > ds:  # Model picks the higher seed (underdog)
                    is_upset = True
                    upset_text = f"#{fs} over #{ds}"
            except (ValueError, TypeError):
                pass
        dog_prob = 1 - fav_prob
        bt_ml_pct = get_bt_pct(fav_prob, "ml")
        bt_ats_pct = get_bt_pct(fav_prob, "ats")

        has_vegas_ml = pick["vegas_ml_t1"] is not None
        has_vegas_spread = pick["vegas_spread"] is not None
        has_vegas_total = pick["vegas_total"] is not None
        has_vegas = has_vegas_ml or has_vegas_spread or has_vegas_total
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

        # Try to find broadcast + final score from ESPN data via resolved team IDs
        game_final = False
        t1_final_score = 0
        t2_final_score = 0
        espn_match = espn_by_ids.get(frozenset({t1, t2}))
        if espn_match:
            broadcast_display = espn_match.get("broadcast", "")
            if not game_time_display and espn_match.get("start_time"):
                try:
                    ct = datetime.fromisoformat(espn_match["start_time"].replace("Z", "+00:00"))
                    et = ct - timedelta(hours=4)
                    game_time_display = et.strftime("%I:%M %p ET").lstrip("0")
                except Exception:
                    pass
            if espn_match["status"] == "STATUS_FINAL":
                game_final = True
                # Map scores to t1/t2 using resolved home_tid
                if espn_match["home_tid"] == t1:
                    t1_final_score = espn_match["home_score"]
                    t2_final_score = espn_match["away_score"]
                else:
                    t1_final_score = espn_match["away_score"]
                    t2_final_score = espn_match["home_score"]

        card = {
            "game": f"{s1t}{n1} vs {s2t}{n2}",
            "t1": t1, "t2": t2, "n1": n1, "n2": n2, "s1t": s1t, "s2t": s2t,
            "ml": None, "spread": None, "total": None,
            "best_rating": 0,
            "commence_time": commence_time,
            "game_time_display": game_time_display,
            "broadcast_display": broadcast_display,
            "game_final": game_final,
            "final_score": f"{t1_final_score}-{t2_final_score}" if game_final else "",
            "t1_final": t1_final_score,
            "t2_final": t2_final_score,
            "is_upset": is_upset,
            "upset_text": upset_text,
            # Per-team inline data for condensed display
            "t1_prob": f"{p*100:.0f}%",
            "t2_prob": f"{(1-p)*100:.0f}%",
            "t1_spread": f"{pick['model_spread']:+.1f}",
            "t2_spread": f"{-pick['model_spread']:+.1f}",
            "t1_ml_odds": pick["vegas_ml_t1"] if pick["vegas_ml_t1"] is not None else pick["model_ml_t1"],
            "t2_ml_odds": pick["vegas_ml_t2"] if pick["vegas_ml_t2"] is not None else pick["model_ml_t2"],
            "has_vegas": has_vegas_ml or has_vegas_spread,
            "v_spread": pick["vegas_spread"],
            "m_total": m_total,
            "v_total": pick.get("vegas_total"),
        }

        # ── Moneyline Pick ──
        if has_vegas_ml:
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
        if has_vegas_spread:
            v_spread = pick["vegas_spread"]
            spread_diff = v_spread - m_spread
            spread_ev = abs(spread_diff) * 4.0
            spread_kelly = min(abs(spread_diff) / 20, 0.15)
            spread_score = _edge_rating(spread_ev if abs(spread_diff) >= 1.0 else 0, spread_kelly, bt_ats_pct)
            spread_label, spread_color = _edge_label(spread_score)

            if m_spread < v_spread:
                # Model says t1 is better than Vegas → take t1 side
                pick_text = f"{n1} {v_spread:+.1f}"
            else:
                # Model says t2 is better than Vegas → take t2 side
                pick_text = f"{n2} {-v_spread:+.1f}"

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
        if has_vegas_total:
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

    # Sort games by tip-off time (next game first), then by edge as tiebreaker
    # Upcoming games first (sorted by tip-off), then finals at the bottom
    game_cards.sort(key=lambda x: (x.get("game_final", False), x["commence_time"] or "9999", -x["best_rating"]))

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
        ml_c = "#4ade80" if ml_w > ml_l else "#f87171" if ml_l > ml_w else "#41B6E6"
        ats_c = "#4ade80" if ats_w > ats_l else "#f87171" if ats_l > ats_w else "#41B6E6"
        ou_c = "#4ade80" if ou_w > ou_l else "#f87171" if ou_l > ou_w else "#41B6E6"
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

    # ── Render Game Cards (condensed sportsbook-style grid) ──
    grid_html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
    for card in game_cards:
        best = card["best_rating"]
        border_color = "#4ade80" if best >= 70 else "#41B6E6" if best >= 45 else "#60a5fa" if best >= 25 else "#444"
        is_final = card.get("game_final", False)

        ct1 = card.get("t1")
        ct2 = card.get("t2")
        ct1_color = _team_color(ct1) if ct1 else '#666'
        ct2_color = _team_color(ct2) if ct2 else '#666'

        # Status bar (time, broadcast)
        status_parts = []
        if is_final:
            status_parts.append(f'<span style="color:#4ade80; font-weight:700;">FINAL</span>')
        if card.get("game_time_display"):
            status_parts.append(f'<span style="color:#888;">{card["game_time_display"]}</span>')
        if card.get("broadcast_display"):
            status_parts.append(f'<span style="color:#555;">{card["broadcast_display"]}</span>')

        # Edge/record badge in status bar
        if is_final:
            results = [card["ml"].get("result"), card["spread"].get("result"), card["total"].get("result")]
            wins = sum(1 for r in results if r == "WIN")
            losses = sum(1 for r in results if r == "LOSS")
            pushes = sum(1 for r in results if r == "PUSH")
            record_str = f"{wins}W-{losses}L" + (f"-{pushes}P" if pushes else "")
            record_color = "#4ade80" if wins > losses else "#f87171" if losses > wins else "#41B6E6"
            edge_html = f'<span style="color:{record_color}; font-weight:700;">{record_str}</span>'
        else:
            edge_html = f'<span style="color:{border_color}; font-weight:700;">EDGE {best}</span>'

        status_bar = (
            f'<div style="display:flex; justify-content:space-between; align-items:center; padding:3px 10px; '
            f'border-bottom:1px solid #2a2a2a; background:#1a1c22; font-size:9px; font-weight:700;">'
            f'<div style="display:flex; gap:6px; align-items:center;">'
            + " &middot; ".join(status_parts) if status_parts else ''
        )
        status_bar += f'</div>{edge_html}</div>'

        upset_badge = ""
        if card.get("is_upset"):
            upset_badge = (
                f'<span style="background:#ff4444; color:#fff; font-weight:800; padding:1px 5px; '
                f'border-radius:3px; font-size:7px; letter-spacing:0.3px; margin-left:3px;">UPSET</span>'
            )

        # Format ML odds for inline pills
        t1_ml_str = card["t1_ml_odds"]
        t2_ml_str = card["t2_ml_odds"]
        if isinstance(t1_ml_str, (int, float)):
            t1_ml_str = f"{'+' if t1_ml_str > 0 else ''}{int(t1_ml_str)}"
        if isinstance(t2_ml_str, (int, float)):
            t2_ml_str = f"{'+' if t2_ml_str > 0 else ''}{int(t2_ml_str)}"

        # Spread pills (using vegas spread if available, else model)
        v_sp = card.get("v_spread")
        if v_sp is not None:
            t1_sp_str = f"{v_sp:+.1f}"
            t2_sp_str = f"{-v_sp:+.1f}"
        else:
            t1_sp_str = card["t1_spread"]
            t2_sp_str = card["t2_spread"]

        # Determine which team row highlights (final: winner bold, pre: both normal)
        if is_final:
            t1_won = card["t1_final"] > card["t2_final"]
            t2_won = card["t2_final"] > card["t1_final"]
            t1_bar = ct1_color if t1_won else '#333'
            t2_bar = ct2_color if t2_won else '#333'
            t1_name_style = 'color:#FAFAFA; font-weight:700;' if t1_won else 'color:#555;'
            t2_name_style = 'color:#FAFAFA; font-weight:700;' if t2_won else 'color:#555;'
            t1_score_html = f'<span style="font-size:15px; font-weight:900; color:{"#FAFAFA" if t1_won else "#555"}; font-variant-numeric:tabular-nums; min-width:24px; text-align:right;">{card["t1_final"]}</span>'
            t2_score_html = f'<span style="font-size:15px; font-weight:900; color:{"#FAFAFA" if t2_won else "#555"}; font-variant-numeric:tabular-nums; min-width:24px; text-align:right;">{card["t2_final"]}</span>'
        else:
            t1_bar = ct1_color
            t2_bar = ct2_color
            t1_name_style = 'color:#FAFAFA; font-weight:600;'
            t2_name_style = 'color:#FAFAFA; font-weight:600;'
            t1_score_html = ''
            t2_score_html = ''

        # Pill style helper
        pill = (
            'display:inline-block; padding:2px 6px; border-radius:3px; font-size:10px; '
            'font-weight:700; font-variant-numeric:tabular-nums; margin-left:4px; '
        )

        # ML result badges
        ml_r1 = ml_r2 = ats_r1 = ats_r2 = ''
        if is_final:
            ml_res = card["ml"].get("result", "")
            ats_res = card["spread"].get("result", "")
            # Determine which team the ML pick was on
            ml_pick_text = card["ml"]["pick"]
            ml_on_t1 = card["n1"] in ml_pick_text
            if ml_res == "WIN":
                badge = '<span style="color:#4ade80; font-weight:900; font-size:9px; margin-left:2px;">W</span>'
            elif ml_res == "LOSS":
                badge = '<span style="color:#f87171; font-weight:900; font-size:9px; margin-left:2px;">L</span>'
            else:
                badge = ''
            if ml_on_t1:
                ml_r1 = badge
            else:
                ml_r2 = badge

        grid_html += (
            f'<div style="background:#18191f; border:1px solid {border_color}; border-radius:4px; overflow:hidden;">'
            f'{status_bar}'
            # Team 1 row: [bar] [seed] [logo] [name] [prob] [spread pill] [ML pill] [score]
            f'<div style="display:flex; align-items:center; padding:5px 8px; border-bottom:1px solid #2a2a2a; gap:2px;">'
            f'<div style="width:3px; height:22px; border-radius:1px; background:{t1_bar}; margin-right:4px; flex-shrink:0;"></div>'
            f'{card["s1t"]}{_team_logo_img(ct1, espn_map, size=16)}'
            f'<span style="{t1_name_style} font-size:12px; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{card["n1"]}</span>'
            f'<span style="color:#aaa; font-size:11px; font-weight:600; min-width:28px; text-align:right;">{card["t1_prob"]}</span>'
            f'<span style="{pill}background:#2a2d35; color:#ccc;">{t1_sp_str} {t1_ml_str}{ml_r1}</span>'
            f'{t1_score_html}'
            f'</div>'
            # Team 2 row
            f'<div style="display:flex; align-items:center; padding:5px 8px; gap:2px;">'
            f'<div style="width:3px; height:22px; border-radius:1px; background:{t2_bar}; margin-right:4px; flex-shrink:0;"></div>'
            f'{card["s2t"]}{_team_logo_img(ct2, espn_map, size=16)}'
            f'<span style="{t2_name_style} font-size:12px; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{card["n2"]}{upset_badge}</span>'
            f'<span style="color:#aaa; font-size:11px; font-weight:600; min-width:28px; text-align:right;">{card["t2_prob"]}</span>'
            f'<span style="{pill}background:#2a2d35; color:#ccc;">{t2_sp_str} {t2_ml_str}{ml_r2}</span>'
            f'{t2_score_html}'
            f'</div>'
        )

        # Pick recommendation footer
        ml_data = card["ml"]
        sp_data = card["spread"]
        ou_data = card["total"]
        pick_parts = []
        for btype, data in [("ML", ml_data), ("ATS", sp_data), ("O/U", ou_data)]:
            result = data.get("result", "")
            if result == "WIN":
                r_dot = '<span style="color:#4ade80;">W</span> '
            elif result == "LOSS":
                r_dot = '<span style="color:#f87171;">L</span> '
            elif result == "PUSH":
                r_dot = '<span style="color:#41B6E6;">P</span> '
            else:
                r_dot = ''
            col = data["color"]
            pick_parts.append(
                f'<span style="color:{col}; font-weight:700;">{btype}:</span> '
                f'<span style="color:#ccc;">{data["pick"]}</span> {r_dot}'
            )

        # O/U line
        ou_str = ''
        if card.get("v_total") is not None:
            ou_str = f'O/U {card["v_total"]:.1f}'
        elif card.get("m_total"):
            ou_str = f'Proj {card["m_total"]:.1f}'

        grid_html += (
            f'<div style="padding:4px 10px; border-top:1px solid #2a2a2a; background:#131418; '
            f'font-size:10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;">'
            + " &middot; ".join(pick_parts)
            + f'</div>'
            f'</div>'
        )
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # ── Legend ──
    st.markdown("---")
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
        "<div style='text-align:center; margin:16px 0 6px;'>"
        "<span style='font-size:36px; font-weight:900; letter-spacing:-2px;'>"
        "<span style='color:#41B6E6;'>VIL</span><span style='color:#FAFAFA;'>POM</span></span>"
        "</div>"
        "<div style='width:40px; height:2px; background:#41B6E6; "
        "margin:6px auto 10px;'></div>"
        "<div style='text-align:center; color:#555; font-size:10px; letter-spacing:3px; font-weight:700; margin-bottom:16px;'>"
        "NCAA TOURNAMENT ANALYTICS</div>",
        unsafe_allow_html=True,
    )

    # ── Model Performance ──
    st.markdown("---")
    st.markdown('<div class="vp-section">MODEL PERFORMANCE</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            '<div class="vp-metric" style="border-top:3px solid #4ade80;">'
            '<div class="label">MEN\'S BRIER</div>'
            '<div class="value" style="color:#4ade80;">0.134</div>'
            '<div class="sub">vs 0.250 baseline</div></div>',
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            '<div class="vp-metric" style="border-top:3px solid #60a5fa;">'
            '<div class="label">WOMEN\'S BRIER</div>'
            '<div class="value" style="color:#60a5fa;">0.097</div>'
            '<div class="sub">vs 0.250 baseline</div></div>',
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            '<div class="vp-metric" style="border-top:3px solid #41B6E6;">'
            '<div class="label">ENSEMBLE</div>'
            '<div class="value" style="color:#41B6E6;">3</div>'
            '<div class="sub">XGBoost + LightGBM + CatBoost</div></div>',
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            '<div class="vp-metric" style="border-top:3px solid #41B6E6;">'
            '<div class="label">SIMULATIONS</div>'
            '<div class="value" style="color:#41B6E6;">10K</div>'
            '<div class="sub">Monte Carlo per bracket</div></div>',
            unsafe_allow_html=True,
        )

    st.caption("Brier score measures prediction accuracy — lower is better. "
               "A coin-flip model scores 0.250. Our model cuts that error roughly in half.")

    # ── Feature Architecture ──
    st.markdown("---")
    st.markdown('<div class="vp-section">FEATURE ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown("Each matchup is evaluated on **50+ features** across 6 categories:")

    fa1, fa2, fa3 = st.columns(3)
    with fa1:
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#41B6E6; margin-bottom:6px;">Elo Ratings</div>'
            '<div style="color:#aaa; font-size:12px;">Custom Elo system with margin-of-victory scaling, '
            'home-court adjustment, and season decay. Tracks every game back to 1985.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#4ade80; margin-bottom:6px;">Efficiency Metrics</div>'
            '<div style="color:#aaa; font-size:12px;">Time-weighted offensive/defensive efficiency, tempo, '
            'eFG%, rebounding and turnover margins. Recent games weighted more heavily.</div></div>',
            unsafe_allow_html=True,
        )
    with fa2:
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#60a5fa; margin-bottom:6px;">Massey Ordinals</div>'
            '<div style="color:#aaa; font-size:12px;">Composite rankings from 60+ computer systems '
            '(KenPom, Sagarin, BPI, etc.). Mean, median, min, max, and spread.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#41B6E6; margin-bottom:6px;">KNN Opponents</div>'
            '<div style="color:#aaa; font-size:12px;">Finds the 5 most statistically similar teams to each '
            'opponent, then analyzes head-to-head results against those proxies.</div></div>',
            unsafe_allow_html=True,
        )
    with fa3:
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#c084fc; margin-bottom:6px;">Coaching Data</div>'
            '<div style="color:#aaa; font-size:12px;">Coach tenure, all-time NCAA tournament win rate, '
            'and games coached. Experience matters in March.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vp-card">'
            '<div style="font-weight:700; color:#f87171; margin-bottom:6px;">Conference Strength</div>'
            '<div style="color:#aaa; font-size:12px;">Average and median conference win percentage as a proxy '
            'for strength of schedule. Adjusts for weak vs. strong leagues.</div></div>',
            unsafe_allow_html=True,
        )

    # ── How It Works ──
    st.markdown("---")
    st.markdown('<div class="vp-section">HOW IT WORKS</div>', unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3)
    with hw1:
        st.markdown(
            '<div class="vp-card" style="text-align:center;">'
            '<div style="font-size:28px; margin-bottom:8px;">1</div>'
            '<div style="font-weight:700; color:#41B6E6; margin-bottom:6px;">Train</div>'
            '<div style="color:#aaa; font-size:12px;">Three models independently learn from 20+ years of '
            'tournament results using Optuna-tuned hyperparameters and temporal cross-validation.</div></div>',
            unsafe_allow_html=True,
        )
    with hw2:
        st.markdown(
            '<div class="vp-card" style="text-align:center;">'
            '<div style="font-size:28px; margin-bottom:8px;">2</div>'
            '<div style="font-weight:700; color:#4ade80; margin-bottom:6px;">Predict</div>'
            '<div style="color:#aaa; font-size:12px;">Each model outputs a win probability for every possible '
            'matchup. Predictions are calibrated with Platt scaling, then averaged.</div></div>',
            unsafe_allow_html=True,
        )
    with hw3:
        st.markdown(
            '<div class="vp-card" style="text-align:center;">'
            '<div style="font-size:28px; margin-bottom:8px;">3</div>'
            '<div style="font-weight:700; color:#60a5fa; margin-bottom:6px;">Simulate</div>'
            '<div style="color:#aaa; font-size:12px;">10,000 Monte Carlo bracket simulations generate '
            'championship odds, round-by-round advancement, and futures pricing.</div></div>',
            unsafe_allow_html=True,
        )

    # ── Live Data Pipeline ──
    st.markdown("---")
    st.markdown('<div class="vp-section">LIVE DATA PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vp-card">'
        '<div style="display:flex; gap:24px; flex-wrap:wrap; align-items:center; justify-content:center;">'
        '<div style="text-align:center; min-width:140px;">'
        '<div style="color:#41B6E6; font-weight:700; font-size:13px;">ESPN API</div>'
        '<div style="color:#888; font-size:11px;">Live scores & game status</div></div>'
        '<div style="color:#444; font-size:18px;">&#8594;</div>'
        '<div style="text-align:center; min-width:140px;">'
        '<div style="color:#4ade80; font-weight:700; font-size:13px;">GitHub Actions</div>'
        '<div style="color:#888; font-size:11px;">Auto-fetch every 30 min</div></div>'
        '<div style="color:#444; font-size:18px;">&#8594;</div>'
        '<div style="text-align:center; min-width:140px;">'
        '<div style="color:#60a5fa; font-weight:700; font-size:13px;">The Odds API</div>'
        '<div style="color:#888; font-size:11px;">Vegas lines from major books</div></div>'
        '<div style="color:#444; font-size:18px;">&#8594;</div>'
        '<div style="text-align:center; min-width:140px;">'
        '<div style="color:#41B6E6; font-weight:700; font-size:13px;">Vilpom</div>'
        '<div style="color:#888; font-size:11px;">Model vs Vegas edge detection</div></div>'
        '</div></div>',
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
    "<div style='text-align:center; padding:8px 0 4px;'>"
    "<span style='font-size:32px; font-weight:900; letter-spacing:-1.5px;'>"
    "<span style='color:#41B6E6;'>VIL</span><span style='color:#FAFAFA;'>POM</span></span>"
    "</div>"
    "<div style='text-align:center; color:#555; font-size:9px; letter-spacing:3px; font-weight:700; margin-bottom:4px;'>"
    "ANALYTICS & INTELLIGENCE</div>"
    "<div style='width:40px; height:2px; background:linear-gradient(90deg, #41B6E6, #6dcbf2); "
    "margin:0 auto 4px; border-radius:2px;'></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], horizontal=True)
prefix = "M" if gender == "Men's" else "W"


gender_seeds = seeds[seeds.TeamID < 3000] if prefix == "M" else seeds[seeds.TeamID >= 3000]
gender_slots = m_slots if prefix == "M" else w_slots

page = st.sidebar.radio(
    "Navigate",
    ["Betting Picks", "Head-to-Head", "Rankings",
     "Bracket", "Championship Odds", "Backtest", "About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; padding:4px 0;'>"
    "<div style='color:#333; font-size:9px; letter-spacing:1.5px; font-weight:600;'>"
    "VILPOM &copy; 2026 &middot; v2.0</div>"
    "<div style='color:#333; font-size:8px; margin-top:4px; letter-spacing:0.5px;'>"
    "For entertainment only</div>"
    "</div>",
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
    massey_ranks = load_massey_ranks() if prefix == "M" else {}
    page_rankings(prefix, teams, gender_seeds, conferences, massey_ranks)
elif "Bracket" in page:
    page_bracket(prefix, teams, gender_seeds, gender_slots, preds)
elif "Championship Odds" in page:
    page_odds(prefix, teams, gender_seeds, gender_slots, preds)
elif "Backtest" in page:
    page_backtest(prefix, teams)
elif "About" in page:
    page_about()
