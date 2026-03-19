"""
March ML Mania 2026 — Interactive Dashboard
ESPN/KenPom-style stats, head-to-head odds, tournament odds, and bracket.
"""

import os
import math

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEASON = 2026
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(
    page_title="March ML Mania 2026",
    page_icon="\U0001f3c0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }
    .bracket { display: flex; gap: 6px; min-height: 540px; overflow-x: auto; }
    .round {
        display: flex; flex-direction: column; justify-content: space-around;
        min-width: 165px;
    }
    .round-title {
        text-align: center; font-size: 11px; color: #888;
        margin-bottom: 4px; font-weight: 600; letter-spacing: 0.5px;
    }
    .matchup {
        border: 1px solid #333; border-radius: 4px; overflow: hidden;
        margin: 2px 0; background: #1a1d24;
    }
    .team-slot {
        padding: 3px 8px; font-size: 12px; display: flex;
        justify-content: space-between; align-items: center;
        border-bottom: 1px solid #2a2d34; color: #ccc;
    }
    .team-slot:last-child { border-bottom: none; }
    .team-slot.winner {
        background: rgba(34, 197, 94, 0.13); color: #4ade80;
        font-weight: 700;
    }
    .seed-tag { color: #666; margin-right: 4px; font-size: 11px; }
    .prob-tag { color: #666; font-size: 10px; }
    .big-prob {
        font-size: 48px; font-weight: 800; text-align: center;
        line-height: 1.1; margin: 10px 0;
    }
    .stat-bar {
        height: 6px; border-radius: 3px; margin: 2px 0;
    }
    .ff-bracket { display: flex; gap: 12px; min-height: 180px; align-items: center; }
    .ff-round {
        display: flex; flex-direction: column; justify-content: space-around;
        min-width: 180px; min-height: 160px;
    }
    .champ-banner {
        text-align: center; font-size: 28px; font-weight: 800;
        padding: 16px; border: 2px solid #fbbf24; border-radius: 8px;
        background: rgba(251, 191, 36, 0.08); color: #fbbf24;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


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
def monte_carlo_odds(_seeds_df, _slots_df, _preds, n_sims=10000):
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

def page_rankings(prefix, teams, seeds_df, conferences):
    st.header(f"{'Men' if prefix == 'M' else 'Women'}'s Rankings")

    stats = compute_season_stats(prefix)
    elo = compute_elo(prefix)

    team_seeds = dict(zip(seeds_df.TeamID, seeds_df.SeedNum))

    rows = []
    for tid in stats.index:
        s = stats.loc[tid]
        rows.append({
            "Team": tname(teams, tid),
            "Conf": conferences.get(tid, ""),
            "Seed": team_seeds.get(tid, ""),
            "Record": f"{int(s.Wins)}-{int(s.Games - s.Wins)}",
            "Elo": int(elo.get(tid, 1500)),
            "NetEff": s.NetEff,
            "OffEff": s.OffEff,
            "DefEff": s.DefEff,
            "Tempo": s.Tempo,
            "PPG": round(s.PPG, 1),
            "OppPPG": round(s.OppPPG, 1),
            "Margin": round(s.Margin, 1),
            "eFG%": s.EffFGPct,
            "FG%": s.FGPct,
            "3P%": s.FG3Pct,
            "FT%": s.FTPct,
            "RPG": round(s.RPG, 1),
            "APG": round(s.APG, 1),
            "TOPG": round(s.TOPG, 1),
            "SPG": round(s.SPG, 1),
            "BPG": round(s.BPG, 1),
        })

    df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    col1, col2 = st.columns([1, 3])
    with col1:
        show_tourney_only = st.checkbox("Tournament teams only", value=False)
        conf_filter = st.selectbox("Conference", ["All"] + sorted(df["Conf"].unique().tolist()))
    with col2:
        search = st.text_input("Search team", "")

    if show_tourney_only:
        df = df[df["Seed"] != ""]
    if conf_filter != "All":
        df = df[df["Conf"] == conf_filter]
    if search:
        df = df[df["Team"].str.contains(search, case=False)]

    st.dataframe(
        df, use_container_width=True, height=600,
        column_config={
            "Elo": st.column_config.NumberColumn("Elo", format="%d"),
            "NetEff": st.column_config.NumberColumn("Net Eff", format="%.1f"),
            "OffEff": st.column_config.NumberColumn("Off Eff", format="%.1f"),
            "DefEff": st.column_config.NumberColumn("Def Eff", format="%.1f"),
        },
    )


# ──────────────────────────── Page: Head-to-Head ────────────────────────────

def page_h2h(prefix, teams, seeds_df, preds, coach_info, knn_data, h2h_history, seed_history):
    st.header("Head-to-Head Matchup")

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
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        color1 = "#4ade80" if p >= 0.5 else "#f87171"
        st.markdown(
            f'<div class="big-prob" style="color:{color1}">{p*100:.1f}%</div>'
            f'<div style="text-align:center; font-size:18px; font-weight:600;">{tname(teams, t1)}</div>',
            unsafe_allow_html=True,
        )
        s1 = team_seeds.get(t1)
        if s1:
            st.markdown(f'<div style="text-align:center; color:#888;">Seed: {s1}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div style="text-align:center; font-size:24px; color:#666; margin-top:20px;">vs</div>',
            unsafe_allow_html=True,
        )
    with c3:
        color2 = "#4ade80" if (1 - p) >= 0.5 else "#f87171"
        st.markdown(
            f'<div class="big-prob" style="color:{color2}">{(1-p)*100:.1f}%</div>'
            f'<div style="text-align:center; font-size:18px; font-weight:600;">{tname(teams, t2)}</div>',
            unsafe_allow_html=True,
        )
        s2 = team_seeds.get(t2)
        if s2:
            st.markdown(f'<div style="text-align:center; color:#888;">Seed: {s2}</div>', unsafe_allow_html=True)

    # Probability bar
    st.markdown(
        f'<div style="display:flex; height:12px; border-radius:6px; overflow:hidden; margin:16px 0;">'
        f'<div style="width:{p*100}%; background:#4ade80;"></div>'
        f'<div style="width:{(1-p)*100}%; background:#f87171;"></div>'
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
                    c1_style = "color:#4ade80; font-weight:700;" if v1 > v2 else ""
                    c2_style = "color:#4ade80; font-weight:700;" if v2 > v1 else ""
                else:
                    c1_style = "color:#4ade80; font-weight:700;" if v1 < v2 else ""
                    c2_style = "color:#4ade80; font-weight:700;" if v2 < v1 else ""
            else:
                c1_style = c2_style = ""

            rows_html += f"""<tr>
                <td style="text-align:right; padding:4px 12px; {c1_style}">{v1}</td>
                <td style="text-align:center; padding:4px 12px; color:#888; font-weight:600;">{label}</td>
                <td style="text-align:left; padding:4px 12px; {c2_style}">{v2}</td>
            </tr>"""

        st.markdown(
            f'<table style="width:100%; max-width:500px; margin:auto; border-collapse:collapse;">'
            f'<tr><th style="text-align:right; padding:6px 12px;">{tname(teams, t1)}</th>'
            f'<th style="text-align:center; padding:6px 12px; color:#888;">Stat</th>'
            f'<th style="text-align:left; padding:6px 12px;">{tname(teams, t2)}</th></tr>'
            f'{rows_html}</table>',
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
                    st.markdown(f"**{tname(teams, tid)}**")
                    st.markdown(f"**Coach:** {info['name']}")
                    st.markdown(f"**Tenure:** {info['tenure']} season{'s' if info['tenure'] != 1 else ''}")
                    if info['tourney_games'] > 0:
                        st.markdown(
                            f"**NCAA Tournament Record:** {info['tourney_wins']}-"
                            f"{info['tourney_games'] - info['tourney_wins']} "
                            f"({info['tourney_pct']}%)"
                        )
                    else:
                        st.markdown("**NCAA Tournament Record:** No prior tournament games")
                else:
                    st.markdown(f"**{tname(teams, tid)}** — Coach data not available")

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
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

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

    ec1, ec2 = st.columns(2)
    with ec1:
        st.metric("Elo Only", f"{elo_wp*100:.1f}%", help=f"Elo ratings: {int(elo1)} vs {int(elo2)}")
    with ec2:
        diff_pp = (p - elo_wp) * 100
        st.metric("Full Model", f"{p*100:.1f}%", delta=f"{diff_pp:+.1f}pp")


# ──────────────────────────── Page: Tournament Odds ────────────────────────────

def page_odds(prefix, teams, seeds_df, slots_df, preds):
    st.header("Tournament Championship Odds")

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
    st.dataframe(ff_df.drop(columns=["_ff"]).head(30), use_container_width=True, hide_index=False)

    # Top contenders bar chart
    st.markdown("---")
    top = df.head(16).copy()
    top["Champ %"] = top["_champ_pct"] * 100

    st.subheader("Top Championship Contenders")
    chart_data = top.set_index("Team")["Champ %"]
    st.bar_chart(chart_data, height=350)

    # Full table
    st.subheader("Full Round-by-Round Advancement Odds")
    st.dataframe(
        df.drop(columns=["_champ_pct"]),
        use_container_width=True, height=600,
    )


# ──────────────────────────── Page: Bracket ────────────────────────────

def page_bracket(prefix, teams, seeds_df, slots_df, preds):
    st.header("Predicted Tournament Bracket")

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

    # Champion banner
    champ_result = sim_results.get("R6CH", {})
    champ = champ_result.get("winner")
    if champ:
        s = team_seed_map.get(champ, "")
        seed_txt = f" ({s} seed)" if s else ""
        st.markdown(
            f'<div class="champ-banner">\U0001f3c6 Predicted Champion: '
            f'{tname(teams, champ)}{seed_txt}</div>',
            unsafe_allow_html=True,
        )

    # Final Four
    st.subheader("Final Four")
    st.markdown(final_four_html(sim_results, teams, team_seed_map), unsafe_allow_html=True)

    st.markdown("---")

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
            st.dataframe(pd.DataFrame(game_rows), use_container_width=True, hide_index=True)
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
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
            f'<div style="color:#888; font-size:12px;">MONEYLINE</div>'
            f'<div style="font-size:28px; font-weight:800; color:{ml_color};">{ml_wins}-{total_games - ml_wins}</div>'
            f'<div style="font-size:14px; color:#aaa;">{ml_pct:.1f}% Win Rate</div>'
            f'<div style="font-size:16px; font-weight:700; color:{ml_color}; margin-top:4px;">'
            f'{"+" if ml_total_profit >= 0 else ""}{ml_total_profit:.0f}u ({ml_roi:+.1f}% ROI)</div>'
            f'</div>', unsafe_allow_html=True)
    with kc2:
        ats_color = "#4ade80" if ats_pct > 52.4 else "#f87171"
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
            f'<div style="color:#888; font-size:12px;">AGAINST THE SPREAD</div>'
            f'<div style="font-size:28px; font-weight:800; color:{ats_color};">{ats_wins}-{len(ats_valid) - ats_wins}</div>'
            f'<div style="font-size:14px; color:#aaa;">{ats_pct:.1f}% Cover Rate</div>'
            f'<div style="font-size:16px; font-weight:700; color:{ats_color}; margin-top:4px;">'
            f'{"+" if ats_total_profit >= 0 else ""}{ats_total_profit:.0f}u ({ats_roi:+.1f}% ROI)</div>'
            f'</div>', unsafe_allow_html=True)
    with kc3:
        ou_color = "#4ade80" if abs(ou_pct - 50) < 5 else "#fbbf24"
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
            f'<div style="color:#888; font-size:12px;">TOTALS (O/U)</div>'
            f'<div style="font-size:28px; font-weight:800; color:#60a5fa;">'
            f'{ou_over_ct}O / {ou_under_ct}U</div>'
            f'<div style="font-size:14px; color:#aaa;">Over hits: {ou_pct:.1f}%</div>'
            f'<div style="font-size:16px; font-weight:700; color:#fbbf24; margin-top:4px;">'
            f'MAE: {mae_total:.1f} pts</div>'
            f'</div>', unsafe_allow_html=True)
    with kc4:
        brier = ((bt.Fav_Prob - bt.ML_Correct.astype(float)) ** 2).mean()
        st.markdown(
            f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
            f'<div style="color:#888; font-size:12px;">MODEL QUALITY</div>'
            f'<div style="font-size:28px; font-weight:800; color:#c084fc;">{brier:.3f}</div>'
            f'<div style="font-size:14px; color:#aaa;">Avg Brier Score</div>'
            f'<div style="font-size:16px; font-weight:700; color:#c084fc; margin-top:4px;">'
            f'{total_games} games tested</div>'
            f'</div>', unsafe_allow_html=True)


def _fetch_espn_scores():
    """Fetch live NCAA tournament scores from ESPN's free API."""
    import requests as rq
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        resp = rq.get(url, params={"groups": 100, "limit": 200}, timeout=10)
        if resp.status_code != 200:
            return None, f"ESPN API returned {resp.status_code}"
        data = resp.json()
        games = []
        for ev in data.get("events", []):
            status = ev.get("status", {}).get("type", {}).get("name", "")
            comps = ev.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = competitors[0]
            away = competitors[1]
            games.append({
                "game_id": ev.get("id"),
                "name": ev.get("name", ""),
                "status": status,
                "status_detail": ev.get("status", {}).get("type", {}).get("shortDetail", ""),
                "home_team": home.get("team", {}).get("displayName", ""),
                "home_score": int(home.get("score", 0)),
                "home_id": home.get("team", {}).get("id", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
                "away_score": int(away.get("score", 0)),
                "away_id": away.get("team", {}).get("id", ""),
            })
        return games, None
    except Exception as e:
        return None, str(e)


def _get_odds_key():
    """Get Odds API key from session state, env var, or hardcoded fallback."""
    if "odds_api_key" in st.session_state and st.session_state["odds_api_key"]:
        return st.session_state["odds_api_key"]
    return os.environ.get("ODDS_API_KEY", "")


def _fetch_odds():
    """Fetch NCAAB odds from The Odds API (free tier, 500 req/month).
    Caches in session state to avoid burning credits on re-renders."""
    import requests as rq
    api_key = _get_odds_key()
    if not api_key:
        return None, "No API key set. Add one in Settings (sidebar)."
    try:
        url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
        resp = rq.get(url, params={
            "apiKey": api_key, "regions": "us",
            "markets": "h2h,spreads,totals", "oddsFormat": "american",
        }, timeout=10)
        if resp.status_code == 401:
            return None, "Invalid API key. Check your key in Settings."
        if resp.status_code == 429:
            return None, "Monthly credit limit reached. Resets next month."
        if resp.status_code != 200:
            return None, f"Odds API returned {resp.status_code}"
        remaining = resp.headers.get("x-requests-remaining", "?")
        data = resp.json()
        # Cache in session state so re-renders don't burn credits
        result = {"games": data, "remaining": remaining}
        st.session_state["cached_odds"] = result
        return result, None
    except Exception as e:
        return None, str(e)


def page_backtest(prefix, teams):
    st.header("Model Backtest")
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
                f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
                f'<div style="color:#888; font-size:12px;">PROJECTION BIAS</div>'
                f'<div style="font-size:28px; font-weight:800; color:{bias_color};">{bias:+.1f}</div>'
                f'<div style="font-size:13px; color:#aaa;">Avg Proj: {avg_proj:.1f} | Avg Actual: {avg_actual:.1f}</div>'
                f'</div>', unsafe_allow_html=True)
        with tc2:
            rmse = (bt.Total_Diff ** 2).mean() ** 0.5
            st.markdown(
                f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
                f'<div style="color:#888; font-size:12px;">RMSE</div>'
                f'<div style="font-size:28px; font-weight:800; color:#60a5fa;">{rmse:.1f}</div>'
                f'<div style="font-size:13px; color:#aaa;">Root Mean Squared Error</div>'
                f'</div>', unsafe_allow_html=True)
        with tc3:
            within_5 = (bt.Total_Diff.abs() <= 5).sum()
            within_10 = (bt.Total_Diff.abs() <= 10).sum()
            st.markdown(
                f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; padding:16px; text-align:center;">'
                f'<div style="color:#888; font-size:12px;">ACCURACY</div>'
                f'<div style="font-size:28px; font-weight:800; color:#fbbf24;">'
                f'{within_5/total_games*100:.0f}%</div>'
                f'<div style="font-size:13px; color:#aaa;">Within 5 pts: {within_5} | '
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
            st.dataframe(pd.DataFrame(total_tier_rows), use_container_width=True, hide_index=True)

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

    st.dataframe(pd.DataFrame(season_rows), use_container_width=True, hide_index=True)

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
    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(upset_rows), use_container_width=True, hide_index=True)

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
            st.dataframe(cal_df, use_container_width=True, hide_index=True)

    # ── Live Scores & Odds ──
    st.markdown("---")
    st.subheader("Live Scores & Odds")

    live_col1, live_col2 = st.columns(2)
    with live_col1:
        fetch_scores = st.button("Refresh Live Scores")
    with live_col2:
        fetch_odds = st.button("Refresh Vegas Odds")

    if fetch_scores:
        with st.spinner("Loading scores..."):
            espn_games, err = _fetch_espn_scores()
        if err:
            st.error(f"Could not load scores: {err}")
        elif espn_games:
            final_games = [g for g in espn_games if g["status"] == "STATUS_FINAL"]
            live_games = [g for g in espn_games if g["status"] == "STATUS_IN_PROGRESS"]
            scheduled = [g for g in espn_games if g["status"] == "STATUS_SCHEDULED"]

            if live_games:
                st.markdown("**In Progress**")
                for g in live_games:
                    st.markdown(
                        f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; '
                        f'padding:10px 16px; margin:4px 0; display:flex; justify-content:space-between;">'
                        f'<span>{g["away_team"]} {g["away_score"]} @ {g["home_team"]} {g["home_score"]}</span>'
                        f'<span style="color:#fbbf24;">{g["status_detail"]}</span>'
                        f'</div>', unsafe_allow_html=True)

            if final_games:
                st.markdown("**Final Scores**")
                for g in final_games:
                    winner = g["home_team"] if g["home_score"] > g["away_score"] else g["away_team"]
                    st.markdown(
                        f'<div style="background:#1a1d24; border:1px solid #333; border-radius:8px; '
                        f'padding:10px 16px; margin:4px 0; display:flex; justify-content:space-between;">'
                        f'<span>{g["away_team"]} {g["away_score"]} @ {g["home_team"]} {g["home_score"]}</span>'
                        f'<span style="color:#4ade80;">FINAL</span>'
                        f'</div>', unsafe_allow_html=True)

            if scheduled:
                st.markdown(f"**Upcoming:** {len(scheduled)} games scheduled")

            if not espn_games:
                st.info("No NCAA tournament games found on today's schedule.")
        else:
            st.info("No games returned from ESPN.")

    if fetch_odds:
        with st.spinner("Loading odds..."):
            odds_data, err = _fetch_odds()
        if err:
            st.warning(f"Could not load odds: {err}")
        elif odds_data:
            st.success(f"Loaded {len(odds_data['games'])} games. "
                       f"API requests remaining: {odds_data['remaining']}")
            for game in odds_data["games"][:20]:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                bookmakers = game.get("bookmakers", [])
                if not bookmakers:
                    continue
                bk = bookmakers[0]  # first bookmaker
                markets = {m["key"]: m for m in bk.get("markets", [])}

                line_parts = [f"**{away} @ {home}** ({bk['title']})"]
                if "h2h" in markets:
                    outcomes = {o["name"]: o["price"] for o in markets["h2h"]["outcomes"]}
                    ml_str = " | ".join(f"{k}: {'+' if v > 0 else ''}{v}" for k, v in outcomes.items())
                    line_parts.append(f"ML: {ml_str}")
                if "spreads" in markets:
                    outcomes = markets["spreads"]["outcomes"]
                    sp_str = " | ".join(f"{o['name']}: {o.get('point', '')} ({'+' if o['price'] > 0 else ''}{o['price']})"
                                        for o in outcomes)
                    line_parts.append(f"Spread: {sp_str}")
                if "totals" in markets:
                    outcomes = markets["totals"]["outcomes"]
                    tot_str = " | ".join(f"{o['name']}: {o.get('point', '')} ({'+' if o['price'] > 0 else ''}{o['price']})"
                                         for o in outcomes)
                    line_parts.append(f"Total: {tot_str}")

                st.markdown(" | ".join(line_parts[:1]) + "\n" + " | ".join(line_parts[1:]))

    # ── Live Results Tracker ──
    live_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_results.json")
    if os.path.exists(live_path):
        import json
        with open(live_path) as f:
            live = json.load(f)

        if live.get("games"):
            st.markdown("---")
            st.subheader("2026 Tournament Results Tracker")
            st.success(f"Tracking {len(live['games'])} completed games")
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
                st.markdown(f'<div style="text-align:center;font-size:20px;font-weight:700;color:{c};">ML: {ml_w}-{ml_l}</div>',
                            unsafe_allow_html=True)
            with lc2:
                c = "#4ade80" if ats_w > ats_l else "#f87171"
                st.markdown(f'<div style="text-align:center;font-size:20px;font-weight:700;color:{c};">ATS: {ats_w}-{ats_l}</div>',
                            unsafe_allow_html=True)
            with lc3:
                st.markdown(f'<div style="text-align:center;font-size:20px;font-weight:700;color:#fbbf24;">O/U: {ou_o}O-{ou_u}U</div>',
                            unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)


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


def page_picks(prefix, teams, seeds_df, preds):
    st.header("Betting Picks")
    st.caption(
        "Find edges where our model disagrees with Vegas. "
        "Bets are ranked by a composite score of expected value, Kelly sizing, and historical accuracy."
    )

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
        """Get historical backtest accuracy for this confidence level."""
        pct_val = prob * 100
        for (lo, hi), data in bt_tiers.items():
            if lo <= pct_val < hi:
                return data[f"{bet_type}_pct"]
        return 50.0

    # ── Data Source Selection ──
    st.markdown("---")
    data_source = st.radio(
        "Where are the odds coming from?",
        ["Tournament Bracket", "Enter Odds Manually", "Live Odds (API)"],
        horizontal=True,
    )

    picks = []  # list of dicts with all pick data

    if data_source == "Tournament Bracket":
        # Generate picks for all R1 tournament matchups using model lines as "Vegas" baseline
        m_slots_df, w_slots_df = load_slots()
        slots_df = m_slots_df if prefix == "M" else w_slots_df
        sim_results, _ = simulate_bracket(seeds_df, slots_df, preds, deterministic=True)

        st.info("Showing model projections for all Round 1 games. "
                "Switch to 'Enter Odds Manually' or 'Live Odds' to compare against real Vegas lines.")

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
                    # Use model lines as default "vegas" — user can override
                    "vegas_ml_t1": None, "vegas_ml_t2": None,
                    "vegas_spread": None, "vegas_total": None,
                    "region": region,
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
            })

    elif data_source == "Live Odds (API)":
        bc1, bc2 = st.columns([1, 2])
        with bc1:
            fetch_btn = st.button("Fetch Live Odds")
        with bc2:
            if "cached_odds" in st.session_state:
                st.caption(f"Using cached odds ({len(st.session_state.get('odds_data', []))} games) "
                           f"| Credits remaining: {st.session_state['cached_odds'].get('remaining', '?')}/500")

        if fetch_btn:
            with st.spinner("Loading live odds..."):
                odds_data, err = _fetch_odds()
            if err:
                st.error(err)
            elif odds_data and odds_data["games"]:
                st.success(f"Loaded odds for {len(odds_data['games'])} games")
                st.session_state["odds_data"] = odds_data["games"]
            else:
                st.warning("No games with odds available right now.")

        if "odds_data" in st.session_state:
            # Try to match odds games to our teams
            # Build a fuzzy name map
            all_team_names = {tname(teams, tid).lower(): tid for tid in teams}

            for game in st.session_state["odds_data"]:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                bookmakers = game.get("bookmakers", [])
                if not bookmakers:
                    continue

                # Try to find team IDs by name matching
                home_tid = all_team_names.get(home.lower())
                away_tid = all_team_names.get(away.lower())

                # Try partial matching if exact fails
                if not home_tid:
                    for name, tid in all_team_names.items():
                        if home.lower() in name or name in home.lower():
                            home_tid = tid
                            break
                if not away_tid:
                    for name, tid in all_team_names.items():
                        if away.lower() in name or name in away.lower():
                            away_tid = tid
                            break

                if not home_tid or not away_tid or home_tid not in stats.index or away_tid not in stats.index:
                    continue

                # Get consensus odds (average across bookmakers or use first)
                bk = bookmakers[0]
                markets = {m["key"]: m for m in bk.get("markets", [])}

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
                    "vegas_ml_t1": v_ml1, "vegas_ml_t2": v_ml2,
                    "vegas_spread": v_spread, "vegas_total": v_total,
                    "region": "",
                })

    # ── Render Picks ──
    if not picks:
        st.info("Select a data source above to generate picks.")
        return

    st.markdown("---")
    st.subheader("Recommended Bets")

    all_bets = []  # collect all bets for summary table

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
        fav_prob = p if fav == t1 else 1 - p
        bt_ml_pct = get_bt_pct(fav_prob, "ml")
        bt_ats_pct = get_bt_pct(fav_prob, "ats")

        has_vegas = pick["vegas_ml_t1"] is not None

        game_bets = []

        # ── Moneyline Analysis ──
        if has_vegas:
            v_ml_fav = pick["vegas_ml_t1"] if fav == t1 else pick["vegas_ml_t2"]
            v_ml_dog = pick["vegas_ml_t2"] if fav == t1 else pick["vegas_ml_t1"]
            vegas_implied = _implied_prob(v_ml_fav)
            ml_ev = _ev(fav_prob, v_ml_fav)
            ml_kelly = _kelly(fav_prob, v_ml_fav)
            ml_edge = fav_prob - vegas_implied
            ml_score = _edge_rating(ml_ev, ml_kelly, bt_ml_pct)
            ml_label, ml_color = _edge_label(ml_score)

            if ml_ev > 0:
                game_bets.append({
                    "Game": f"{s1t}{n1} vs {s2t}{n2}",
                    "Bet Type": "Moneyline",
                    "Pick": f"{fav_name} ({'+' if v_ml_fav > 0 else ''}{v_ml_fav})",
                    "Model %": f"{fav_prob*100:.1f}%",
                    "Vegas Implied": f"{vegas_implied*100:.1f}%",
                    "Edge": f"{ml_edge*100:+.1f}%",
                    "EV ($100)": f"${ml_ev:+.1f}",
                    "Kelly %": f"{ml_kelly*100:.1f}%",
                    "Backtest": f"{bt_ml_pct:.0f}%",
                    "Rating": ml_score,
                    "_label": ml_label, "_color": ml_color,
                })

            # Check dog value too
            dog_prob = 1 - fav_prob
            dog_name = n2 if fav == t1 else n1
            dog_ev = _ev(dog_prob, v_ml_dog)
            if dog_ev > 0:
                dog_implied = _implied_prob(v_ml_dog)
                dog_kelly = _kelly(dog_prob, v_ml_dog)
                dog_edge = dog_prob - dog_implied
                dog_score = _edge_rating(dog_ev, dog_kelly, 50)
                dog_label, dog_color = _edge_label(dog_score)
                game_bets.append({
                    "Game": f"{s1t}{n1} vs {s2t}{n2}",
                    "Bet Type": "ML Dog",
                    "Pick": f"{dog_name} ({'+' if v_ml_dog > 0 else ''}{v_ml_dog})",
                    "Model %": f"{dog_prob*100:.1f}%",
                    "Vegas Implied": f"{dog_implied*100:.1f}%",
                    "Edge": f"{dog_edge*100:+.1f}%",
                    "EV ($100)": f"${dog_ev:+.1f}",
                    "Kelly %": f"{dog_kelly*100:.1f}%",
                    "Backtest": "—",
                    "Rating": dog_score,
                    "_label": dog_label, "_color": dog_color,
                })

        # ── Spread Analysis ──
        if has_vegas and pick["vegas_spread"] is not None:
            v_spread = pick["vegas_spread"]
            m_spread = pick["model_spread"]
            spread_diff = v_spread - m_spread  # positive = vegas giving more points than model
            # If model has team favored by 7 but vegas only by 3, take the favorite ATS
            if abs(spread_diff) >= 1.0:
                spread_ev = abs(spread_diff) * 4.0  # rough EV scaling
                spread_kelly = min(abs(spread_diff) / 20, 0.15)
                spread_score = _edge_rating(spread_ev, spread_kelly, bt_ats_pct)
                spread_label, spread_color = _edge_label(spread_score)

                if m_spread < v_spread:
                    # Model has fav winning by MORE than Vegas thinks -> bet favorite ATS
                    pick_text = f"{fav_name} {v_spread:+.1f}"
                else:
                    # Model has underdog closer or winning -> bet dog ATS
                    dog_name = n2 if fav == t1 else n1
                    pick_text = f"{dog_name} {-v_spread:+.1f}"

                game_bets.append({
                    "Game": f"{s1t}{n1} vs {s2t}{n2}",
                    "Bet Type": "Spread",
                    "Pick": pick_text,
                    "Model %": f"Model: {m_spread:+.1f}",
                    "Vegas Implied": f"Vegas: {v_spread:+.1f}",
                    "Edge": f"{abs(spread_diff):.1f} pts",
                    "EV ($100)": f"${spread_ev:+.1f}",
                    "Kelly %": f"{spread_kelly*100:.1f}%",
                    "Backtest": f"{bt_ats_pct:.0f}%",
                    "Rating": spread_score,
                    "_label": spread_label, "_color": spread_color,
                })

        # ── Totals Analysis ──
        if has_vegas and pick["vegas_total"] is not None:
            v_total = pick["vegas_total"]
            m_total = pick["model_total"]
            total_diff = m_total - v_total
            if abs(total_diff) >= 2.0:
                total_ev = abs(total_diff) * 3.0
                total_kelly = min(abs(total_diff) / 25, 0.12)
                total_score = _edge_rating(total_ev, total_kelly, 55)
                total_label, total_color = _edge_label(total_score)

                ou_pick = "OVER" if m_total > v_total else "UNDER"
                game_bets.append({
                    "Game": f"{s1t}{n1} vs {s2t}{n2}",
                    "Bet Type": "Total",
                    "Pick": f"{ou_pick} {v_total}",
                    "Model %": f"Proj: {m_total:.1f}",
                    "Vegas Implied": f"Line: {v_total:.1f}",
                    "Edge": f"{abs(total_diff):.1f} pts",
                    "EV ($100)": f"${total_ev:+.1f}",
                    "Kelly %": f"{total_kelly*100:.1f}%",
                    "Backtest": "—",
                    "Rating": total_score,
                    "_label": total_label, "_color": total_color,
                })

        # If no Vegas odds, show model projections only
        if not has_vegas:
            game_bets.append({
                "Game": f"{s1t}{n1} vs {s2t}{n2}",
                "Bet Type": "Projection",
                "Pick": f"{fav_name} ({pick['model_ml_t1'] if fav == t1 else pick['model_ml_t2']})",
                "Model %": f"{fav_prob*100:.1f}%",
                "Vegas Implied": "No Vegas line",
                "Edge": "—",
                "EV ($100)": "—",
                "Kelly %": "—",
                "Backtest": f"{bt_ml_pct:.0f}%",
                "Rating": 0,
                "_label": "NO LINE", "_color": "#666",
            })

        all_bets.extend(game_bets)

    # Sort by rating descending
    all_bets.sort(key=lambda x: x["Rating"], reverse=True)

    # ── Flagged Bets (top picks) ──
    flagged = [b for b in all_bets if b["Rating"] >= 40]
    if flagged:
        st.markdown("### TOP PICKS")
        for bet in flagged:
            label, color = bet["_label"], bet["_color"]
            st.markdown(
                f'<div style="background:#1a1d24; border-left:4px solid {color}; '
                f'border-radius:4px; padding:12px 16px; margin:6px 0; '
                f'display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'<div style="flex:2; min-width:200px;">'
                f'<span style="font-weight:700; font-size:15px;">{bet["Game"]}</span><br>'
                f'<span style="color:{color}; font-weight:700; font-size:18px;">{bet["Pick"]}</span>'
                f'<span style="color:#888; margin-left:8px;">({bet["Bet Type"]})</span>'
                f'</div>'
                f'<div style="flex:1; min-width:120px; text-align:center;">'
                f'<div style="font-size:12px; color:#888;">Model vs Vegas</div>'
                f'<div style="font-size:14px;">{bet["Model %"]} vs {bet["Vegas Implied"]}</div>'
                f'</div>'
                f'<div style="flex:1; min-width:100px; text-align:center;">'
                f'<div style="font-size:12px; color:#888;">Edge</div>'
                f'<div style="font-size:16px; font-weight:700; color:{color};">{bet["Edge"]}</div>'
                f'</div>'
                f'<div style="flex:1; min-width:100px; text-align:center;">'
                f'<div style="font-size:12px; color:#888;">EV / Kelly</div>'
                f'<div style="font-size:14px;">{bet["EV ($100)"]} / {bet["Kelly %"]}</div>'
                f'</div>'
                f'<div style="flex:0; min-width:80px; text-align:center;">'
                f'<div style="background:{color}; color:#000; font-weight:800; padding:4px 12px; '
                f'border-radius:4px; font-size:13px;">{label} ({bet["Rating"]})</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    # ── Full Bet Board ──
    st.markdown("---")
    st.subheader("Full Bet Board")

    board_rows = []
    for bet in all_bets:
        row = {k: v for k, v in bet.items() if not k.startswith("_")}
        row["Signal"] = bet["_label"]
        board_rows.append(row)

    if board_rows:
        st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)

    # ── Legend ──
    with st.expander("How to read this page"):
        st.markdown("""
**Edge** — How much our model disagrees with Vegas. Positive = our model likes it more.

**EV ($100)** — Expected profit on a $100 bet. Positive = profitable long-term.

**Kelly %** — Recommended bet size as % of bankroll. Use half or quarter Kelly for safety.

**Backtest** — How often this type of pick has won historically.

**Rating** (0-100) — Overall strength combining EV, Kelly, and backtest accuracy.

| Signal | Rating | Action |
|--------|--------|--------|
| **STRONG** | 70+ | High-confidence play |
| **MODERATE** | 45-69 | Standard bet |
| **SLIGHT** | 25-44 | Small position |
| **SKIP** | <25 | Pass |
""")


# ──────────────────────────── Main ────────────────────────────

teams = load_teams()
preds = load_predictions()
seeds = load_seeds()
m_slots, w_slots = load_slots()
conferences = load_conferences()

# Sidebar
st.sidebar.title("\U0001f3c0 March ML Mania")
st.sidebar.caption("2026 NCAA Tournament Predictions & Betting")

gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], horizontal=True)
prefix = "M" if gender == "Men's" else "W"

# Settings
with st.sidebar.expander("Settings"):
    st.text_input(
        "Odds API Key", value=_get_odds_key(), key="odds_api_key", type="password",
        help="Get a free key at the-odds-api.com (500 requests/month)",
    )
    if "cached_odds" in st.session_state:
        remaining = st.session_state["cached_odds"].get("remaining", "?")
        st.caption(f"API credits remaining: {remaining}/500")

gender_seeds = seeds[seeds.TeamID < 3000] if prefix == "M" else seeds[seeds.TeamID >= 3000]
gender_slots = m_slots if prefix == "M" else w_slots

page = st.sidebar.radio(
    "Navigate",
    ["\U0001f4ca Rankings", "\U0001f93c Head-to-Head", "\U0001f4c8 Tournament Odds",
     "\U0001f3c6 Bracket", "\U0001f4b0 Betting Picks", "\U0001f9ea Backtest"],
)

st.sidebar.markdown("---")
st.sidebar.caption("For entertainment purposes only. Not financial advice.")

if "Rankings" in page:
    page_rankings(prefix, teams, gender_seeds, conferences)
elif "Head-to-Head" in page:
    coach_info = load_coach_data() if prefix == "M" else {}
    knn_data = build_knn_lookup(prefix)
    h2h_history = build_h2h_history(prefix)
    seed_history = build_seed_history(prefix)
    page_h2h(prefix, teams, gender_seeds, preds, coach_info, knn_data, h2h_history, seed_history)
elif "Tournament Odds" in page:
    page_odds(prefix, teams, gender_seeds, gender_slots, preds)
elif "Bracket" in page:
    page_bracket(prefix, teams, gender_seeds, gender_slots, preds)
elif "Betting Picks" in page:
    page_picks(prefix, teams, gender_seeds, preds)
elif "Backtest" in page:
    page_backtest(prefix, teams)
