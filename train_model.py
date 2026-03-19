"""
March ML Mania 2026 — Optimized ensemble prediction model (v2).

All hot loops vectorized with numpy for speed.
Features: Elo, time-weighted stats, KNN similar opponents, coach tenure/success,
Massey ordinals, conference strength, head-to-head, efficiency metrics.
Tuned with Optuna, temporal CV, Platt scaling, feature selection.
"""

import os
import sys
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def log(msg):
    print(msg, flush=True)


# ===========================================================================
# 1. Load data
# ===========================================================================
def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def load_data():
    log("Loading data...")
    d = {}
    for prefix in ("M", "W"):
        d[f"{prefix}RegularSeasonDetailedResults"] = load_csv(f"{prefix}RegularSeasonDetailedResults.csv")
        d[f"{prefix}RegularSeasonCompactResults"] = load_csv(f"{prefix}RegularSeasonCompactResults.csv")
        d[f"{prefix}NCAATourneyCompactResults"] = load_csv(f"{prefix}NCAATourneyCompactResults.csv")
        d[f"{prefix}NCAATourneyDetailedResults"] = load_csv(f"{prefix}NCAATourneyDetailedResults.csv")
        d[f"{prefix}NCAATourneySeeds"] = load_csv(f"{prefix}NCAATourneySeeds.csv")
        d[f"{prefix}TeamConferences"] = load_csv(f"{prefix}TeamConferences.csv")
    d["MTeamCoaches"] = load_csv("MTeamCoaches.csv")
    d["MMasseyOrdinals"] = load_csv("MMasseyOrdinals.csv")
    d["SampleSubmissionStage1"] = load_csv("SampleSubmissionStage1.csv")
    try:
        d["SampleSubmissionStage2"] = load_csv("SampleSubmissionStage2.csv")
    except FileNotFoundError:
        pass
    log("  Data loaded.")
    return d


# ===========================================================================
# 2. Elo rating system (vectorized with numpy arrays)
# ===========================================================================
def build_elo_ratings(compact_results, K=20, HOME_ADV=100, SEASON_DECAY=0.75):
    """Vectorized Elo using numpy arrays and dict lookups."""
    log("Building Elo ratings...")
    df = compact_results.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    seasons = df["Season"].values
    w_ids = df["WTeamID"].values
    l_ids = df["LTeamID"].values
    w_scores = df["WScore"].values
    l_scores = df["LScore"].values
    w_locs = df["WLoc"].values if "WLoc" in df.columns else np.full(len(df), "N")

    elo = {}  # team_id -> current rating
    snapshots = {}  # (season, team_id) -> rating
    prev_season = None

    for i in range(len(df)):
        season = seasons[i]
        w_id, l_id = int(w_ids[i]), int(l_ids[i])

        # Season transition
        if prev_season is not None and season != prev_season:
            for tid, rating in elo.items():
                snapshots[(int(prev_season), tid)] = rating
                elo[tid] = 1500 + SEASON_DECAY * (rating - 1500)
        prev_season = season

        elo.setdefault(w_id, 1500)
        elo.setdefault(l_id, 1500)

        w_elo, l_elo = elo[w_id], elo[l_id]
        loc = w_locs[i]
        w_adj = w_elo + (HOME_ADV if loc == "H" else 0)
        l_adj = l_elo + (HOME_ADV if loc == "A" else 0)

        exp_w = 1.0 / (1.0 + 10 ** ((l_adj - w_adj) / 400))
        mov = int(w_scores[i]) - int(l_scores[i])
        k_mult = K * np.log(max(mov, 1) + 1) * 0.7

        elo_diff = abs(w_elo - l_elo)
        dampener = 2.2 / ((elo_diff * 0.001) + 2.2)
        k_mult *= dampener

        elo[w_id] = w_elo + k_mult * (1 - exp_w)
        elo[l_id] = l_elo - k_mult * (1 - exp_w)

    # Final season snapshot
    if prev_season is not None:
        for tid, rating in elo.items():
            snapshots[(int(prev_season), tid)] = rating

    log(f"  Elo: {len(snapshots):,} (season, team) ratings")
    return snapshots


# ===========================================================================
# 3. Time-weighted team season stats (vectorized)
# ===========================================================================
STAT_COLS_W = ["WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
STAT_COLS_L = ["LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]
STAT_NAMES = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def build_team_season_stats(detailed_results, decay=0.03):
    """Build time-weighted season stats per team."""
    log("Building time-weighted team stats...")

    col_map_w = dict(zip(
        ["WTeamID", "WScore", "LScore"] + STAT_COLS_W + STAT_COLS_L,
        ["TeamID", "Score", "OppScore"] + STAT_NAMES + ["Opp" + s for s in STAT_NAMES]
    ))
    col_map_l = dict(zip(
        ["LTeamID", "LScore", "WScore"] + STAT_COLS_L + STAT_COLS_W,
        ["TeamID", "Score", "OppScore"] + STAT_NAMES + ["Opp" + s for s in STAT_NAMES]
    ))

    keep_cols = ["Season", "DayNum", "TeamID", "Score", "OppScore"] + STAT_NAMES + ["Opp" + s for s in STAT_NAMES]

    wins = detailed_results.rename(columns=col_map_w)[keep_cols].copy()
    wins["Win"] = 1
    losses = detailed_results.rename(columns=col_map_l)[keep_cols].copy()
    losses["Win"] = 0

    all_games = pd.concat([wins, losses], ignore_index=True).sort_values(["Season", "TeamID", "DayNum"])

    stat_cols = ["Score", "OppScore"] + STAT_NAMES + ["Opp" + s for s in STAT_NAMES]
    results = []

    for (season, team_id), grp in all_games.groupby(["Season", "TeamID"]):
        max_day = grp["DayNum"].max()
        weights = np.exp(decay * (grp["DayNum"].values - max_day))
        total_w = weights.sum()

        row = {"Season": season, "TeamID": team_id}
        row["Games"] = len(grp)
        row["Wins"] = grp["Win"].sum()
        row["WinPct"] = grp["Win"].sum() / len(grp)

        for col in stat_cols:
            row[f"Avg{col}"] = np.average(grp[col].values, weights=weights)

        last10 = grp.tail(10)
        row["Last10_WinPct"] = last10["Win"].mean()
        row["Last10_AvgMargin"] = (last10["Score"] - last10["OppScore"]).mean()
        results.append(row)

    agg = pd.DataFrame(results)

    agg["AvgMargin"] = agg["AvgScore"] - agg["AvgOppScore"]
    agg["FGPct"] = agg["AvgFGM"] / agg["AvgFGA"].replace(0, 1)
    agg["FG3Pct"] = agg["AvgFGM3"] / agg["AvgFGA3"].replace(0, 1)
    agg["FTPct"] = agg["AvgFTM"] / agg["AvgFTA"].replace(0, 1)
    agg["OppFGPct"] = agg["AvgOppFGM"] / agg["AvgOppFGA"].replace(0, 1)
    agg["AvgReb"] = agg["AvgOR"] + agg["AvgDR"]
    agg["AvgOppReb"] = agg["AvgOppOR"] + agg["AvgOppDR"]
    agg["RebMargin"] = agg["AvgReb"] - agg["AvgOppReb"]
    agg["TOMargin"] = agg["AvgOppTO"] - agg["AvgTO"]
    agg["AstTORatio"] = agg["AvgAst"] / agg["AvgTO"].replace(0, 1)
    agg["EffFGPct"] = (agg["AvgFGM"] + 0.5 * agg["AvgFGM3"]) / agg["AvgFGA"].replace(0, 1)
    agg["OppEffFGPct"] = (agg["AvgOppFGM"] + 0.5 * agg["AvgOppFGM3"]) / agg["AvgOppFGA"].replace(0, 1)
    agg["Poss"] = agg["AvgFGA"] - agg["AvgOR"] + agg["AvgTO"] + 0.475 * agg["AvgFTA"]
    agg["OppPoss"] = agg["AvgOppFGA"] - agg["AvgOppOR"] + agg["AvgOppTO"] + 0.475 * agg["AvgOppFTA"]
    agg["OffEff"] = agg["AvgScore"] / agg["Poss"].replace(0, 1) * 100
    agg["DefEff"] = agg["AvgOppScore"] / agg["OppPoss"].replace(0, 1) * 100
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]

    # Pre-index for fast lookup
    agg = agg.set_index(["Season", "TeamID"])
    log(f"  Stats: {len(agg):,} (season, team) rows")
    return agg


# ===========================================================================
# 4. KNN similar-opponent features
# ===========================================================================
KNN_PROFILE_COLS = [
    "WinPct", "AvgScore", "AvgOppScore", "FGPct", "FG3Pct", "FTPct",
    "AvgReb", "AvgAst", "AvgTO", "AvgStl", "AvgBlk", "AvgMargin",
]


class KNNFeatureLookup:
    def __init__(self, team_stats, detailed_results, n_neighbors=5):
        log("Building KNN lookup...")
        self._season_data = {}

        wins = detailed_results[["Season", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()
        wins.columns = ["Season", "TeamID", "OppTeamID", "Score", "OppScore"]
        wins["Win"] = 1
        losses = detailed_results[["Season", "LTeamID", "WTeamID", "LScore", "WScore"]].copy()
        losses.columns = ["Season", "TeamID", "OppTeamID", "Score", "OppScore"]
        losses["Win"] = 0
        game_results = pd.concat([wins, losses], ignore_index=True)

        # Reset index temporarily for groupby access
        ts = team_stats.reset_index() if team_stats.index.names[0] is not None else team_stats

        for season in ts["Season"].unique():
            ss = ts[ts["Season"] == season]
            if len(ss) < n_neighbors + 1:
                continue

            profile = ss[KNN_PROFILE_COLS].fillna(0).values
            scaler = StandardScaler()
            profile_scaled = scaler.fit_transform(profile)
            team_ids = ss["TeamID"].values

            nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(ss)), metric="euclidean")
            nn.fit(profile_scaled)
            _, indices = nn.kneighbors(profile_scaled)

            neighbors_map = {}
            for i, tid in enumerate(team_ids):
                neighbors_map[tid] = set(int(team_ids[j]) for j in indices[i] if team_ids[j] != tid)

            # Build game lookup using vectorized groupby
            sg = game_results[game_results["Season"] == season]
            game_lookup = {}
            for (tid, opp), grp in sg.groupby(["TeamID", "OppTeamID"]):
                game_lookup[(int(tid), int(opp))] = (grp["Win"].values, (grp["Score"] - grp["OppScore"]).values)

            self._season_data[season] = (neighbors_map, game_lookup)

        log(f"  KNN ready: {len(self._season_data)} seasons")

    def get(self, season, team_a, team_b):
        if season not in self._season_data:
            return None
        neighbors_map, game_lookup = self._season_data[season]
        similar_to_b = neighbors_map.get(team_b)
        if not similar_to_b:
            return None

        all_wins, all_margins = [], []
        for sim_team in similar_to_b:
            data = game_lookup.get((team_a, sim_team))
            if data:
                all_wins.extend(data[0])
                all_margins.extend(data[1])

        if not all_wins:
            return None
        return {
            "KNN_WinRate": np.mean(all_wins),
            "KNN_AvgMargin": np.mean(all_margins),
            "KNN_NumGames": len(all_wins),
        }

    def __len__(self):
        return len(self._season_data)


# ===========================================================================
# 5. Coach features (men only) — vectorized
# ===========================================================================
def build_coach_features(coaches_df, tourney_results):
    log("Building coach features...")
    end_coaches = coaches_df.sort_values("LastDayNum").groupby(["Season", "TeamID"]).last().reset_index()
    end_coaches = end_coaches[["Season", "TeamID", "CoachName"]]

    # Tenure: consecutive years with same team
    tenure_rows = []
    for (team_id, coach), grp in end_coaches.groupby(["TeamID", "CoachName"]):
        seasons = sorted(grp["Season"].values)
        for i, s in enumerate(seasons):
            tenure = 1
            for j in range(i - 1, -1, -1):
                if seasons[j] == seasons[j + 1] - 1:
                    tenure += 1
                else:
                    break
            tenure_rows.append({"Season": s, "TeamID": team_id, "CoachName": coach, "CoachTenure": tenure})
    tenure_df = pd.DataFrame(tenure_rows)

    # Prior tourney success — vectorized
    coach_map = end_coaches.set_index(["Season", "TeamID"])["CoachName"].to_dict()

    tr = tourney_results[["Season", "WTeamID", "LTeamID"]].copy()
    w_coaches = tr.apply(lambda r: coach_map.get((r["Season"], r["WTeamID"]), None), axis=1)
    l_coaches = tr.apply(lambda r: coach_map.get((r["Season"], r["LTeamID"]), None), axis=1)

    records = []
    for _, r in tr.iterrows():
        wc = coach_map.get((r["Season"], r["WTeamID"]))
        lc = coach_map.get((r["Season"], r["LTeamID"]))
        if wc:
            records.append((r["Season"], wc, 1))
        if lc:
            records.append((r["Season"], lc, 0))

    if not records:
        tenure_df["CoachTourneyWins"] = 0
        tenure_df["CoachTourneyGames"] = 0
        tenure_df["CoachTourneyWinRate"] = 0.0
        return tenure_df.set_index(["Season", "TeamID"])

    tcw = pd.DataFrame(records, columns=["Season", "CoachName", "Win"])
    # Cumulative prior record
    coach_records = []
    for coach, cdf in tcw.groupby("CoachName"):
        season_groups = cdf.groupby("Season").agg(wins=("Win", "sum"), games=("Win", "count")).sort_index()
        cum_wins, cum_games = 0, 0
        for s, row in season_groups.iterrows():
            coach_records.append({"Season": s, "CoachName": coach,
                                  "CoachTourneyWins": cum_wins, "CoachTourneyGames": cum_games})
            cum_wins += row["wins"]
            cum_games += row["games"]

    cr = pd.DataFrame(coach_records)
    cr["CoachTourneyWinRate"] = cr["CoachTourneyWins"] / cr["CoachTourneyGames"].replace(0, 1)

    result = tenure_df.merge(cr, on=["Season", "CoachName"], how="left")
    for col in ["CoachTourneyWins", "CoachTourneyGames", "CoachTourneyWinRate"]:
        result[col] = result[col].fillna(0)

    result = result[["Season", "TeamID", "CoachTenure", "CoachTourneyWins", "CoachTourneyGames", "CoachTourneyWinRate"]]
    result = result.set_index(["Season", "TeamID"])
    log(f"  Coach features: {len(result):,} rows")
    return result


# ===========================================================================
# 6. Seeds
# ===========================================================================
def build_seed_features(seeds_df):
    seeds = seeds_df.copy()
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    return seeds.set_index(["Season", "TeamID"])["SeedNum"]


# ===========================================================================
# 7. Massey ordinals
# ===========================================================================
def build_massey_features(massey_df):
    log("Building Massey features...")
    massey = massey_df[massey_df["RankingDayNum"] <= 133].copy()
    last_day = massey.groupby(["Season", "SystemName"])["RankingDayNum"].transform("max")
    massey = massey[massey["RankingDayNum"] == last_day]

    agg = massey.groupby(["Season", "TeamID"]).agg(
        MasseyMeanRank=("OrdinalRank", "mean"),
        MasseyMedianRank=("OrdinalRank", "median"),
        MasseyMinRank=("OrdinalRank", "min"),
        MasseyMaxRank=("OrdinalRank", "max"),
        MasseyStdRank=("OrdinalRank", "std"),
    )
    agg["MasseyStdRank"] = agg["MasseyStdRank"].fillna(0)
    log(f"  Massey: {len(agg):,} rows")
    return agg  # already indexed by (Season, TeamID)


# ===========================================================================
# 8. Conference strength
# ===========================================================================
def build_conference_strength(team_confs, team_stats):
    log("Building conference strength...")
    ts = team_stats.reset_index() if team_stats.index.names[0] is not None else team_stats
    merged = team_confs.merge(ts[["Season", "TeamID", "WinPct"]], on=["Season", "TeamID"], how="left")
    conf_agg = merged.groupby(["Season", "ConfAbbrev"]).agg(
        ConfAvgWinPct=("WinPct", "mean"),
        ConfMedianWinPct=("WinPct", "median"),
    ).reset_index()
    result = team_confs.merge(conf_agg, on=["Season", "ConfAbbrev"], how="left")
    return result.set_index(["Season", "TeamID"])[["ConfAvgWinPct", "ConfMedianWinPct"]]


# ===========================================================================
# 9. Head-to-head (fast dict-based)
# ===========================================================================
def build_h2h_features(compact_results):
    """H2H record between team pairs, snapshot per season."""
    log("Building H2H features...")
    df = compact_results.sort_values(["Season", "DayNum"])

    # Accumulate all-time H2H
    h2h = {}  # (low_id, high_id) -> [low_id_wins, total_games]
    snapshots = {}

    seasons = df["Season"].values
    w_ids = df["WTeamID"].values
    l_ids = df["LTeamID"].values
    prev_season = None

    for i in range(len(df)):
        s = int(seasons[i])
        if prev_season is not None and s != prev_season:
            for key, val in h2h.items():
                snapshots[(prev_season, key[0], key[1])] = (val[0] / val[1] if val[1] > 0 else 0.5, val[1])
        prev_season = s

        w, l = int(w_ids[i]), int(l_ids[i])
        low, high = min(w, l), max(w, l)
        if (low, high) not in h2h:
            h2h[(low, high)] = [0, 0]
        h2h[(low, high)][1] += 1
        if w == low:
            h2h[(low, high)][0] += 1

    if prev_season is not None:
        for key, val in h2h.items():
            snapshots[(prev_season, key[0], key[1])] = (val[0] / val[1] if val[1] > 0 else 0.5, val[1])

    log(f"  H2H: {len(snapshots):,} snapshots")
    return snapshots


# ===========================================================================
# 10. Build matchup features (fast indexed lookups)
# ===========================================================================
TEAM_STAT_FEATURES = [
    "WinPct", "AvgScore", "AvgOppScore", "AvgMargin",
    "FGPct", "FG3Pct", "FTPct", "OppFGPct", "EffFGPct", "OppEffFGPct",
    "AvgReb", "AvgOppReb", "RebMargin", "TOMargin", "AstTORatio",
    "OffEff", "DefEff", "NetEff",
    "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3", "AvgFTM", "AvgFTA",
    "AvgOR", "AvgDR", "AvgAst", "AvgTO", "AvgStl", "AvgBlk", "AvgPF",
    "AvgOppFGM", "AvgOppFGA", "AvgOppFGM3", "AvgOppFGA3",
    "Last10_WinPct", "Last10_AvgMargin",
]
COACH_FEATURES = ["CoachTenure", "CoachTourneyWins", "CoachTourneyGames", "CoachTourneyWinRate"]
MASSEY_FEATURES = ["MasseyMeanRank", "MasseyMedianRank", "MasseyMinRank", "MasseyMaxRank", "MasseyStdRank"]


def _safe_get(indexed_df, key, col, default=0):
    """Fast .loc with fallback."""
    try:
        val = indexed_df.loc[key, col]
        return default if pd.isna(val) else val
    except KeyError:
        return default


def build_matchup_features(t1_id, t2_id, season, team_stats, seed_feats,
                           coach_feats, massey_feats, conf_feats, knn_feats,
                           elo_ratings, h2h_snapshots, is_men):
    key1 = (season, t1_id)
    key2 = (season, t2_id)

    if key1 not in team_stats.index or key2 not in team_stats.index:
        return None

    features = {}
    t1 = team_stats.loc[key1]
    t2 = team_stats.loc[key2]

    # Team stats diffs
    for col in TEAM_STAT_FEATURES:
        v1 = t1.get(col, 0)
        v2 = t2.get(col, 0)
        if pd.isna(v1): v1 = 0
        if pd.isna(v2): v2 = 0
        features[f"Diff_{col}"] = v1 - v2
        features[f"T1_{col}"] = v1
        features[f"T2_{col}"] = v2

    # Elo
    elo1 = elo_ratings.get(key1, 1500)
    elo2 = elo_ratings.get(key2, 1500)
    features["T1_Elo"] = elo1
    features["T2_Elo"] = elo2
    features["Diff_Elo"] = elo1 - elo2
    features["Elo_WinProb"] = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))

    # Seeds
    seed1 = seed_feats.get(key1, 8) if key1 in seed_feats.index else 8
    seed2 = seed_feats.get(key2, 8) if key2 in seed_feats.index else 8
    features["T1_Seed"] = seed1
    features["T2_Seed"] = seed2
    features["Diff_Seed"] = seed1 - seed2
    features["SeedMatchup"] = seed1 * seed2

    # Coach (men only)
    if is_men and coach_feats is not None:
        for col in COACH_FEATURES:
            v1 = _safe_get(coach_feats, key1, col, 0)
            v2 = _safe_get(coach_feats, key2, col, 0)
            features[f"Diff_{col}"] = v1 - v2
            features[f"T1_{col}"] = v1
            features[f"T2_{col}"] = v2

    # Massey (men only)
    if is_men and massey_feats is not None:
        for col in MASSEY_FEATURES:
            v1 = _safe_get(massey_feats, key1, col, 200)
            v2 = _safe_get(massey_feats, key2, col, 200)
            features[f"Diff_{col}"] = v1 - v2
            features[f"T1_{col}"] = v1
            features[f"T2_{col}"] = v2

    # Conference strength
    if conf_feats is not None:
        for col in ["ConfAvgWinPct", "ConfMedianWinPct"]:
            v1 = _safe_get(conf_feats, key1, col, 0.5)
            v2 = _safe_get(conf_feats, key2, col, 0.5)
            features[f"Diff_{col}"] = v1 - v2

    # KNN
    if knn_feats is not None and len(knn_feats) > 0:
        k1 = knn_feats.get(season, t1_id, t2_id)
        k2 = knn_feats.get(season, t2_id, t1_id)
        features["T1_KNN_WinRate"] = k1["KNN_WinRate"] if k1 else 0.5
        features["T1_KNN_AvgMargin"] = k1["KNN_AvgMargin"] if k1 else 0
        features["T1_KNN_NumGames"] = k1["KNN_NumGames"] if k1 else 0
        features["T2_KNN_WinRate"] = k2["KNN_WinRate"] if k2 else 0.5
        features["T2_KNN_AvgMargin"] = k2["KNN_AvgMargin"] if k2 else 0
        features["T2_KNN_NumGames"] = k2["KNN_NumGames"] if k2 else 0
        features["Diff_KNN_WinRate"] = features["T1_KNN_WinRate"] - features["T2_KNN_WinRate"]
        features["Diff_KNN_AvgMargin"] = features["T1_KNN_AvgMargin"] - features["T2_KNN_AvgMargin"]

    # H2H
    if h2h_snapshots is not None:
        low, high = min(t1_id, t2_id), max(t1_id, t2_id)
        h2h_data = h2h_snapshots.get((season - 1, low, high))
        if h2h_data:
            wr, ng = h2h_data
            features["H2H_WinRate"] = wr if t1_id < t2_id else 1 - wr
            features["H2H_NumGames"] = ng
        else:
            features["H2H_WinRate"] = 0.5
            features["H2H_NumGames"] = 0

    return features


# ===========================================================================
# 11. Build training data
# ===========================================================================
def build_training_data(prefix, data, team_stats, seed_feats, coach_feats,
                        massey_feats, conf_feats, knn_feats, elo_ratings,
                        h2h_snapshots):
    log("Building training data...")
    is_men = prefix == "M"
    tourney = data[f"{prefix}NCAATourneyCompactResults"]

    rows, labels, seasons = [], [], []
    for _, game in tourney.iterrows():
        season = game["Season"]
        w_id, l_id = game["WTeamID"], game["LTeamID"]
        if w_id < l_id:
            t1, t2, label = w_id, l_id, 1.0
        else:
            t1, t2, label = l_id, w_id, 0.0

        feats = build_matchup_features(t1, t2, season, team_stats, seed_feats,
                                        coach_feats, massey_feats, conf_feats, knn_feats,
                                        elo_ratings, h2h_snapshots, is_men)
        if feats is not None:
            rows.append(feats)
            labels.append(label)
            seasons.append(season)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    season_arr = np.array(seasons)
    log(f"  Samples: {len(X):,}, Features: {X.shape[1]}")
    return X, y, season_arr


# ===========================================================================
# 12. Feature selection
# ===========================================================================
def select_features(X, y, feature_cols, min_importance=0.001):
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               eval_metric="logloss", random_state=42)
    model.fit(X[feature_cols].values, y)
    selected = [col for col, imp in zip(feature_cols, model.feature_importances_) if imp >= min_importance]
    log(f"  Feature selection: {len(feature_cols)} -> {len(selected)}")
    return selected


# ===========================================================================
# 13. Temporal CV splits
# ===========================================================================
def temporal_cv_splits(seasons, min_train_seasons=8):
    unique = sorted(set(seasons))
    splits = []
    for i in range(min_train_seasons, len(unique)):
        val_season = unique[i]
        tr_mask = seasons < val_season
        va_mask = seasons == val_season
        if tr_mask.sum() > 0 and va_mask.sum() > 0:
            splits.append((np.where(tr_mask)[0], np.where(va_mask)[0]))
    return splits


def brier_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ===========================================================================
# 14. Optuna + train + calibrate
# ===========================================================================
def optimize_and_train(X, y, seasons, feature_cols, n_trials=30):
    Xf = X[feature_cols].values
    splits = temporal_cv_splits(seasons)
    log(f"  Temporal CV: {len(splits)} folds")

    # --- XGBoost ---
    def xgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        scores = []
        for tr_i, va_i in splits:
            m = xgb.XGBClassifier(**p, eval_metric="logloss", random_state=42, early_stopping_rounds=50)
            m.fit(Xf[tr_i], y[tr_i], eval_set=[(Xf[va_i], y[va_i])], verbose=False)
            scores.append(brier_score(y[va_i], m.predict_proba(Xf[va_i])[:, 1]))
        return np.mean(scores)

    log("  Tuning XGBoost...")
    s_xgb = optuna.create_study(direction="minimize")
    s_xgb.optimize(xgb_obj, n_trials=n_trials)
    log(f"    Best: {s_xgb.best_value:.5f}")

    # --- LightGBM ---
    def lgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        }
        scores = []
        for tr_i, va_i in splits:
            m = lgb.LGBMClassifier(**p, random_state=42, verbose=-1)
            m.fit(Xf[tr_i], y[tr_i], eval_set=[(Xf[va_i], y[va_i])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            scores.append(brier_score(y[va_i], m.predict_proba(Xf[va_i])[:, 1]))
        return np.mean(scores)

    log("  Tuning LightGBM...")
    s_lgb = optuna.create_study(direction="minimize")
    s_lgb.optimize(lgb_obj, n_trials=n_trials)
    log(f"    Best: {s_lgb.best_value:.5f}")

    # --- CatBoost ---
    def cb_obj(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5.0),
        }
        scores = []
        for tr_i, va_i in splits:
            m = cb.CatBoostClassifier(**p, random_seed=42, verbose=0, early_stopping_rounds=50)
            m.fit(Xf[tr_i], y[tr_i], eval_set=(Xf[va_i], y[va_i]))
            scores.append(brier_score(y[va_i], m.predict_proba(Xf[va_i])[:, 1]))
        return np.mean(scores)

    log("  Tuning CatBoost...")
    s_cb = optuna.create_study(direction="minimize")
    s_cb.optimize(cb_obj, n_trials=n_trials)
    log(f"    Best: {s_cb.best_value:.5f}")

    # --- Train final models with temporal CV for OOF ---
    log("  Training final models...")
    bp_xgb, bp_lgb, bp_cb = s_xgb.best_params, s_lgb.best_params, s_cb.best_params

    oof_xgb = np.full(len(y), np.nan)
    oof_lgb = np.full(len(y), np.nan)
    oof_cb = np.full(len(y), np.nan)

    for tr_i, va_i in splits:
        m = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42, early_stopping_rounds=50)
        m.fit(Xf[tr_i], y[tr_i], eval_set=[(Xf[va_i], y[va_i])], verbose=False)
        oof_xgb[va_i] = m.predict_proba(Xf[va_i])[:, 1]

        m = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbose=-1)
        m.fit(Xf[tr_i], y[tr_i], eval_set=[(Xf[va_i], y[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[va_i] = m.predict_proba(Xf[va_i])[:, 1]

        m = cb.CatBoostClassifier(**bp_cb, random_seed=42, verbose=0, early_stopping_rounds=50)
        m.fit(Xf[tr_i], y[tr_i], eval_set=(Xf[va_i], y[va_i]))
        oof_cb[va_i] = m.predict_proba(Xf[va_i])[:, 1]

    valid = ~np.isnan(oof_xgb)
    y_v = y[valid]
    ox, ol, oc = oof_xgb[valid], oof_lgb[valid], oof_cb[valid]

    log(f"  OOF Brier — XGB: {brier_score(y_v, ox):.5f}, LGB: {brier_score(y_v, ol):.5f}, CB: {brier_score(y_v, oc):.5f}")

    # --- Optimize ensemble weights ---
    best_w, best_b = None, 1.0
    for w1 in np.arange(0.05, 0.9, 0.05):
        for w2 in np.arange(0.05, 0.9 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            b = brier_score(y_v, w1 * ox + w2 * ol + w3 * oc)
            if b < best_b:
                best_b = b
                best_w = (w1, w2, w3)

    weights = np.array(best_w)
    log(f"  Weights — XGB: {weights[0]:.2f}, LGB: {weights[1]:.2f}, CB: {weights[2]:.2f}")

    oof_ens = weights[0] * ox + weights[1] * ol + weights[2] * oc
    log(f"  Ensemble Brier (pre-cal): {brier_score(y_v, oof_ens):.5f}")

    # --- Platt scaling ---
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(oof_ens, y_v)
    oof_iso = iso.predict(oof_ens)

    lr = LogisticRegression(C=1.0)
    lr.fit(oof_ens.reshape(-1, 1), y_v)
    oof_lr = lr.predict_proba(oof_ens.reshape(-1, 1))[:, 1]

    b_iso = brier_score(y_v, oof_iso)
    b_lr = brier_score(y_v, oof_lr)
    log(f"  Calibrated — Isotonic: {b_iso:.5f}, Logistic: {b_lr:.5f}")

    if b_iso < b_lr and b_iso < best_b:
        calibrator = ("isotonic", iso)
        log(f"  Using isotonic. Final: {b_iso:.5f}")
    elif b_lr < best_b:
        calibrator = ("logistic", lr)
        log(f"  Using logistic. Final: {b_lr:.5f}")
    else:
        calibrator = None
        log(f"  No calibration helped. Final: {best_b:.5f}")

    # --- Train on ALL data ---
    log("  Training final models on all data...")
    final_xgb = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42)
    final_xgb.fit(Xf, y, verbose=False)

    final_lgb = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbose=-1)
    final_lgb.fit(Xf, y)

    final_cb = cb.CatBoostClassifier(**bp_cb, random_seed=42, verbose=0)
    final_cb.fit(Xf, y)

    return {
        "final_xgb": final_xgb, "final_lgb": final_lgb, "final_cb": final_cb,
        "weights": weights, "feature_cols": feature_cols, "calibrator": calibrator,
    }


# ===========================================================================
# 15. Predict
# ===========================================================================
def predict_ensemble(X, bundle):
    fc = bundle["feature_cols"]
    w = bundle["weights"]
    Xf = X[fc].fillna(0).values

    p_x = bundle["final_xgb"].predict_proba(Xf)[:, 1]
    p_l = bundle["final_lgb"].predict_proba(Xf)[:, 1]
    p_c = bundle["final_cb"].predict_proba(Xf)[:, 1]
    pred = w[0] * p_x + w[1] * p_l + w[2] * p_c

    cal = bundle["calibrator"]
    if cal:
        if cal[0] == "isotonic":
            pred = cal[1].predict(pred)
        else:
            pred = cal[1].predict_proba(pred.reshape(-1, 1))[:, 1]

    return np.clip(pred, 0.01, 0.99)


# ===========================================================================
# 16. Generate submission
# ===========================================================================
def generate_submission(data, ts_m, ts_w, sd_m, sd_w, coach, massey, cf_m, cf_w,
                        knn_m, knn_w, elo_m, elo_w, h2h_m, h2h_w, mod_m, mod_w):
    log("Generating submission...")
    sub_key = "SampleSubmissionStage2" if "SampleSubmissionStage2" in data else "SampleSubmissionStage1"
    sample = data[sub_key].copy()

    men_rows, women_rows = [], []
    men_idx, women_idx = [], []

    for i, row in sample.iterrows():
        parts = row["ID"].split("_")
        season, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])
        is_men = t1 < 3000

        if is_men:
            f = build_matchup_features(t1, t2, season, ts_m, sd_m, coach, massey, cf_m, knn_m, elo_m, h2h_m, True)
            men_rows.append(f if f else {})
            men_idx.append(i)
        else:
            f = build_matchup_features(t1, t2, season, ts_w, sd_w, None, None, cf_w, knn_w, elo_w, h2h_w, False)
            women_rows.append(f if f else {})
            women_idx.append(i)

    result = sample.copy()

    if men_rows:
        X_men = pd.DataFrame(men_rows)
        for col in mod_m["feature_cols"]:
            if col not in X_men.columns:
                X_men[col] = 0
        result.loc[men_idx, "Pred"] = predict_ensemble(X_men, mod_m)

    if women_rows:
        X_w = pd.DataFrame(women_rows)
        for col in mod_w["feature_cols"]:
            if col not in X_w.columns:
                X_w[col] = 0
        result.loc[women_idx, "Pred"] = predict_ensemble(X_w, mod_w)

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    data = load_data()

    for prefix, label in [("M", "MEN'S"), ("W", "WOMEN'S")]:
        log(f"\n{'='*60}\n{label} MODEL\n{'='*60}")

        is_men = prefix == "M"
        all_games = pd.concat([
            data[f"{prefix}RegularSeasonCompactResults"],
            data[f"{prefix}NCAATourneyCompactResults"],
        ]).sort_values(["Season", "DayNum"])

        elo = build_elo_ratings(all_games)
        ts = build_team_season_stats(data[f"{prefix}RegularSeasonDetailedResults"])

        sd = build_seed_features(data[f"{prefix}NCAATourneySeeds"])
        coach = build_coach_features(data["MTeamCoaches"], data["MNCAATourneyCompactResults"]) if is_men else None
        massey = build_massey_features(data["MMasseyOrdinals"]) if is_men else None
        cf = build_conference_strength(data[f"{prefix}TeamConferences"], ts)
        knn = KNNFeatureLookup(ts, data[f"{prefix}RegularSeasonDetailedResults"])
        h2h = build_h2h_features(all_games)

        X, y, seasons = build_training_data(prefix, data, ts, sd, coach, massey, cf, knn, elo, h2h)
        all_cols = list(X.columns)

        log("Feature selection...")
        feat_cols = select_features(X, y, all_cols)

        log("Optuna + temporal CV + calibration...")
        models = optimize_and_train(X, y, seasons, feat_cols, n_trials=30)

        # Store for submission
        if is_men:
            ts_m, sd_m, coach_m, massey_m, cf_m, knn_m, elo_m, h2h_m, mod_m = ts, sd, coach, massey, cf, knn, elo, h2h, models
        else:
            ts_w, sd_w, cf_w, knn_w, elo_w, h2h_w, mod_w = ts, sd, cf, knn, elo, h2h, models

    # Submission
    sub = generate_submission(data, ts_m, ts_w, sd_m, sd_w, coach_m, massey_m,
                              cf_m, cf_w, knn_m, knn_w, elo_m, elo_w,
                              h2h_m, h2h_w, mod_m, mod_w)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
    try:
        sub.to_csv(out, index=False)
    except PermissionError:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v2.csv")
        sub.to_csv(out, index=False)
    log(f"\nSubmission saved to {out}")
    log(f"  Shape: {sub.shape}")
    log(f"  Pred range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    log(f"  Pred mean: {sub['Pred'].mean():.4f}")


if __name__ == "__main__":
    main()
