# March Machine Learning Mania 2026

Predicting NCAA Men's and Women's Basketball Tournament outcomes for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition. The goal is to predict the probability that the lower-seeded team wins each possible tournament matchup, evaluated by **Brier score** (lower is better).

## Results

| Model | Men's Brier | Women's Brier |
|-------|-------------|---------------|
| Baseline (coin flip) | 0.250 | 0.250 |
| Seed-only model | ~0.200 | ~0.180 |
| **This model (v2)** | **0.134** | **0.097** |

## Architecture

An ensemble of three gradient boosting frameworks, Optuna-tuned and calibrated:

```
                  +------------------+
                  |   Raw Game Data  |
                  |  (35 CSV files)  |
                  +--------+---------+
                           |
            +--------------+--------------+
            |              |              |
     +------v------+ +----v----+ +-------v--------+
     | Elo Ratings  | | Season  | | KNN Similar    |
     | (custom,     | | Stats   | | Opponents      |
     | MOV-scaled,  | | (time-  | | (5 neighbors,  |
     | season decay)| | weighted| | per-season)    |
     +------+------+ | exponen-| +-------+--------+
            |         | tial    |         |
            |         | decay)  |         |
            |         +----+----+         |
            |              |              |
     +------v------+ +----v--------+ +---v-----------+
     | Coach Tenure | | Massey      | | Conference    |
     | & Tournament | | Ordinals    | | Strength      |
     | Success      | | (composite) | | (avg WinPct)  |
     +------+------+ +----+--------+ +---+-----------+
            |              |              |
            +--------------+--------------+
                           |
                  +--------v---------+
                  | 156 Features (M) |
                  | 127 Features (W) |
                  +--------+---------+
                           |
              +------------+------------+
              |            |            |
       +------v---+  +----v-----+  +---v-------+
       | XGBoost  |  | LightGBM |  | CatBoost  |
       | (Optuna  |  | (Optuna  |  | (Optuna   |
       |  tuned)  |  |  tuned)  |  |  tuned)   |
       +------+---+  +----+-----+  +---+-------+
              |            |            |
              +-----+------+------+-----+
                    |             |
           +--------v---------+  |
           | Weighted Ensemble |  |
           | (grid-searched   |  |
           |  weights)        |  |
           +--------+---------+  |
                    |             |
           +--------v---------+  |
           | Isotonic          |  |
           | Calibration       |  |
           | (Platt scaling)   |  |
           +--------+---------+
                    |
           +--------v---------+
           |  Final Predictions |
           |  (clipped 0.01-0.99)|
           +--------------------+
```

## Feature Engineering

### 1. Elo Ratings (Custom Implementation)
- Built from scratch on all historical games (regular season + tournament)
- **Margin-of-victory scaling**: K-factor scaled by `log(MOV + 1)` so blowouts update ratings more than close games
- **Home court advantage**: +100 Elo points for home team
- **Season carryover decay**: 75% carryover between seasons (regress 25% toward 1500 mean)
- **Elo difference dampener**: Prevents runaway ratings when a strong team beats a weak one

### 2. Time-Weighted Season Statistics
All per-game statistics are aggregated with **exponential decay weighting** — games later in the season count more heavily than early-season games. This captures team improvement/regression throughout the year.

**Box score stats** (all weighted averages):
- Field goals (made, attempted, percentage)
- 3-point field goals (made, attempted, percentage)
- Free throws (made, attempted, percentage)
- Offensive and defensive rebounds
- Assists, turnovers, steals, blocks, personal fouls

**Derived advanced metrics**:
- **Effective FG%**: `(FGM + 0.5 * FGM3) / FGA` — weights threes at 1.5x
- **Offensive Efficiency**: Points per 100 possessions (Dean Oliver's formula)
- **Defensive Efficiency**: Opponent points per 100 possessions
- **Net Efficiency**: Offensive - Defensive efficiency (best single predictor of team quality)
- **Assist-to-Turnover Ratio**: Ball security and passing quality
- **Rebound Margin**: Total rebounds vs opponent rebounds
- **Turnover Margin**: Forced turnovers minus own turnovers
- **Last 10 Games**: Win percentage and average margin (momentum/form)

### 3. KNN Similar-Opponent Features
For each tournament matchup (Team A vs Team B), we ask: *"How did Team A perform against regular-season opponents that statistically resemble Team B?"*

- For each season, all teams are profiled using 12 key stats (win%, scoring, shooting, rebounds, assists, turnovers, steals, blocks, margin)
- Profiles are standardized and a KNN model (k=5) finds the 5 most similar teams to any given team
- For a matchup A vs B: look up A's record against opponents similar to B (win rate, average margin, number of games)
- This provides a "proxy head-to-head" when teams haven't actually played each other

### 4. Coach Features (Men's Only)
- **Coach Tenure**: Consecutive seasons the current coach has been with the team
- **Prior Tournament Wins**: Coach's total NCAA tournament wins before the current season (across all teams coached)
- **Prior Tournament Games**: Total tournament games coached
- **Prior Tournament Win Rate**: Historical tournament win rate

### 5. Massey Ordinal Rankings
- Aggregates end-of-regular-season rankings from 60+ computer ranking systems (Sagarin, KenPom, BPI, etc.)
- Features: mean rank, median rank, min rank, max rank, standard deviation
- Uses last available ranking day before tournament (day <= 133)

### 6. Conference Strength
- **Average Win Percentage**: Mean win% of all teams in the conference
- **Median Win Percentage**: Median win% (robust to outliers)

### 7. Head-to-Head History
- All-time win rate between the two teams (prior seasons only — no leakage)
- Number of historical matchups

### 8. Feature Construction for Matchups
For each feature, three versions are created:
- `T1_{feature}` — Team 1's value (lower TeamID)
- `T2_{feature}` — Team 2's value (higher TeamID)
- `Diff_{feature}` — Team 1 minus Team 2 (the difference)

Plus interaction features:
- `Elo_WinProb` — Expected win probability from Elo alone
- `SeedMatchup` — Seed1 * Seed2 (captures seed interaction effects)

## Model Training Pipeline

### Temporal Cross-Validation
Unlike standard k-fold CV which leaks future data, we use **season-based temporal splits**:
- Train on seasons 2003-2010, validate on 2011
- Train on seasons 2003-2011, validate on 2012
- ... and so on through 2025
- This gives 14 folds for men's, 7 for women's

### Optuna Hyperparameter Tuning
Each of the three models is independently tuned with 30 Optuna trials, optimizing Brier score across all temporal CV folds.

**Tuned parameters** include:
- `n_estimators` / `iterations` (100-600)
- `max_depth` / `depth` (3-7)
- `learning_rate` (0.01-0.12, log scale)
- `subsample` (0.6-1.0)
- `colsample_bytree` (0.5-1.0)
- `reg_alpha` and `reg_lambda` (1e-3 to 10, log scale)
- Model-specific: `min_child_weight`, `num_leaves`, `l2_leaf_reg`, `bagging_temperature`

### Feature Selection
A quick XGBoost model is trained on all features, and any feature with importance < 0.001 is dropped. This removes noise features that hurt generalization.

### Ensemble Weighting
Ensemble weights are optimized via exhaustive grid search (5% increments) to minimize Brier score on the temporal CV out-of-fold predictions.

Typical weights:
- **Men's**: XGB ~60%, LGB ~15%, CB ~25%
- **Women's**: XGB ~15%, LGB ~10%, CB ~75%

### Probability Calibration (Platt Scaling)
After ensembling, predictions are calibrated using both:
- **Isotonic Regression**: Non-parametric monotonic mapping
- **Logistic Regression**: Parametric sigmoid mapping

The method that produces the best Brier score on OOF predictions is selected. Isotonic calibration typically wins, providing an additional ~0.005 Brier improvement.

## Project Structure

```
march-ml-mania-2026/
├── README.md               # This file
├── requirements.txt         # Python dependencies
├── .gitignore              # Excludes data/, submissions, caches
├── download_data.py        # Downloads all 35 CSVs from Kaggle API
├── load_data.py            # Utility to load and inspect datasets
└── train_model.py          # Full pipeline: features -> training -> submission
```

## Data

35 CSV files from Kaggle covering 1985-2026 seasons (men's) and 1998-2026 (women's):

| Category | Files | Description |
|----------|-------|-------------|
| Game Results | `M/WRegularSeasonDetailedResults`, `M/WNCAATourneyDetailedResults` | Full box scores for every game |
| Seeds | `M/WNCAATourneySeeds` | Tournament seedings by year |
| Teams | `M/WTeams`, `M/WTeamConferences` | Team IDs, names, conference affiliations |
| Coaches | `MTeamCoaches` | Coach assignments per team per season |
| Rankings | `MMasseyOrdinals` | 60+ computer ranking systems (~5.8M rows) |
| Geography | `Cities`, `M/WGameCities` | Game locations |
| Submissions | `SampleSubmissionStage1/2` | Required prediction format |

Data is downloaded via `download_data.py` and stored in `data/` (gitignored).

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download competition data
```bash
export KAGGLE_TOKEN=KGAT_your_token_here
python download_data.py
```

### 3. Train model and generate submission
```bash
python train_model.py
```

This will:
1. Load all 35 datasets
2. Build Elo ratings, time-weighted stats, KNN features, coach features, Massey rankings
3. Run Optuna hyperparameter tuning (30 trials x 3 models x temporal CV)
4. Train the final ensemble on all data
5. Generate `submission.csv` ready for Kaggle upload

**Runtime**: ~20-30 minutes on a modern CPU.

### 4. Inspect data (optional)
```bash
python load_data.py
```

## Key Design Decisions

1. **Separate men's and women's models**: The datasets have different characteristics (different number of teams, seasons available, and no coach data for women's), so independent models perform better than a unified one.

2. **Elo as a foundation**: Custom Elo with MOV scaling provides a strong baseline (~0.14 Brier alone for men's) that captures team quality through the lens of game outcomes rather than box score stats.

3. **Temporal validation over k-fold**: Standard k-fold would train on 2020 data and validate on 2015, leaking future information. Temporal CV gives honest estimates of out-of-sample performance.

4. **Isotonic over logistic calibration**: Tournament game probabilities have non-linear miscalibration patterns (models tend to be overconfident in the 0.6-0.8 range). Isotonic regression captures these patterns better than a simple sigmoid.

5. **Conservative prediction clipping (0.01-0.99)**: Brier score heavily penalizes confident wrong predictions. Clipping prevents catastrophic losses from extreme probabilities.

## License

MIT
