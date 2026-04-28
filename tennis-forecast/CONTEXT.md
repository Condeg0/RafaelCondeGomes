# BreakPoint AI — Full Technical Context

> **Purpose:** This document is a complete, self-contained reference for the BreakPoint AI project. It is intended to be fed to an LLM so that the LLM can answer any question about the project without needing access to the source code.

---

## 1. Project Overview

**BreakPoint AI** is a production-grade machine learning system for **ATP tennis match outcome prediction**. The core innovation is a hybrid architecture that combines:

- A **Siamese LSTM** (deep learning) model that processes each player's recent match history as a time series
- **Tabular ensemble models** (XGBoost, Random Forest, Logistic Regression) that consume hand-engineered contextual features
- A **stacking meta-learner** (XGBoost) that combines all base model outputs into a final probability

The system achieves **0.7168 Test AUC** and **65.35% accuracy** on 2024 ATP matches (unseen test data). It implements strict temporal validation to eliminate look-ahead bias — a critical requirement when working with sequential sports data.

---

## 2. Repository Structure

```
BreakPoint-AI/
├── configs/
│   └── config.yaml                  # Master config (hyperparams, paths, feature specs)
├── src/
│   ├── config.py                    # Pydantic-based config schema validation
│   ├── data.py                      # Dataset loading, splitting, preprocessing
│   ├── features.py                  # FeatureEngineer: rolling stats, H2H, streaks
│   ├── training.py                  # Trainer: orchestrates model training loops
│   ├── tuning.py                    # Tuner: Optuna hyperparameter optimization
│   ├── inference.py                 # MetaLearnerPipeline: batch inference engine
│   ├── evaluation.py                # Evaluator: metrics, SHAP, calibration curves
│   ├── logger.py                    # Production logging (file + console handlers)
│   └── models/
│       ├── nn.py                    # SiameseLSTM architecture
│       ├── xgb.py                   # XGBoostModel wrapper
│       ├── baselines.py             # RandomForestBaseline, LogisticBaseline
│       └── stacking.py              # StackingMetaLearner ensemble combiner
├── main.py                          # Training orchestrator (Phases 1–5)
├── cli_batch_predict.py             # Inference CLI for upcoming matches
├── data/
│   ├── raw/                         # ATP match CSVs (1990–2024, 35 files)
│   └── processed/                   # Engineered feature matrices (train/val/test)
├── model_prod/                      # Production frozen model weights
│   ├── global_preprocessor.pkl
│   ├── lstm/
│   │   ├── best_model.pt
│   │   └── global_preprocessor.pkl
│   └── stacking/
│       ├── best_model.pt
│       ├── meta_learner.joblib
│       └── global_preprocessor.pkl
├── assets/                          # Evaluation visualizations (10 PNG files)
├── outputs/                         # Artifact directories from training runs
├── logs/                            # pipeline.log (persistent execution logs)
├── requirements.txt
├── .github/workflows/ci.yml         # GitHub Actions: ruff linting
└── README.md
```

---

## 3. Dataset & Data Pipeline

### 3.1 Data Source

- **Source:** ATP match database
- **Coverage:** 1990–2024 (35 raw CSV files, one per year)
- **Excluded:** Davis Cup, Laver Cup matches (filtered out during loading)

### 3.2 Raw Data Schema

Each CSV row represents one match with the following columns:

| Column | Description |
|---|---|
| `tourney_id`, `tourney_name` | Tournament identifiers |
| `tourney_date` | Match date |
| `match_num` | Match number within tournament |
| `round` | Tournament round (R128, R64, …, F) |
| `winner_name`, `loser_name` | Player names |
| `winner_rank`, `loser_rank` | ATP ranking at time of match |
| `winner_rank_points`, `loser_rank_points` | Ranking points |
| `surface` | Court surface: Clay / Grass / Hard / Carpet |
| `tourney_level` | Tournament level: G (Grand Slam), M (Masters), A (ATP 250/500), etc. |
| `best_of` | Best of 3 or 5 sets |
| `minutes` | Match duration |
| `w_ace`, `l_ace` | Aces for winner/loser |
| `w_df`, `l_df` | Double faults |
| `w_svpt`, `l_svpt` | Serve points played |
| `w_1stIn`, `l_1stIn` | First serves in |
| `w_1stWon`, `l_1stWon` | Points won on first serve |
| `w_2ndWon`, `l_2ndWon` | Points won on second serve |
| `w_bpSaved`, `l_bpSaved` | Break points saved |
| `w_bpFaced`, `l_bpFaced` | Break points faced |

### 3.3 Temporal Splits

| Split | Date Range | Purpose |
|---|---|---|
| **Train** | ≤ 2022-12-31 | Model fitting |
| **Validation** | 2023-01-01 – 2023-12-31 | Hyperparameter tuning, early stopping |
| **Test** | ≥ 2024-01-01 | Final unseen evaluation |

These splits are strictly temporal — no future data ever leaks into training.

### 3.4 Data Processing Steps (`src/data.py`)

1. Load all 35 CSV files and concatenate
2. Parse `tourney_date` to datetime, sort by date + `match_num`
3. Filter out Davis Cup and Laver Cup
4. Apply temporal split into train/val/test DataFrames
5. **Preprocessor** (`Preprocessor` class):
   - `StandardScaler` for all numeric features (fit on train only, applied to val/test)
   - `OneHotEncoder` for categorical features (`surface`, `tourney_level`)
   - Combined via `ColumnTransformer`
   - Saved to `global_preprocessor.pkl` for inference consistency

### 3.5 PyTorch Dataset (`TennisDataset`)

Two operating modes:

- **`"tabular"` mode:** Returns `(context_matrix[idx], label)` — used by XGBoost, RF, LogReg
- **`"lstm"` mode:** Returns `(seq_a, seq_b, context, label)` — used by Siamese LSTM
  - `seq_a` / `seq_b`: 10-match rolling window sequences per player (shape `(10, 9)`)
  - Sequences built with `_get_sequence()` enforcing `candidate_dates < current_date` (strict past-only)
  - Pads with zeros if fewer than 10 prior matches exist

---

## 4. Feature Engineering (`src/features.py`)

The `FeatureEngineer` class transforms raw match-level data into a rich feature matrix. All features are computed with **temporal integrity** — no future information is used.

### 4.1 Long-Format Transformation

Raw matches (1 row = 1 match) are first expanded to player-level records (2 rows = 1 match):
- One row for the winner (`label = 1`)
- One row for the loser (`label = 0`)
- Preserves `surface`, `tourney_level`, `tourney_date`, all serve stats

### 4.2 Rolling Statistics (10-match window)

All rolling features use `shift(1).rolling(10).mean()` — the shift ensures the current match is never included.

| Feature | Description |
|---|---|
| `ace_roll` | Rolling mean aces over last 10 matches |
| `df_roll` | Rolling mean double faults |
| `svpt_roll` | Rolling mean serve points played |
| `1stIn_roll` | Rolling mean 1st serves in |
| `1stWon_roll` | Rolling mean 1st serve points won |
| `2ndWon_roll` | Rolling mean 2nd serve points won |
| `1stIn_pct_roll` | Rolling 1st serve in percentage |
| `win_pct_roll` | Rolling win rate (recent form) |
| `{feature}_lag` | Previous match value (lag-1) for each feature |

### 4.3 Surface-Specific Rolling Win Rate

```python
surf_win_pct_roll = groupby(['player', 'surface'])['win_pct'].shift(1).rolling(10).mean()
```

Captures each player's specialization on Clay, Grass, or Hard courts separately.

### 4.4 Serve Efficiency Ratios

Derived from match-level stats, then rolled:

| Feature | Formula |
|---|---|
| `bp_save_rate` | `bpSaved / bpFaced` — break point save percentage |
| `first_srv_win_pct` | `1stWon / 1stIn` — 1st serve win percentage |
| `second_srv_win_pct` | `2ndWon / (svpt - 1stIn)` — 2nd serve win percentage |

Each is rolled over 10 matches with the same `shift(1)` leakage guard.

### 4.5 Win/Loss Streak

A signed streak counter:
- `+1, +2, +3, …` for consecutive wins
- `-1, -2, -3, …` for consecutive losses
- Resets to ±1 on alternation

Computed with a custom loop over `groupby('player')['label'].shift(1)`.

### 4.6 Rank Momentum

```python
rank_trend = current_rank - rolling_mean_rank(5)
```

Positive = rank worsening (climbing number = losing ground), Negative = improving.

### 4.7 Head-to-Head (H2H) Features

| Feature | Formula |
|---|---|
| `h2h_win_rate` | `cumsum(wins) / count(matches)` for each (player, opponent) pair |
| `surf_h2h_win_rate` | Same, but grouped by (player, opponent, surface) |

Both use `shift(1).cumsum()` to prevent future leakage. Default to **0.5** if no prior history exists.

### 4.8 Days Since Last Match

```python
days_since = groupby('player')['tourney_date'].diff().dt.days
```

Filled with **365** for a player's first match (signals a long rest or new player).

### 4.9 Log-Transformed Rank Points Difference

```python
rank_pts_diff = log(player_rank_points + 1) - log(opponent_rank_points + 1)
```

Log transformation captures the non-linear relationship between ranking points and match outcomes. This is the **strongest single feature** (SHAP |correlation| ≈ 0.338).

### 4.10 Pivot Back to Match Format

The long-format player rows are pivoted back to single-row matches:
- Player A features vs. opponent features are merged
- **Difference features** computed: `{feature}_diff = player_value - opponent_value`
- Final output: single row per match with both absolute and differential features

### 4.11 Final Feature Sets

**Context features (23 total) fed to all models:**
```
rank_diff, rank, opponent_rank
ace_roll_diff, df_roll_diff, win_pct_roll_diff
h2h_win_rate
days_since
rank_pts_diff
surf_win_pct_roll_diff
bp_save_rate_roll_diff
first_srv_win_pct_roll_diff
streak_diff, rank_trend_diff
surf_h2h_win_rate_diff
surf_win_pct_roll, streak, rank_trend   (absolute values)
```

**Sequence features (9 total) fed to LSTM per player per timestep:**
```
ace, df, svpt, 1stIn, 1stWon, 2ndWon
rank, bp_save_rate, first_srv_win_pct
```

---

## 5. Model Architectures & Hyperparameters

### 5.1 Siamese LSTM (`src/models/nn.py`)

**Architecture (forward pass):**

```
Input:
  seq_a:    (batch_size, 10, 9)   — Player A: 10 past matches × 9 features
  seq_b:    (batch_size, 10, 9)   — Player B: 10 past matches × 9 features
  context:  (batch_size, 23)      — Context features

Twin LSTM Encoders (shared weights — same network for both players):
  LSTM(input_size=9, hidden_size=64, num_layers=2, dropout=0.25, batch_first=True)
  Output: (batch_size, 10, 64) hidden states for each timestep

Attention (per player):
  scores  = Linear(64 → 1)(lstm_out)           # (batch, 10, 1)
  weights = softmax(scores, dim=1)              # (batch, 10, 1)
  emb     = sum(weights * lstm_out, dim=1)      # (batch, 64)

Concatenation:
  fused = concat([emb_a, emb_b, context])       # (batch, 64+64+23) = (batch, 151)

Fusion MLP:
  Dense(151 → 64) + ReLU + Dropout(0.25)
  Dense(64  → 32) + ReLU + Dropout(0.25)
  Dense(32  →  1)                               # Raw logit

Output: sigmoid(logit) → win probability ∈ (0, 1)
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Sequence length | 10 matches |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.25 |
| Fusion dim | 64 |
| Batch size | 64 |
| Epochs | 75 (with early stopping) |
| Learning rate | 0.0002 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping patience | 5 epochs |
| Gradient clipping | max_norm=1.0 |

**Why Siamese?** Both players go through the exact same network — shared weights force the model to learn universal momentum patterns rather than player-specific features, which improves generalization and reduces parameters.

**Why Attention?** Learned soft-attention over 10 timesteps allows the model to weight recent matches more heavily without a hard-coded decay function. More flexible than simple last-timestep selection.

### 5.2 XGBoost (`src/models/xgb.py`)

| Parameter | Value |
|---|---|
| `n_estimators` | 400 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `min_child_weight` | 3 |
| `reg_alpha` (L1) | 0.05 |
| `reg_lambda` (L2) | 1.5 |
| Objective | `binary:logistic` |
| Eval metric | AUC |
| Early stopping | 20 rounds on validation AUC |
| Input | 23 context features only |

### 5.3 Random Forest (`src/models/baselines.py`)

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `min_samples_split` | 2 |
| Post-processing | `CalibratedClassifierCV(method='sigmoid', cv=3)` |

Platt scaling (sigmoid calibration) post-processes probability outputs to improve reliability.

### 5.4 Logistic Regression (`src/models/baselines.py`)

| Parameter | Value |
|---|---|
| `penalty` | L2 |
| `C` | 1.0 |
| `max_iter` | 1000 |
| `solver` | lbfgs |

Serves as a linear baseline; useful for understanding which features have linear separability.

### 5.5 Stacking Meta-Learner (`src/models/stacking.py`)

**Meta-learner:** XGBoost with `n_estimators=100`, `max_depth=2`, `learning_rate=0.05`

**Training procedure:**
1. Base models (LSTM, XGBoost, RF, LogReg) generate **out-of-fold (OOF)** predictions on the validation set
2. Meta-learner trains on these 4 OOF columns → label
3. At test time: stack base model test predictions → feed to meta-learner

This prevents data leakage because the meta-learner never sees raw training labels during its own fitting.

---

## 6. Hyperparameter Tuning (`src/tuning.py`)

**Framework:** Optuna

**Enabled only for LSTM** (`lstm.training.tuning_enabled: true`); XGBoost and RF use fixed hand-tuned values.

**Search space:**

| Parameter | Search Type | Range |
|---|---|---|
| `hidden_size` | Integer (step=16) | 32 – 128 |
| `num_layers` | Integer | 1 – 3 |
| `dropout` | Float (continuous) | 0.2 – 0.5 |
| `learning_rate` | Float (log scale) | 5e-5 – 3e-3 |
| `weight_decay` | Float (log scale) | 1e-5 – 1e-2 |
| `fusion_dim` | Categorical | {64, 128, 256} |
| `batch_size` | Categorical | {64, 128, 256} |

**Tuning settings:**
- Direction: maximize AUC
- Training epochs per trial: 15 (vs. 75 for production)
- Pruner: `MedianPruner` (cuts unpromising trials early)
- Number of trials: 20 (configured in `main.py`)

Best hyperparameters from tuning patch the config before the full production training run.

---

## 7. Training Pipeline (`main.py`)

Training is a 5-phase sequential pipeline:

### Phase 1: Data Ingestion
- Load 35 raw ATP CSVs from `data/raw/`
- Filter Davis Cup / Laver Cup
- Produce: `train_raw`, `val_raw`, `test_raw` DataFrames

### Phase 2: Feature Engineering
- Concatenate all splits (to allow rolling stats to span boundaries)
- Sort by date
- Run `FeatureEngineer.generate_features()`
- Re-split by date back into train/val/test

### Phase 3: Preprocessing
```python
if mode == "stacking":
    # Load frozen preprocessor from base artifact directory
else:
    # Fit StandardScaler + OneHotEncoder on training set ONLY
    preprocessor.fit(train_df)
    preprocessor.save("global_preprocessor.pkl")
```

### Phase 4: Model Training

For each model in `config.pipeline.models_to_train`:

1. **LSTM** (if enabled):
   - Optional Optuna tuning (20 trials, 15 epochs each)
   - Full training: 75 epochs, Adam, BCEWithLogitsLoss, EarlyStopping(patience=5)
   - Save `best_model.pt` on best validation AUC

2. **XGBoost:**
   - `fit(train_ds, eval_set=[(val_ds, "val")], early_stopping_rounds=20)`
   - Save `model.joblib`

3. **Random Forest:**
   - `fit(train_ds)` + `CalibratedClassifierCV`
   - Save `model.joblib`

4. **Logistic Regression:**
   - `fit(train_ds)`
   - Save `model.joblib`

5. **Stacking** (separate run, requires base models already trained):
   - Load base model artifacts from `stacking_base_artifact_dir`
   - Generate OOF predictions on validation set
   - Fit XGBoost meta-learner on OOF stack
   - Save `meta_learner.joblib`

Each model saves `hyperparameters.json` alongside its weights.

### Phase 5: Evaluation

For each trained model:
- Generate predictions on test set
- Compute: AUC, Accuracy, Precision, Recall, F1
- Generate and save:
  - Confusion matrix (`confusion_matrix_{model}.png`)
  - ROC curve (`roc_curve_{model}.png`)
  - Calibration curve (`calibration_curve_{model}.png`)
  - Metrics bar chart (`metrics_summary_{model}.png`)
  - SHAP summary plot (`shap_summary_{model}.png`)
- Write `metrics.json` and `report.txt`

---

## 8. Evaluation Results

### 8.1 Test Set Performance (2024 ATP Matches)

| Model | Test AUC | Accuracy | Notes |
|---|---|---|---|
| **Stacking Ensemble** | **0.7168** | **65.35%** | Meta-learner over all 4 base models |
| **XGBoost** | **0.7185** | **65.23%** | Strongest individual model |
| **Siamese LSTM** | **0.7140** | **65.50%** | Best accuracy, third-best AUC |
| **Random Forest** | **0.7112** | **65.00%** | Sigmoid-calibrated |
| **Logistic Regression** | **0.7086** | **65.05%** | Linear baseline |

### 8.2 Metrics Computed per Model

- **AUC** (primary metric): Area Under the ROC Curve
- **Accuracy**: % correct binary classifications at threshold 0.5
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: 2 × (Precision × Recall) / (Precision + Recall)

### 8.3 Feature Importance (SHAP Values)

| Feature | SHAP Importance | Description |
|---|---|---|
| `rank_pts_diff` | 0.338 (strongest) | Log-transformed ranking points difference |
| `rank_diff` | 0.228 | Ordinal rank difference |
| `surf_win_pct_roll_diff` | High | Surface specialization rolling win % |
| `first_srv_win_pct_roll_diff` | High | 1st serve efficiency difference |

**SHAP method per model:**
- XGBoost / RF: `TreeExplainer` (fast, exact)
- Logistic Regression: `LinearExplainer`
- LSTM: `KernelExplainer` (slow — uses 20-sample approximation)

### 8.4 Context: ATP Prediction Ceiling

Academic literature suggests a maximum of ~0.75–0.77 AUC for pure statistical ATP prediction without external signals (injury, weather, player psychology). The 0.7168 AUC achieved here is competitive with published benchmarks.

---

## 9. Inference Pipeline (`src/inference.py`, `cli_batch_predict.py`)

### 9.1 Batch Inference CLI

```bash
python cli_batch_predict.py --config configs/config.yaml --model stacking
```

**Workflow:**

1. Load last 2 years of ATP match history
2. Load upcoming matches CSV
3. Tag upcoming rows with `is_inference=True`
4. Concatenate history + upcoming, run full feature engineering pipeline
   - Feature values for upcoming matches computed only from historical data
5. Load frozen artifacts: `global_preprocessor.pkl` + model weights
6. Generate probabilities for each upcoming match
7. Output predictions CSV

### 9.2 Production Pipeline (`MetaLearnerPipeline`)

```python
from src.inference import MetaLearnerPipeline
from pathlib import Path

pipeline = MetaLearnerPipeline.load_frozen_model(Path("model_prod"))
results = pipeline.predict_batch(combined_df)
```

- Handles missing models gracefully (fills with 0.5 neutral probability)
- Outputs: `player_1_win_probability`, `player_2_win_probability`, `confidence_spread`

### 9.3 Inference Input Format

```
tourney_id, tourney_date, match_num, winner_name, loser_name,
winner_rank, loser_rank, winner_rank_points, loser_rank_points,
surface, tourney_level, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, ...
```

### 9.4 Inference Output Format

```
player_1, player_2, player_1_win_probability, player_2_win_probability, confidence_spread
Djokovic, Sinner, 0.6234, 0.3766, 0.2468
```

---

## 10. Configuration Schema (`configs/config.yaml` + `src/config.py`)

The master configuration file governs all aspects of the pipeline. Schema is validated with Pydantic:

```yaml
project:
  name: "breakpoint_ai"

data:
  paths:
    raw_dir: "data/raw/"
    processed_dir: "data/processed/"
    artifact_dir: "artifacts/"
  temporal_splits:
    train_cutoff: "2022-12-31"
    test_start: "2024-01-01"
  features:
    context: [23 features listed above]
    sequence: [9 features listed above]
    categorical: ["surface", "tourney_level"]
    target: "label"

pipeline:
  models_to_train: ["lstm", "xgboost", "random_forest", "logistic_regression"]
  # Or ["stacking"] for the meta-learner phase
  stacking_base_artifact_dir: "artifacts/20260403_141809"
  use_stacking: false
  run_evaluation: true
  inference_artifact_dir: "model_prod"
  inference_input_file: "data/inference/upcoming_matches.csv"
  inference_output_file: "data/inference/predictions.csv"

models:
  lstm:
    architecture:
      seq_len: 10
      hidden_size: 64
      num_layers: 2
      dropout: 0.25
      fusion_dim: 64
    training:
      batch_size: 64
      epochs: 75
      learning_rate: 0.0002
      tuning_enabled: true
  xgboost:
    hyperparameters:
      n_estimators: 400
      max_depth: 5
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.7
      min_child_weight: 3
      reg_alpha: 0.05
      reg_lambda: 1.5
    training:
      tuning_enabled: false
  random_forest:
    hyperparameters:
      n_estimators: 200
      max_depth: 8
      min_samples_split: 2
    training:
      tuning_enabled: false
  logistic_regression:
    hyperparameters:
      penalty: "l2"
      C: 1.0
      max_iter: 1000
    training:
      tuning_enabled: false
  stacking:
    meta_learner: "xgboost"
    cv_folds: 5
```

**Pydantic classes** defined in `src/config.py`:
`ProjectConfig`, `DataConfig`, `TemporalSplits`, `FeatureSets`, `LSTMConfig`, `XGBoostConfig`, `RandomForestConfig`, `LogRegConfig`, `StackingConfig`

---

## 11. Production Artifacts (`model_prod/`)

```
model_prod/
├── global_preprocessor.pkl          # Global StandardScaler + OneHotEncoder
├── lstm/
│   ├── best_model.pt                # Torch state dict (SiameseLSTM weights)
│   └── global_preprocessor.pkl     # LSTM-specific preprocessor copy
└── stacking/
    ├── best_model.pt                # Unused (kept for compatibility)
    ├── meta_learner.joblib          # XGBoost meta-learner
    └── global_preprocessor.pkl     # Stacking preprocessor copy
```

**Per-run artifacts** (saved under `artifacts/{run_id}/{model_name}/`):
- `hyperparameters.json` — model config used for this run
- `best_model.pt` — LSTM weights (or `model.joblib` for sklearn models)
- `metrics.json` — AUC, Accuracy, Precision, Recall, F1 on test set
- `report.txt` — human-readable summary
- `plots/` — all 5 evaluation visualizations

---

## 12. Leakage Prevention — Implementation Details

Look-ahead bias is the most critical failure mode in sports prediction. Three layers of protection:

**1. Rolling windows with shift:**
```python
rolling_stats = grouped['feature'].shift(1).rolling(10).mean()
# shift(1) excludes the current match from its own rolling feature
```

**2. Sequence building with strict date filter:**
```python
mask = candidate_dates < current_date   # strict "<" not "<="
past_indices = all_indices[mask]
```

**3. Cumulative H2H with shift:**
```python
h2h_wins = h2h_grp['win'].apply(lambda x: x.shift(1).cumsum())
# Only counts matches before the current one
```

**4. Preprocessor fit only on train:**
```python
preprocessor.fit(train_df)   # val and test are only .transform()'d
```

**5. Meta-learner uses OOF predictions:**
The stacking meta-learner trains on base model predictions for the validation set — never on training labels directly.

---

## 13. Logging & Monitoring (`src/logger.py`)

```python
def get_logger(name: str, artifact_dir: str = "logs") -> logging.Logger:
    # Routes to both:
    #   console (stdout)
    #   persistent file: logs/pipeline.log
    # Format: "YYYY-MM-DD HH:MM:SS | LEVEL | MODULE | MESSAGE"
```

`logs/pipeline.log` accumulates across runs (551+ lines from training history).

**Logged events:**
- Data ingestion: file counts, date ranges, row counts after filtering
- Feature engineering: input/output shapes at each step
- Preprocessing: feature normalization stats
- Training: per-epoch loss, AUC, LR schedule changes, early stopping triggers
- Evaluation: per-model metrics, visualization file paths

---

## 14. Dependencies & Environment

**Python:** 3.11+

**Key dependencies (`requirements.txt`):**

| Package | Version | Purpose |
|---|---|---|
| `numpy` | >=1.24 | Numerical computation |
| `pandas` | >=2.0 | Data manipulation |
| `scikit-learn` | >=1.3 | RF, LogReg, preprocessing, calibration |
| `torch` | >=2.1 | LSTM training (CUDA optional, auto-detected) |
| `xgboost` | latest | XGBoost models |
| `optuna` | latest | Hyperparameter tuning |
| `pydantic` | >=2.0 | Config schema validation |
| `pyyaml` | >=6.0 | YAML config parsing |
| `shap` | >=0.40.0 | Feature importance |
| `matplotlib` | >=3.7.0 | Plotting |
| `seaborn` | >=0.12.0 | Statistical visualizations |
| `joblib` | >=1.3 | Model serialization |
| `tqdm` | >=4.66 | Progress bars |

**Runtime:**
- CPU and CUDA both supported (auto-detected at training start)
- Tested on Linux (Arch Linux, Ubuntu)

---

## 15. CI/CD (`.github/workflows/ci.yml`)

```yaml
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check src/ main.py cli_batch_predict.py
```

Only linting is automated. No automated test suite (training is too expensive to run in CI).

---

## 16. Design Choices & Rationale

| Choice | Rationale |
|---|---|
| Siamese LSTM | Player histories are sequences; shared weights encode universal momentum patterns rather than player-specific quirks |
| 10-match window | Captures ~1–2 months of play; longer windows lose recency signal, shorter windows lack context |
| Log-transformed rank points | Ranking points have diminishing returns — log is more linearly predictive than ordinal rank |
| Surface-specific rolling win rate | Clay/grass/hard courts reward different playing styles; surface form differs from overall form |
| Attention over LSTM states | More flexible than last-timestep only; lets model learn which past matches matter most |
| Stacking ensemble | Combines complementary strengths: LSTM (sequence patterns) + XGBoost (tabular interactions) + RF (robustness) |
| Optuna tuning for LSTM only | LSTM has many more hyperparameters; XGBoost/RF are more robust to suboptimal settings |
| Calibration on RF | Random Forest probability outputs are poorly calibrated by default; Platt scaling corrects this |
| OOF meta-learner | Prevents the meta-learner from overfitting to base model training errors |
| CalibratedClassifierCV | In probabilistic forecasting, calibration matters as much as discrimination (AUC) |

---

## 17. Known Limitations

1. **Prediction ceiling:** Academic literature puts the statistical ATP prediction ceiling at ~0.75–0.77 AUC. Closing this gap requires non-statistical signals (injuries, fatigue, travel schedule, coaching changes).
2. **No injury data:** Model uses only match statistics — player health is not modeled.
3. **New player cold start:** Players with < 10 historical matches get zero-padded sequences; early predictions are unreliable.
4. **SHAP speed on LSTM:** `KernelExplainer` uses a 20-sample approximation — SHAP values are approximate, not exact.
5. **Stacking requires two training runs:** Base models must be fully trained first, then the meta-learner is trained separately using their validation OOF predictions.
6. **Static H2H:** H2H only counts raw win rate; does not weight recency or surface.

---

## 18. Reproducibility Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure 35 raw CSVs are present
ls data/raw/atp_matches_*.csv   # should list 1990-2024

# 3. Train base models (edit config if needed)
#    models_to_train: ["lstm", "xgboost", "random_forest", "logistic_regression"]
python main.py --config configs/config.yaml

# 4. Note the run ID printed at start (e.g. 20260403_141809)
#    Update config: stacking_base_artifact_dir: "artifacts/20260403_141809"
#    Set: models_to_train: ["stacking"]

# 5. Train stacking meta-learner
python main.py --config configs/config.yaml

# 6. Run batch inference on upcoming matches
python cli_batch_predict.py --config configs/config.yaml --model stacking
```

**Output artifacts per run:**
```
artifacts/{run_id}/
├── {model_name}/
│   ├── hyperparameters.json
│   ├── best_model.pt  (or model.joblib)
│   ├── metrics.json
│   ├── report.txt
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── calibration_curve.png
│       ├── metrics_summary.png
│       └── shap_summary.png
```

---

## 19. Evaluation Visualizations (`assets/`)

Pre-generated plots from a production training run:

| File | Description |
|---|---|
| `calibration_curve_lstm.png` | Reliability diagram: predicted probability vs. actual win rate |
| `calibration_curve_rf.png` | Same for Random Forest (post-Platt-scaling) |
| `confusion_matrix_lstm.png` | True/False positive breakdown at threshold 0.5 |
| `confusion_matrix_rf.png` | Same for Random Forest |
| `roc_curve_lstm.png` | AUC ~0.714 ROC curve |
| `roc_curve_rf.png` | AUC ~0.711 ROC curve |
| `metrics_summary_lstm.png` | Bar chart: AUC, Accuracy, Precision, Recall, F1 |
| `metrics_summary_rf.png` | Same for Random Forest |
| `shap_summary_lstm.png` | Feature importance via KernelExplainer (20-sample approx) |
| `shap_summary_rf.png` | Feature importance via TreeExplainer (exact) |

---

## 20. Glossary

| Term | Definition |
|---|---|
| AUC | Area Under ROC Curve — measures discrimination ability of a classifier |
| BCEWithLogitsLoss | Binary Cross-Entropy loss combining sigmoid + log-likelihood |
| Calibration | How well predicted probabilities match true empirical frequencies |
| Look-ahead bias | Using future data to predict past events — the cardinal sin of time-series ML |
| OOF | Out-of-Fold — predictions generated on held-out data during cross-validation |
| Platt scaling | Post-hoc probability calibration using logistic regression on model outputs |
| SHAP | SHapley Additive exPlanations — game-theoretic feature importance |
| Siamese network | Twin networks with shared weights, used to compare two inputs |
| Stacking | Ensemble method where a meta-learner trains on base model predictions |
| ReduceLROnPlateau | PyTorch LR scheduler that halves LR when validation metric stops improving |
