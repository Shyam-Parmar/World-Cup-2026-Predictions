# =========================
# Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import accuracy_score, log_loss, classification_report
from sklearn.calibration   import CalibratedClassifierCV, calibration_curve
from xgboost               import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# Constants
# =========================

RAW_URL          = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SPLIT_DATE       = "2018-01-01"
TOURNAMENT_START = "2026-06-11"
MODEL_PATH       = "wc2026_model.pkl"
SAVE_MODEL       = True

N_SHORT = 5
N_LONG  = 10

CALIB_METHOD = "isotonic"   # switch to "sigmoid" if the reliability curves look jagged
CALIB_FRAC   = 0.2          # fraction of train (most recent slice) used to calibrate

STAGE_ORDER = {
    "Group Stage"    : 1,
    "Round of 16"    : 2,
    "Quarter-finals" : 3,
    "Semi-finals"    : 4,
    "3rd Place Final": 5,
    "Final"          : 6,
}

ROLLING_STATS = [
    "goals_for", "goals_against", "goal_diff",
    "win", "draw", "loss",
]

FEATURE_COLS = [
    # ELO
    "elo_home", "elo_away", "elo_diff",
    # Rolling 10, home
    "home_roll10_goals_for", "home_roll10_goals_against", "home_roll10_goal_diff",
    # Rolling 10, away
    "away_roll10_goals_for", "away_roll10_goals_against", "away_roll10_goal_diff",
    # Rolling 5, home
    "home_roll5_goals_for", "home_roll5_goals_against", "home_roll5_goal_diff",
    # Rolling 5, away
    "away_roll5_goals_for", "away_roll5_goals_against", "away_roll5_goal_diff",
    # Head to head
    "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_n_matches",
    # Context
    "neutral", "stage_numeric",
]

TARGET_COL = "actual_result"


# =========================
# Load & Clean
# =========================

def load_data(url):
    """
    Load international results CSV and split into:
      df_historical : completed matches with scores
      df_upcoming   : scheduled matches with missing scores
    Excludes friendlies.
    """
    df = pd.read_csv(url)

    df["date"]       = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"]    = df["neutral"].astype(str).str.upper() == "TRUE"

    df = df[df["tournament"] != "Friendly"].reset_index(drop=True)
    df = df.dropna(subset=["home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    has_scores    = df["home_score"].notna() & df["away_score"].notna()
    df_historical = df[has_scores].reset_index(drop=True)
    df_upcoming   = df[~has_scores].reset_index(drop=True)

    print(f"Historical : {len(df_historical):,} matches "
          f"({df_historical['date'].min().date()} to {df_historical['date'].max().date()})")
    print(f"Upcoming   : {len(df_upcoming):,} matches "
          f"({df_upcoming['date'].min().date()} to {df_upcoming['date'].max().date()})")

    return df_historical, df_upcoming


# =========================
# Feature Engineering
# =========================

def melt_to_team_level(df):
    """Convert wide match DataFrame into long format (one row per team per match)."""
    shared = ["date", "tournament", "neutral"]

    home_df = df[shared + ["home_team", "away_team", "home_score", "away_score"]].copy()
    home_df = home_df.rename(columns={
        "home_team" : "team",
        "away_team" : "opponent",
        "home_score": "goals_for",
        "away_score": "goals_against",
    })
    home_df["is_home"] = True

    away_df = df[shared + ["away_team", "home_team", "away_score", "home_score"]].copy()
    away_df = away_df.rename(columns={
        "away_team" : "team",
        "home_team" : "opponent",
        "away_score": "goals_for",
        "home_score": "goals_against",
    })
    away_df["is_home"] = False

    long_df              = pd.concat([home_df, away_df], ignore_index=True)
    long_df              = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df["win"]       = (long_df["goals_for"] >  long_df["goals_against"]).astype(int)
    long_df["draw"]      = (long_df["goals_for"] == long_df["goals_against"]).astype(int)
    long_df["loss"]      = (long_df["goals_for"] <  long_df["goals_against"]).astype(int)

    return long_df


def compute_rolling_features(long_df, n):
    """Rolling mean of the last N matches per team. Shifted by 1 to avoid leakage (training path)."""
    rolled = long_df.groupby("team")[ROLLING_STATS].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )
    rolled.columns = [f"roll{n}_{c}" for c in rolled.columns]
    return rolled


def compute_latest_rolling(long_df, n):
    """Mean of each team's most recent n matches. No shift, since upcoming games come after all of these."""
    rolled = long_df.groupby("team")[ROLLING_STATS].transform(
        lambda x: x.rolling(n, min_periods=1).mean()
    )
    rolled.columns = [f"roll{n}_{c}" for c in rolled.columns]
    return rolled


def build_rolling(df_raw):
    """Build rolling features and merge back to match level."""
    long_df    = melt_to_team_level(df_raw)
    roll_short = compute_rolling_features(long_df, N_SHORT)
    roll_long  = compute_rolling_features(long_df, N_LONG)
    long_df    = pd.concat([long_df, roll_short, roll_long], axis=1)

    roll_cols = [c for c in long_df.columns if c.startswith("roll")]
    home_roll = long_df[long_df["is_home"] == True].copy()
    away_roll = long_df[long_df["is_home"] == False].copy()

    home_roll = home_roll[["date", "team", "opponent"] + roll_cols].rename(
        columns={"team": "home_team", "opponent": "away_team", **{c: f"home_{c}" for c in roll_cols}}
    )
    away_roll = away_roll[["date", "team", "opponent"] + roll_cols].rename(
        columns={"team": "away_team", "opponent": "home_team", **{c: f"away_{c}" for c in roll_cols}}
    )

    df = df_raw.merge(home_roll, on=["date", "home_team", "away_team"], how="left")
    df = df.merge(away_roll,     on=["date", "home_team", "away_team"], how="left")

    print(f"Matches after rolling merge: {len(df):,}")
    return df


def compute_elo(df, k=30, initial=1500):
    """Compute ELO chronologically. Returns (df with pre-match ratings, final post-match rating dict)."""
    df            = df.sort_values("date").copy()
    elo           = {}
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h  = elo.get(h, initial)
        r_a  = elo.get(a, initial)

        elo_home_list.append(r_h)
        elo_away_list.append(r_a)

        e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        e_a = 1 - e_h

        if row["home_score"] > row["away_score"]:
            s_h, s_a = 1, 0
        elif row["home_score"] < row["away_score"]:
            s_h, s_a = 0, 1
        else:
            s_h, s_a = 0.5, 0.5

        elo[h] = r_h + k * (s_h - e_h)
        elo[a] = r_a + k * (s_a - e_a)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    return df, elo


def _h2h_rates(past, h, a):
    """Helper: home/away/draw rates and match count for pair (h, a) from prior matches."""
    h2h = past[
        ((past["home_team"] == h) & (past["away_team"] == a)) |
        ((past["home_team"] == a) & (past["away_team"] == h))
    ]
    n = len(h2h)
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    home_wins = (
        ((h2h["home_team"] == h) & (h2h["home_score"] > h2h["away_score"])) |
        ((h2h["away_team"] == h) & (h2h["away_score"] > h2h["home_score"]))
    ).sum()
    away_wins = (
        ((h2h["home_team"] == a) & (h2h["home_score"] > h2h["away_score"])) |
        ((h2h["away_team"] == a) & (h2h["away_score"] > h2h["home_score"]))
    ).sum()
    draws = n - home_wins - away_wins

    return home_wins / n, away_wins / n, draws / n, n


def compute_h2h(df):
    """Historical H2H win/draw/loss rates between each pair using only prior matches."""
    df = df.sort_values("date").reset_index(drop=True)

    rates = [_h2h_rates(df.iloc[:i], row["home_team"], row["away_team"])
             for i, row in df.iterrows()]

    df["h2h_home_win_rate"], df["h2h_away_win_rate"], df["h2h_draw_rate"], df["h2h_n_matches"] = \
        zip(*rates)

    return df


def add_stage_encoding(df):
    """Encode tournament stage as ordinal and flag knockout rounds."""
    def infer_stage(tournament):
        t = str(tournament).lower()
        if "final" in t and "semi" not in t and "3rd" not in t and "quarter" not in t:
            return "Final"
        if "semi" in t:
            return "Semi-finals"
        if "quarter" in t:
            return "Quarter-finals"
        if "round of 16" in t or "last 16" in t:
            return "Round of 16"
        if "3rd" in t or "third" in t:
            return "3rd Place Final"
        if "group" in t:
            return "Group Stage"
        return "Other"

    df["stage"]         = df["tournament"].apply(infer_stage)
    df["stage_numeric"] = df["stage"].map(STAGE_ORDER).fillna(0).astype(int)
    df["is_knockout"]   = (df["stage_numeric"] >= 2).astype(int)
    return df


def add_neutral(df):
    """Convert neutral ground flag to binary integer."""
    df["neutral"] = df["neutral"].astype(int)
    return df


def add_target(df):
    """Add match result as a 3-class target: home_win, draw, away_win."""
    df["actual_result"] = np.where(
        df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw")
    )
    return df


def build_feature_matrix(df):
    """Apply all feature engineering steps and return the final feature matrix (training path)."""
    df    = build_rolling(df)
    df, _ = compute_elo(df)
    df    = compute_h2h(df)
    df    = add_stage_encoding(df)
    df    = add_neutral(df)
    df    = add_target(df)

    keep = ["date", "home_team", "away_team", "home_score", "away_score"] + FEATURE_COLS + [TARGET_COL]
    df_features = df[keep].copy()

    print(f"Feature matrix shape: {df_features.shape}")
    print(f"Features: {len(FEATURE_COLS)}")
    return df_features


# =========================
# Build Upcoming Features
# =========================

def build_upcoming_features(df_upcoming, df_historical):
    """Build prediction features for upcoming fixtures using only historical data, matching the training logic."""

    # ELO: use FINAL post-match ratings, not stale pre-match values
    _, final_elo = compute_elo(df_historical.copy())

    df_upcoming             = df_upcoming.copy()
    df_upcoming["elo_home"] = df_upcoming["home_team"].map(final_elo)
    df_upcoming["elo_away"] = df_upcoming["away_team"].map(final_elo)
    df_upcoming["elo_diff"] = df_upcoming["elo_home"] - df_upcoming["elo_away"]

    # Rolling: mean of each team's last N historical matches, including the most recent
    long_hist  = melt_to_team_level(df_historical)
    roll_short = compute_latest_rolling(long_hist, N_SHORT)
    roll_long  = compute_latest_rolling(long_hist, N_LONG)
    long_hist  = pd.concat([long_hist, roll_short, roll_long], axis=1)

    roll_cols    = [c for c in long_hist.columns if c.startswith("roll")]
    latest_stats = long_hist.groupby("team")[roll_cols].last().reset_index()

    home_stats = latest_stats.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in roll_cols}})
    away_stats = latest_stats.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in roll_cols}})

    df_upcoming = df_upcoming.merge(home_stats, on="home_team", how="left")
    df_upcoming = df_upcoming.merge(away_stats, on="away_team", how="left")

    # H2H: all historical matches are strictly prior to upcoming fixtures
    rates = [_h2h_rates(df_historical, row["home_team"], row["away_team"])
             for _, row in df_upcoming.iterrows()]
    df_upcoming["h2h_home_win_rate"], df_upcoming["h2h_away_win_rate"], \
        df_upcoming["h2h_draw_rate"], df_upcoming["h2h_n_matches"] = zip(*rates)

    # Stage & Neutral
    df_upcoming = add_stage_encoding(df_upcoming)
    df_upcoming = add_neutral(df_upcoming)

    return df_upcoming[["date", "home_team", "away_team", "tournament"] + FEATURE_COLS]


# =========================
# Model Training
# =========================

def split_data(df_features):
    """Time-based train/test split. df_features is already chronological."""
    df_train = df_features[df_features["date"] <  SPLIT_DATE].copy()
    df_test  = df_features[df_features["date"] >= SPLIT_DATE].copy()

    X_train = df_train[FEATURE_COLS]
    X_test  = df_test[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    y_test  = df_test[TARGET_COL]

    print(f"Train: {len(df_train):,} matches (up to {SPLIT_DATE})")
    print(f"Test : {len(df_test):,} matches ({SPLIT_DATE} onwards)")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test


def build_base_model():
    """The base XGBoost pipeline. No class weights, so calibration is not fighting a distortion."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            eval_metric      = "mlogloss",
            random_state     = 42,
            n_jobs           = -1,
        )),
    ])


def train_calibrated_classifier(X_train, y_train, X_test, y_test,
                                method=CALIB_METHOD, calib_frac=CALIB_FRAC):
    """Train XGBoost, then calibrate on a time-ordered holdout carved from the end of train."""
    le          = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes     = le.classes_

    # X_train is chronological, so the last slice is the most recent. Use it to calibrate.
    cut              = int(len(X_train) * (1 - calib_frac))
    X_core,  y_core  = X_train.iloc[:cut], y_train_enc[:cut]
    X_calib, y_calib = X_train.iloc[cut:], y_train_enc[cut:]

    base = build_base_model()
    base.fit(X_core, y_core)

    # version-robust prefit calibration
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(base, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)

    proba_base = base.predict_proba(X_test)
    proba_cal  = calibrated.predict_proba(X_test)

    print(f"Log loss  before: {log_loss(y_test_enc, proba_base):.4f}  "
          f"after: {log_loss(y_test_enc, proba_cal):.4f}")
    y_pred_enc = calibrated.predict(X_test)
    print(f"Accuracy  before: {accuracy_score(y_test_enc, base.predict(X_test)):.3f}  "
          f"after: {accuracy_score(y_test_enc, y_pred_enc):.3f}")
    print(f"Classes: {list(classes)}")
    print(classification_report(y_test_enc, y_pred_enc, target_names=classes))

    return calibrated, base, le, classes, proba_base, proba_cal, y_test_enc


def plot_reliability(y_test_enc, proba_base, proba_cal, classes):
    """One-vs-rest reliability curve per class, before and after calibration."""
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    for i, cls in enumerate(classes):
        ax    = axes[i]
        y_bin = (y_test_enc == i).astype(int)
        for proba, label in [(proba_base, "before"), (proba_cal, "after")]:
            frac_pos, mean_pred = calibration_curve(y_bin, proba[:, i], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(cls)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(base_pipeline):
    """Plot XGBoost feature importance from the base (uncalibrated) pipeline."""
    xgb_model   = base_pipeline.named_steps["model"]
    importances = xgb_model.feature_importances_
    feat_df     = pd.DataFrame({
        "feature"   : FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("Top 15 features (XGBoost):")
    print(feat_df.head(15).to_string(index=False))


# =========================
# Predictions
# =========================

def _proba_frame(proba, clf_classes):
    """Turn a probability array into percentage columns."""
    proba_df = pd.DataFrame(proba, columns=clf_classes).rename(columns={
        "away_win": "prob_away_win",
        "draw"    : "prob_draw",
        "home_win": "prob_home_win",
    })
    for c in ["prob_home_win", "prob_draw", "prob_away_win"]:
        proba_df[c] = (proba_df[c] * 100).round(1)
    return proba_df


def _pred_label(proba_df):
    """Pick the highest-probability outcome label per row."""
    return proba_df[["prob_home_win", "prob_draw", "prob_away_win"]].idxmax(axis=1).map({
        "prob_home_win": "home_win",
        "prob_draw"    : "draw",
        "prob_away_win": "away_win",
    })


def generate_predictions(df_predict, clf_pipeline, clf_classes):
    """Generate win/draw/loss probabilities for upcoming fixtures."""
    proba    = clf_pipeline.predict_proba(df_predict[FEATURE_COLS])
    proba_df = _proba_frame(proba, clf_classes)

    df_results = df_predict[["date", "home_team", "away_team"]].reset_index(drop=True)
    df_results = pd.concat([df_results, proba_df.reset_index(drop=True)], axis=1)
    df_results["pred_result"] = _pred_label(df_results)

    print(f"Predictions generated for {len(df_results)} upcoming matches")
    return df_results


def run_backtest(df_features, clf_pipeline, clf_classes):
    """Evaluate the model on WC 2026 matches already played."""
    df_backtest = df_features[df_features["date"] >= TOURNAMENT_START].copy()
    print(f"Backtest matches: {len(df_backtest)}")

    proba    = clf_pipeline.predict_proba(df_backtest[FEATURE_COLS])
    proba_df = _proba_frame(proba, clf_classes)

    df_bt = df_backtest[["date", "home_team", "away_team", "home_score", "away_score", "actual_result"]].reset_index(drop=True)
    df_bt = pd.concat([df_bt, proba_df.reset_index(drop=True)], axis=1)
    df_bt["pred_result"]      = _pred_label(df_bt)
    df_bt["correct"]          = df_bt["pred_result"] == df_bt["actual_result"]
    df_bt["actual_scoreline"] = (df_bt["home_score"].astype(int).astype(str) + " - "
                                 + df_bt["away_score"].astype(int).astype(str))

    if len(df_bt):
        print(f"Backtest accuracy: {df_bt['correct'].mean():.3f}")
        print(f"Correct: {df_bt['correct'].sum()} / {len(df_bt)}")
        print(f"Results breakdown:\n{df_bt["actual_result"].value_counts()}")
        print(f"Predicted breakdown:\n{df_bt['pred_result'].value_counts()}")
        print(f"Correct by outcome:\n{df_bt.groupby("actual_result")['correct'].mean().round(3)}")

    return df_bt[["date", "home_team", "away_team", "actual_scoreline",
                  "prob_home_win", "prob_draw", "prob_away_win",
                  "pred_result", "actual_result", "correct"]]


# =========================
# Main
# =========================

def main():
    # Load
    df_raw, df_upcoming = load_data(RAW_URL)

    # Features
    df_features = build_feature_matrix(df_raw)
    df_predict  = build_upcoming_features(df_upcoming, df_raw)

    # Split
    X_train, X_test, y_train, y_test = split_data(df_features)

    # Train + calibrate
    clf_pipeline, base_pipeline, le, clf_classes, proba_base, proba_cal, y_test_enc = \
        train_calibrated_classifier(X_train, y_train, X_test, y_test)

    # Diagnostics
    plot_reliability(y_test_enc, proba_base, proba_cal, clf_classes)
    plot_feature_importance(base_pipeline)

    # Predictions + backtest
    df_results  = generate_predictions(df_predict, clf_pipeline, clf_classes)
    print(df_results.to_string(index=False))

    df_backtest = run_backtest(df_features, clf_pipeline, clf_classes)
    print(df_backtest.to_string(index=False))

    # Save
    if SAVE_MODEL:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "clf_pipeline"     : clf_pipeline,   # calibrated, serves trustworthy probabilities
                "base_pipeline"    : base_pipeline,  # uncalibrated, for feature importance
                "le"               : le,
                "clf_classes"      : clf_classes,
                "df_predict"       : df_predict,
                "df_features"      : df_features,
                "FEATURE_COLS"     : FEATURE_COLS,
                "TOURNAMENT_START" : TOURNAMENT_START,
            }, f)
        print(f"Model and data saved to {MODEL_PATH}")

    return df_results, df_backtest


if __name__ == "__main__":
    df_results, df_backtest = main()
