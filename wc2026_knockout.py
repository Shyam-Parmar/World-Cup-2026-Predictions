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

# This script is for knockout stage matches only. There is no draw outcome
# since these games go to extra time and penalties until a winner is found.
# The model is trained as a binary classifier: home_win vs away_win.

from wc2026 import (
    RAW_URL, SPLIT_DATE, TOURNAMENT_START,
    N_SHORT, N_LONG, STAGE_ORDER, ROLLING_STATS, FEATURE_COLS,
    load_data, build_rolling, compute_elo, compute_h2h,
    add_stage_encoding, add_neutral,
    build_upcoming_features, build_base_model,
)

MODEL_PATH   = "wc2026_knockout_model.pkl"
SAVE_MODEL   = True

CALIB_METHOD = "isotonic"
CALIB_FRAC   = 0.2

TARGET_COL = "actual_result"


# =========================
# Target
# =========================

def add_target_no_draw(df):
    """
    Binary target for knockout matches. Draws are dropped, since a real
    knockout match always ends in a winner after extra time or penalties.
    """
    df = df.copy()
    df["actual_result"] = np.where(df["home_score"] > df["away_score"], "home_win", "away_win")
    return df


def build_feature_matrix_knockout(df):
    """Same pipeline as the main script, but rows with a regular time draw are removed."""
    df    = build_rolling(df)
    df, _ = compute_elo(df)
    df    = compute_h2h(df)
    df    = add_stage_encoding(df)
    df    = add_neutral(df)

    is_draw = df["home_score"] == df["away_score"]
    print(f"Dropping {int(is_draw.sum()):,} drawn matches (no winner without extra time data)")
    df = df[~is_draw].copy()

    df = add_target_no_draw(df)

    keep = ["date", "home_team", "away_team", "home_score", "away_score"] + FEATURE_COLS + [TARGET_COL]
    df_features = df[keep].copy()

    print(f"Feature matrix shape: {df_features.shape}")
    return df_features


# =========================
# Model Training
# =========================

def split_data(df_features):
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


def train_calibrated_classifier(X_train, y_train, X_test, y_test,
                                method=CALIB_METHOD, calib_frac=CALIB_FRAC):
    le          = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    classes     = le.classes_

    cut              = int(len(X_train) * (1 - calib_frac))
    X_core,  y_core  = X_train.iloc[:cut], y_train_enc[:cut]
    X_calib, y_calib = X_train.iloc[cut:], y_train_enc[cut:]

    base = build_base_model()
    base.fit(X_core, y_core)

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
    proba_df = pd.DataFrame(proba, columns=clf_classes).rename(columns={
        "away_win": "prob_away_win",
        "home_win": "prob_home_win",
    })
    for c in ["prob_home_win", "prob_away_win"]:
        proba_df[c] = (proba_df[c] * 100).round(1)
    return proba_df


def _pred_label(proba_df):
    return np.where(proba_df["prob_home_win"] >= proba_df["prob_away_win"], "home_win", "away_win")


def generate_predictions(df_predict, clf_pipeline, clf_classes):
    """Win probabilities for upcoming knockout fixtures. No draw column, since extra time forces a winner."""
    proba    = clf_pipeline.predict_proba(df_predict[FEATURE_COLS])
    proba_df = _proba_frame(proba, clf_classes)

    df_results = df_predict[["date", "home_team", "away_team"]].reset_index(drop=True)
    df_results = pd.concat([df_results, proba_df.reset_index(drop=True)], axis=1)
    df_results["pred_result"] = _pred_label(df_results)

    print(f"Predictions generated for {len(df_results)} upcoming knockout matches")
    return df_results


def run_backtest(df_features, clf_pipeline, clf_classes):
    """Evaluate on WC 2026 knockout matches already played. Drawn regulation results are excluded
    upstream, so this only scores matches that already have a settled winner in the data."""
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
        print(f"Results breakdown:\n{df_bt['actual_result'].value_counts()}")
        print(f"Predicted breakdown:\n{df_bt['pred_result'].value_counts()}")
        print(f"Correct by outcome:\n{df_bt.groupby('actual_result')['correct'].mean().round(3)}")

    return df_bt[["date", "home_team", "away_team", "actual_scoreline",
                  "prob_home_win", "prob_away_win",
                  "pred_result", "actual_result", "correct"]]


# =========================
# Main
# =========================

def main():
    df_raw, df_upcoming = load_data(RAW_URL)

    df_features = build_feature_matrix_knockout(df_raw)
    df_predict  = build_upcoming_features(df_upcoming, df_raw)

    X_train, X_test, y_train, y_test = split_data(df_features)

    clf_pipeline, base_pipeline, le, clf_classes, proba_base, proba_cal, y_test_enc = \
        train_calibrated_classifier(X_train, y_train, X_test, y_test)

    plot_reliability(y_test_enc, proba_base, proba_cal, clf_classes)
    plot_feature_importance(base_pipeline)

    df_results  = generate_predictions(df_predict, clf_pipeline, clf_classes)
    print(df_results.to_string(index=False))

    df_backtest = run_backtest(df_features, clf_pipeline, clf_classes)
    print(df_backtest.to_string(index=False))

    if SAVE_MODEL:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "clf_pipeline"     : clf_pipeline,
                "base_pipeline"    : base_pipeline,
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
