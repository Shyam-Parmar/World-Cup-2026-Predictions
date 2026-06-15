"""
World Cup 2026 Prediction Dashboard

Trains the model fresh on startup, then on each run:
  - fetches the latest results CSV from RAW_URL
  - rebuilds features for completed and upcoming matches
  - predicts upcoming fixtures
  - backtests WC 2026 matches that now have scores

Newly played matches flow into the backtest automatically.
Requires wc2026.py (the training script) in the same folder.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wc2026 as wc   # feature engineering + constants from the training script


# =========================
# Config
# =========================

RESULT_LABELS = {
    "home_win": "Home Win",
    "draw"    : "Draw",
    "away_win": "Away Win",
}

# =========================
# Prediction Helpers
# =========================

def predict_result(df):
    """Derive predicted result from the highest probability column."""
    return df[["prob_home_win", "prob_draw", "prob_away_win"]].idxmax(axis=1).map({
        "prob_home_win": "home_win",
        "prob_draw"    : "draw",
        "prob_away_win": "away_win",
    })


def _proba_pct(clf_pipeline, X, clf_classes):
    """Predict probabilities and return as integer-percent columns."""
    proba = clf_pipeline.predict_proba(X)
    df    = pd.DataFrame(proba, columns=clf_classes).rename(columns={
        "away_win": "prob_away_win",
        "draw"    : "prob_draw",
        "home_win": "prob_home_win",
    })
    for c in ["prob_home_win", "prob_draw", "prob_away_win"]:
        df[c] = (df[c] * 100).round(0).astype(int)
    return df


def get_predictions(clf_pipeline, clf_classes, df_predict, feature_cols):
    """Probabilities for upcoming fixtures (matches with no scores yet)."""
    if df_predict.empty:
        return df_predict.copy()
    proba_df = _proba_pct(clf_pipeline, df_predict[feature_cols], clf_classes)
    df = df_predict[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True)
    df = pd.concat([df, proba_df.reset_index(drop=True)], axis=1)
    df["pred_result"] = predict_result(df)
    return df


def get_backtest(clf_pipeline, clf_classes, df_features, feature_cols, tournament_start):
    """Predictions vs actuals for WC 2026 matches that now have scores."""
    df_bt = df_features[df_features["date"] >= tournament_start].copy()
    if df_bt.empty:
        return df_bt
    proba_df = _proba_pct(clf_pipeline, df_bt[feature_cols], clf_classes)
    df = df_bt[["date", "home_team", "away_team", "home_score", "away_score", "actual_result"]].reset_index(drop=True)
    df = pd.concat([df, proba_df.reset_index(drop=True)], axis=1)
    df["pred_result"]      = predict_result(df)
    df["correct"]          = df["pred_result"] == df["actual_result"]
    df["actual_scoreline"] = (df["home_score"].astype(int).astype(str) + " - "
                              + df["away_score"].astype(int).astype(str))
    return df


# =========================
# Display Helpers
# =========================

def color_correct(val):
    if val == "✅ Correct":
        return "background-color: #1e8449; color: white; font-weight: bold"
    elif val == "❌ Wrong":
        return "background-color: #922b21; color: white; font-weight: bold"
    return ""


def label_result(row, col):
    """Replace home_win/away_win with team name, keep draw as Draw."""
    val = row[col]
    if val == "home_win":
        return row["home_team"]
    elif val == "away_win":
        return row["away_team"]
    return "Draw"


# =========================
# Page Layout
# =========================

st.set_page_config(page_title="WC 2026 Predictions", page_icon="⚽", layout="wide")
st.title("⚽ FIFA World Cup 2026 — Match Predictions")

@st.cache_resource(show_spinner="Training model on latest data...")
def load_model_and_data():
    df_hist, df_upcoming = wc.load_data(wc.RAW_URL)
    df_features  = wc.build_feature_matrix(df_hist)
    df_predict   = wc.build_upcoming_features(df_upcoming, df_hist)
    last_played  = df_hist["date"].max()
    X_train, X_test, y_train, y_test = wc.split_data(df_features)
    clf_pipeline, _, le, clf_classes, _, _, _ = \
        wc.train_calibrated_classifier(X_train, y_train, X_test, y_test)
    return clf_pipeline, clf_classes, df_features, df_predict, last_played

clf_pipeline, clf_classes, df_features, df_predict, last_played = load_model_and_data()
FEATURE_COLS     = wc.FEATURE_COLS
TOURNAMENT_START = wc.TOURNAMENT_START

df_results  = get_predictions(clf_pipeline, clf_classes, df_predict, FEATURE_COLS)
df_backtest = get_backtest(clf_pipeline, clf_classes, df_features, FEATURE_COLS, TOURNAMENT_START)

# -- Sidebar --
with st.sidebar:
    st.header("Data")
    st.caption(f"Latest result in data: **{last_played.date()}**")
    st.caption(f"Upcoming fixtures: **{len(df_results)}**")
    st.caption(f"WC matches played: **{len(df_backtest)}**")
    if st.button("🔄 Refresh latest results"):
        st.cache_data.clear()
        st.rerun()


# =========================
# Tabs
# =========================

tab_upcoming, tab_backtest, tab_summary = st.tabs([
    "🔮 Upcoming Fixtures",
    "📋 Played Matches",
    "📊 Model Summary",
])


# =========================
# Tab 1 — Upcoming Fixtures
# =========================

with tab_upcoming:
    st.subheader("Upcoming World Cup Fixtures")
    st.caption("Calibrated probabilities from an XGBoost model trained on international results since 1884.")

    if df_results.empty:
        st.info("No upcoming fixtures without scores. Every scheduled match in the data has been played.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.selectbox(
                "Filter by date",
                options=["All"] + sorted(df_results["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
            )
        with col2:
            result_filter = st.selectbox(
                "Filter by predicted result",
                options=["All", "home_win", "draw", "away_win"],
            )

        all_teams = sorted(set(df_results["home_team"].tolist() + df_results["away_team"].tolist()))
        team_filter = st.selectbox("Filter by team", options=["All"] + all_teams)

        df_show = df_results.copy()
        if date_filter != "All":
            df_show = df_show[df_show["date"].dt.strftime("%Y-%m-%d") == date_filter]
        if result_filter != "All":
            df_show = df_show[df_show["pred_result"] == result_filter]
        if team_filter != "All":
            df_show = df_show[(df_show["home_team"] == team_filter) | (df_show["away_team"] == team_filter)]

        df_show["date"] = df_show["date"].dt.strftime("%Y-%m-%d")

        for _, row in df_show.iterrows():
            with st.container():
                col_home, col_vs, col_away = st.columns([2, 1, 2])
                with col_home:
                    st.markdown(f"<h3 style='text-align:right'>{row['home_team']}</h3>", unsafe_allow_html=True)
                with col_vs:
                    st.markdown("<h3 style='text-align:center; color:gray'>vs</h3>", unsafe_allow_html=True)
                with col_away:
                    st.markdown(f"<h3 style='text-align:left'>{row['away_team']}</h3>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<p style='text-align:center; color:#2ecc71; font-size:20px; font-weight:bold'>{row['prob_home_win']}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; color:gray; font-size:12px'>{row['home_team']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p style='text-align:center; color:#f39c12; font-size:20px; font-weight:bold'>{row['prob_draw']}%</p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align:center; color:gray; font-size:12px'>Draw</p>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<p style='text-align:center; color:#e74c3c; font-size:20px; font-weight:bold'>{row['prob_away_win']}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; color:gray; font-size:12px'>{row['away_team']}</p>", unsafe_allow_html=True)

                bar_html = f"""
                <div style='width:100%; height:20px; border-radius:10px; overflow:hidden; display:flex; margin-bottom:8px'>
                    <div style='width:{row["prob_home_win"]}%; background:#2ecc71; height:100%'></div>
                    <div style='width:{row["prob_draw"]}%; background:#f39c12; height:100%'></div>
                    <div style='width:{row["prob_away_win"]}%; background:#e74c3c; height:100%'></div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

                if row["pred_result"] == "home_win":
                    pred_label = row["home_team"]
                elif row["pred_result"] == "away_win":
                    pred_label = row["away_team"]
                else:
                    pred_label = "Draw"

                st.caption(f"📅 {row['date']}  |  🏆 {row['tournament']}  |  Predicted: **{pred_label}**")
                st.divider()

        st.caption(f"Showing {len(df_show)} of {len(df_results)} upcoming fixtures")


# =========================
# Tab 2 — Played Matches
# =========================

with tab_backtest:
    st.subheader("Played World Cup 2026 Matches")
    st.caption("Model predictions vs actual results for matches already played.")

    if df_backtest.empty:
        st.info("No WC 2026 matches with scores yet. The backtest will populate once games are played.")
    else:
        total   = len(df_backtest)
        correct = int(df_backtest["correct"].sum())
        acc     = df_backtest["correct"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Matches Played",      total)
        m2.metric("Correct Predictions", f"{correct} / {total}")
        m3.metric("Accuracy",            f"{acc:.1%}")

        st.divider()

        correct_filter = st.radio("Show", options=["All", "Correct only", "Incorrect only"], horizontal=True)
        all_bt_teams = sorted(set(df_backtest["home_team"].tolist() + df_backtest["away_team"].tolist()))
        team_filter_bt = st.selectbox("Filter by team", options=["All"] + all_bt_teams)

        df_bt_show = df_backtest.copy()
        if correct_filter == "Correct only":
            df_bt_show = df_bt_show[df_bt_show["correct"] == True]
        elif correct_filter == "Incorrect only":
            df_bt_show = df_bt_show[df_bt_show["correct"] == False]
        if team_filter_bt != "All":
            df_bt_show = df_bt_show[(df_bt_show["home_team"] == team_filter_bt) | (df_bt_show["away_team"] == team_filter_bt)]

        df_bt_show["date"]        = df_bt_show["date"].dt.strftime("%Y-%m-%d")
        df_bt_show["pred_result"] = df_bt_show.apply(lambda r: label_result(r, "pred_result"), axis=1)
        df_bt_show["actual_result"]      = df_bt_show.apply(lambda r: label_result(r, "actual_result"),      axis=1)
        df_bt_show["correct"]     = df_bt_show["correct"].map({True: "✅ Correct", False: "❌ Wrong"})

        display_cols = [
            "date", "home_team", "away_team", "actual_scoreline",
            "prob_home_win", "prob_draw", "prob_away_win",
            "pred_result", "actual_result", "correct",
        ]

        st.dataframe(
            df_bt_show[display_cols].style
                .map(color_correct, subset=["correct"])
                .format({"prob_home_win": "{}%", "prob_draw": "{}%", "prob_away_win": "{}%"}),
            use_container_width=True,
            hide_index=True,
        )


# =========================
# Tab 3 — Model Summary
# =========================

with tab_summary:
    st.subheader("Model Performance Summary")

    if df_backtest.empty:
        st.info("Summary metrics appear once WC 2026 matches have been played.")
    else:
        st.markdown("#### Overall Accuracy")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches Played", len(df_backtest))
        col2.metric("Correct",              f"{int(df_backtest['correct'].sum())} / {len(df_backtest)}")
        col3.metric("Accuracy",             f"{df_backtest['correct'].mean():.1%}")

        st.divider()

        st.markdown("#### Accuracy by Outcome")
        by_outcome = (
            df_backtest.groupby("actual_result")["correct"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "Correct", "count": "Total", "mean": "Accuracy"})
            .reset_index()
        )
        by_outcome["Accuracy"] = (by_outcome["Accuracy"] * 100).round(1).astype(str) + "%"
        by_outcome["actual_result"]   = by_outcome["actual_result"].map(RESULT_LABELS)
        by_outcome             = by_outcome.rename(columns={"actual_result": "Outcome"})
        st.dataframe(by_outcome, use_container_width=True, hide_index=True)

        st.divider()

    if not df_results.empty:
        st.markdown("#### Prediction Distribution (Upcoming Fixtures)")
        pred_dist = df_results["pred_result"].value_counts().reset_index()
        pred_dist.columns             = ["Predicted Result", "Count"]
        pred_dist["Predicted Result"] = pred_dist["Predicted Result"].map(RESULT_LABELS)
        st.bar_chart(pred_dist.set_index("Predicted Result"))
        st.divider()

    st.markdown("#### Features Used")
    st.code("\n".join(FEATURE_COLS))
