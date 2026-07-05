"""
World Cup 2026 Full Bracket Dashboard

Streamlit app that walks the real bracket, Round of 32 to Final, using
wc2026_knockout_model.pkl. Same logic as wc2026_bracket_simulate.py,
just returning structured results for display instead of printing.
"""

import os
import sys
import pickle
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wc2026 as wc

MODEL_PATH = "wc2026_knockout_model.pkl"

ROUND_DATE = {
    "Round of 32": pd.Timestamp("2026-06-28"),
    "Round of 16": pd.Timestamp("2026-07-04"),
    "Quarterfinal": pd.Timestamp("2026-07-09"),
    "Semifinal": pd.Timestamp("2026-07-14"),
    "Final": pd.Timestamp("2026-07-19"),
}

ROUND_WINDOW = {
    "Round of 32": (pd.Timestamp("2026-06-28"), pd.Timestamp("2026-07-03")),
    "Round of 16": (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-07")),
    "Quarterfinal": (pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-11")),
    "Semifinal": (pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")),
    "Final": (pd.Timestamp("2026-07-19"), pd.Timestamp("2026-07-19")),
}

R32_SLOTS = [
    {"Paraguay", "Germany"},
    {"France", "Sweden"},
    {"Canada", "South Africa"},
    {"Netherlands", "Morocco"},
    {"Portugal", "Croatia"},
    {"Spain", "Austria"},
    {"United States", "Bosnia and Herzegovina"},
    {"Belgium", "Senegal"},
    {"Brazil", "Japan"},
    {"Ivory Coast", "Norway"},
    {"Mexico", "Ecuador"},
    {"England", "DR Congo"},
    {"Argentina", "Cape Verde"},
    {"Australia", "Egypt"},
    {"Switzerland", "Algeria"},
    {"Colombia", "Ghana"},
]

R16_SLOT_PAIRS = [(0, 1), (2, 3), (8, 9), (10, 11), (4, 5), (6, 7), (12, 13), (14, 15)]
R16_DATES = [
    pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-04"),
    pd.Timestamp("2026-07-05"), pd.Timestamp("2026-07-05"),
    pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
    pd.Timestamp("2026-07-07"), pd.Timestamp("2026-07-07"),
]

QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
QF_DATES = [
    pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-10"),
    pd.Timestamp("2026-07-11"), pd.Timestamp("2026-07-11"),
]

SF_PAIRS = [(0, 1), (2, 3)]
SF_DATES = [pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")]

FINAL_DATE = pd.Timestamp("2026-07-19")


# =========================
# Model and prediction
# =========================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["clf_pipeline"], saved["clf_classes"], saved["FEATURE_COLS"]


def predict_fixtures(df_fixtures, df_hist, clf_pipeline, clf_classes, feature_cols):
    df_pred = wc.build_upcoming_features(df_fixtures, df_hist)
    proba = clf_pipeline.predict_proba(df_pred[feature_cols])
    proba_df = pd.DataFrame(proba, columns=clf_classes).rename(columns={
        "home_win": "prob_home_win",
        "away_win": "prob_away_win",
    })
    for c in ["prob_home_win", "prob_away_win"]:
        proba_df[c] = (proba_df[c] * 100).round(1)

    out = df_fixtures[["date", "home_team", "away_team"]].reset_index(drop=True)
    out = pd.concat([out, proba_df.reset_index(drop=True)], axis=1)
    out["predicted_winner"] = out.apply(
        lambda r: r["home_team"] if r["prob_home_win"] >= r["prob_away_win"] else r["away_team"],
        axis=1,
    )
    return out


def resolve_match(team_a, team_b, round_name, match_date, df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols):
    """Return a dict describing the match and its winner."""
    start, end = ROUND_WINDOW[round_name]
    played = df_hist[
        (df_hist["date"] >= start) & (df_hist["date"] <= end) &
        (((df_hist["home_team"] == team_a) & (df_hist["away_team"] == team_b)) |
         ((df_hist["home_team"] == team_b) & (df_hist["away_team"] == team_a)))
    ]
    if not played.empty:
        row = played.iloc[0]
        winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
        return {
            "round": round_name, "date": row["date"], "team_a": row["home_team"],
            "team_b": row["away_team"], "prob_a": None, "prob_b": None,
            "winner": winner, "status": "played",
        }

    match = df_upcoming[
        ((df_upcoming["home_team"] == team_a) & (df_upcoming["away_team"] == team_b)) |
        ((df_upcoming["home_team"] == team_b) & (df_upcoming["away_team"] == team_a))
    ]
    if not match.empty:
        fixture = match.iloc[[0]]
        status = "scheduled"
    else:
        fixture = pd.DataFrame([{
            "date": match_date,
            "home_team": team_a,
            "away_team": team_b,
            "tournament": "FIFA World Cup",
            "neutral": True,
        }])
        status = "simulated"

    preds = predict_fixtures(fixture, df_hist, clf_pipeline, clf_classes, feature_cols)
    row = preds.iloc[0]
    return {
        "round": round_name, "date": row["date"], "team_a": row["home_team"],
        "team_b": row["away_team"], "prob_a": row["prob_home_win"],
        "prob_b": row["prob_away_win"], "winner": row["predicted_winner"], "status": status,
    }


@st.cache_data(show_spinner="Simulating the bracket...")
def run_bracket(_clf_pipeline, clf_classes, feature_cols, refresh_token):
    df_hist, df_upcoming = wc.load_data(wc.RAW_URL)

    results = {"Round of 32": [], "Round of 16": [], "Quarterfinal": [], "Semifinal": [], "Final": []}

    r32_winners = []
    for slot in R32_SLOTS:
        m = resolve_match(*tuple(slot), "Round of 32", ROUND_DATE["Round of 32"], df_hist, df_upcoming, _clf_pipeline, clf_classes, feature_cols)
        results["Round of 32"].append(m)
        r32_winners.append(m["winner"])

    r16_winners = []
    for k, (i, j) in enumerate(R16_SLOT_PAIRS):
        m = resolve_match(r32_winners[i], r32_winners[j], "Round of 16", R16_DATES[k], df_hist, df_upcoming, _clf_pipeline, clf_classes, feature_cols)
        results["Round of 16"].append(m)
        r16_winners.append(m["winner"])

    qf_winners = []
    for k, (i, j) in enumerate(QF_PAIRS):
        m = resolve_match(r16_winners[i], r16_winners[j], "Quarterfinal", QF_DATES[k], df_hist, df_upcoming, _clf_pipeline, clf_classes, feature_cols)
        results["Quarterfinal"].append(m)
        qf_winners.append(m["winner"])

    sf_winners = []
    for k, (i, j) in enumerate(SF_PAIRS):
        m = resolve_match(qf_winners[i], qf_winners[j], "Semifinal", SF_DATES[k], df_hist, df_upcoming, _clf_pipeline, clf_classes, feature_cols)
        results["Semifinal"].append(m)
        sf_winners.append(m["winner"])

    final = resolve_match(sf_winners[0], sf_winners[1], "Final", FINAL_DATE, df_hist, df_upcoming, _clf_pipeline, clf_classes, feature_cols)
    results["Final"].append(final)

    return results, final["winner"]


# =========================
# Display
# =========================

def match_box_html(m, highlight_winner=True):
    prob_a = m["prob_a"] if m["prob_a"] is not None else (100 if m["winner"] == m["team_a"] else 0)
    prob_b = m["prob_b"] if m["prob_b"] is not None else (100 if m["winner"] == m["team_b"] else 0)
    bold_a = "font-weight:700;" if highlight_winner and m["winner"] == m["team_a"] else "color:#888;"
    bold_b = "font-weight:700;" if highlight_winner and m["winner"] == m["team_b"] else "color:#888;"
    return f"""
    <div class="match-box">
      <div class="team-row" style="{bold_a}">{m['team_a']}<span class="pct">{prob_a:.0f}%</span></div>
      <div class="team-row" style="{bold_b}">{m['team_b']}<span class="pct">{prob_b:.0f}%</span></div>
      <div class="match-date">{m['date'].date()}</div>
    </div>
    """


def render_full_bracket(results):
    round_names = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]

    css = """
    <style>
    .bracket { display: flex; gap: 24px; overflow-x: auto; padding: 12px 0; font-family: sans-serif; color: #eee; }
    .round-col { display: flex; flex-direction: column; justify-content: space-around; min-width: 190px; }
    .match-box {
        background: #1e1e1e; border: 1px solid #333; border-radius: 8px;
        padding: 8px 10px; margin: 10px 0; font-size: 13px;
    }
    .team-row { display: flex; justify-content: space-between; padding: 2px 0; }
    .pct { color: #888; font-size: 11px; }
    .match-date { font-size: 10px; color: #666; margin-top: 4px; }
    .round-title { text-align: center; font-weight: 600; margin-bottom: 4px; color: #ccc; }
    </style>
    """

    columns_html = ""
    for round_name in round_names:
        matches_html = "".join(match_box_html(m) for m in results[round_name])
        columns_html += f"""
        <div class="round-col">
          <div class="round-title">{round_name}</div>
          {matches_html}
        </div>
        """

    full_html = f"""
    <html>
    <head>{css}</head>
    <body style="margin:0; background:#0e1117;">
      <div class="bracket">{columns_html}</div>
    </body>
    </html>
    """
    components.html(full_html, height=650, scrolling=True)


def status_badge(status):
    return {
        "played": "✅ Played",
        "scheduled": "📅 Scheduled",
        "simulated": "🔮 Simulated",
    }[status]


def render_match(m):
    prob_a = m["prob_a"] if m["prob_a"] is not None else (100 if m["winner"] == m["team_a"] else 0)
    prob_b = m["prob_b"] if m["prob_b"] is not None else (100 if m["winner"] == m["team_b"] else 0)

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            bold_a = "**" if m["winner"] == m["team_a"] else ""
            st.markdown(f"{bold_a}{m['team_a']}{bold_a}")
        with col2:
            st.markdown("<div style='text-align:center; color:gray'>vs</div>", unsafe_allow_html=True)
        with col3:
            bold_b = "**" if m["winner"] == m["team_b"] else ""
            st.markdown(f"{bold_b}{m['team_b']}{bold_b}")

        bar_html = f"""
        <div style='width:100%; height:14px; border-radius:7px; overflow:hidden; display:flex; margin:4px 0'>
            <div style='width:{prob_a}%; background:#2ecc71; height:100%'></div>
            <div style='width:{prob_b}%; background:#e74c3c; height:100%'></div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        st.caption(
            f"{m['date'].date()}  |  {status_badge(m['status'])}  |  "
            f"Advancing: **{m['winner']}**"
        )
        st.divider()


st.set_page_config(page_title="WC 2026 Bracket", page_icon="🏆", layout="wide")
st.title("🏆 FIFA World Cup 2026 — Full Bracket Predictions")
st.caption("Round of 32 through the Final. Real results and scheduled fixtures are used where available, remaining matches are predicted.")

clf_pipeline, clf_classes, feature_cols = load_model()

with st.sidebar:
    st.header("Data")
    refresh = st.button("🔄 Refresh latest results")
    refresh_token = st.session_state.get("refresh_token", 0)
    if refresh:
        refresh_token += 1
        st.session_state["refresh_token"] = refresh_token

results, champion = run_bracket(clf_pipeline, clf_classes, feature_cols, st.session_state.get("refresh_token", 0))

st.success(f"Predicted 2026 World Cup champion: **{champion}**")

tabs = st.tabs(["Full Bracket", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"])

with tabs[0]:
    render_full_bracket(results)

for tab, round_name in zip(tabs[1:], results.keys()):
    with tab:
        for m in results[round_name]:
            render_match(m)