"""
World Cup 2026 Bracket Predictor (data driven, real bracket shape)

Loads wc2026_knockout_model.pkl and walks the actual bracket from the
Round of 32 to the Final. For any match already played, the real result
is used. For any match already scheduled in the data, it is predicted
directly. For matches not yet in the data, a fixture is simulated using
the winners already predicted.

The only hardcoded part is the bracket topology itself (which slot plays
which), since that information does not exist in a results only dataset.
Team names, dates already in the data, and every prediction stay dynamic.
"""

import os
import sys
import pickle
import pandas as pd

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

# Date window per round, used to find the real result without picking up an
# earlier group stage meeting between the same two teams.
ROUND_WINDOW = {
    "Round of 32": (pd.Timestamp("2026-06-28"), pd.Timestamp("2026-07-03")),
    "Round of 16": (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-07")),
    "Quarterfinal": (pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-11")),
    "Semifinal": (pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")),
    "Final": (pd.Timestamp("2026-07-19"), pd.Timestamp("2026-07-19")),
}

# Bracket topology, verified against FIFA.com and ESPN, July 2026.
# Each Round of 32 slot is identified by its two teams, since that is
# fixed and known regardless of who wins. This is the one part of the
# real bracket shape that cannot be read from a results dataset, so it
# is the one thing hardcoded here.
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

# Round of 16 pairs, by index into R32_SLOTS, in venue order from the bracket.
R16_SLOT_PAIRS = [(0, 1), (2, 3), (8, 9), (10, 11), (4, 5), (6, 7), (12, 13), (14, 15)]
R16_DATES = [
    pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-04"),
    pd.Timestamp("2026-07-05"), pd.Timestamp("2026-07-05"),
    pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-06"),
    pd.Timestamp("2026-07-07"), pd.Timestamp("2026-07-07"),
]

# Quarterfinal pairs, by index into the Round of 16 winners above.
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
QF_DATES = [
    pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-10"),
    pd.Timestamp("2026-07-11"), pd.Timestamp("2026-07-11"),
]

# Semifinal pairs, by index into the Quarterfinal winners above.
SF_PAIRS = [(0, 1), (2, 3)]
SF_DATES = [pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")]

FINAL_DATE = pd.Timestamp("2026-07-19")


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
    """Return the winner of a match between two teams, using the real result if
    it already happened, the real fixture if it is already scheduled, or a
    simulated fixture on the exact venue date if it is not yet in the data."""

    start, end = ROUND_WINDOW[round_name]
    played = df_hist[
        (df_hist["date"] >= start) & (df_hist["date"] <= end) &
        (((df_hist["home_team"] == team_a) & (df_hist["away_team"] == team_b)) |
         ((df_hist["home_team"] == team_b) & (df_hist["away_team"] == team_a)))
    ]
    if not played.empty:
        row = played.iloc[0]
        winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
        print(f"  {row['date'].date()}  {row['home_team']:20s} vs  {row['away_team']:20s}  "
              f"-> {winner} (already played)")
        return winner

    match = df_upcoming[
        ((df_upcoming["home_team"] == team_a) & (df_upcoming["away_team"] == team_b)) |
        ((df_upcoming["home_team"] == team_b) & (df_upcoming["away_team"] == team_a))
    ]
    if not match.empty:
        fixture = match.iloc[[0]]
        note = ""
    else:
        fixture = pd.DataFrame([{
            "date": match_date,
            "home_team": team_a,
            "away_team": team_b,
            "tournament": "FIFA World Cup",
            "neutral": True,
        }])
        note = "  (simulated, not yet in data)"

    preds = predict_fixtures(fixture, df_hist, clf_pipeline, clf_classes, feature_cols)
    row = preds.iloc[0]
    print(f"  {row['date'].date()}  {row['home_team']:20s} {row['prob_home_win']:5.1f}%  vs  "
          f"{row['away_team']:20s} {row['prob_away_win']:5.1f}%  -> {row['predicted_winner']}{note}")
    return row["predicted_winner"]


def main():
    clf_pipeline, clf_classes, feature_cols = load_model()
    df_hist, df_upcoming = wc.load_data(wc.RAW_URL)

    print("\nRound of 32")
    r32_winners = [
        resolve_match(*tuple(slot), "Round of 32", ROUND_DATE["Round of 32"], df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols)
        for slot in R32_SLOTS
    ]

    print("\nRound of 16")
    r16_winners = [
        resolve_match(r32_winners[i], r32_winners[j], "Round of 16", R16_DATES[k], df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols)
        for k, (i, j) in enumerate(R16_SLOT_PAIRS)
    ]

    print("\nQuarterfinal")
    qf_winners = [
        resolve_match(r16_winners[i], r16_winners[j], "Quarterfinal", QF_DATES[k], df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols)
        for k, (i, j) in enumerate(QF_PAIRS)
    ]

    print("\nSemifinal")
    sf_winners = [
        resolve_match(qf_winners[i], qf_winners[j], "Semifinal", SF_DATES[k], df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols)
        for k, (i, j) in enumerate(SF_PAIRS)
    ]

    print("\nFinal")
    champion = resolve_match(sf_winners[0], sf_winners[1], "Final", FINAL_DATE, df_hist, df_upcoming, clf_pipeline, clf_classes, feature_cols)

    print(f"\nPredicted champion: {champion}")


if __name__ == "__main__":
    main()