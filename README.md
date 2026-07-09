# ⚽ FIFA World Cup 2026 Match Predictor

A machine learning model that predicts World Cup 2026 match outcomes using historical international results, ELO ratings, rolling form, and head to head records. Streamlit apps display live predictions and probabilities for both group stage and knockout stage matches.

**Live demo:** [world-cup-2026-predictions.streamlit.app](https://world-cup-2026-predictions-guyjlhmboyerutqpylct88.streamlit.app/)

---

## How It Works

1. Historical match data is pulled from [martj42/international_results](https://github.com/martj42/international_results)
2. Features are engineered per match (ELO, rolling form, H2H, stage)
3. An XGBoost classifier is trained on pre 2018 data and calibrated on a holdout slice
4. Probabilities are generated for upcoming fixtures

Two model variants are included:

- **Group stage / general model** (`wc2026.py`): three way classifier (home win, draw, away win)
- **Knockout stage model** (`wc2026_knockout.py`): binary classifier (home win, away win). Draws are dropped from training since a real knockout match always ends in a winner after extra time or penalties.

On top of the knockout model sits a **full bracket simulation** (`wc2026_bracket_simulation.py`), which walks the actual Round of 32 to Final bracket:

- The Round of 32 pairings are hardcoded, since bracket topology cannot be derived from a results only dataset
- For each matchup, it checks history first. If the match already happened, the real result decides the winner
- If not played but already scheduled, the real fixture is predicted
- If not yet in the data at all, a fixture is simulated on the expected venue date using the winners already resolved from earlier rounds
- This repeats round by round (Round of 32 → Round of 16 → Quarterfinal → Semifinal → Final) until a predicted champion comes out the other end

---

## Features Used

| Category                    | Features                                |
| --------------------------- | --------------------------------------- |
| ELO                         | Pre match home/away ratings, difference |
| Rolling form (5 & 10 games) | Goals for/against, goal diff per team   |
| Head to head                | Win/draw/loss rates, match count        |
| Context                     | Neutral venue, tournament stage         |

---

## Project Structure

```
├── wc2026.py                            # Feature engineering, training, and prediction pipeline (group stage, 3-way)
├── wc2026_dashboard.py                  # Streamlit app for group stage predictions
├── wc2026_knockout.py                   # Binary (no-draw) model for knockout stage matches, built on wc2026.py
├── wc2026_knockout_dashboard.py         # Streamlit app for knockout stage predictions
├── wc2026_bracket_simulation.py         # Walks the real Round of 32 to Final bracket using the knockout model, prints results to console
├── wc2026_bracket_simulation_dashboard.py  # Streamlit app version of the bracket simulation
├── requirements.txt                     # Python dependencies
├── LICENSE
└── README.md
```

`wc2026_knockout.py`, `wc2026_bracket_simulation.py`, and `wc2026_bracket_simulation_dashboard.py` all import shared data loading and feature engineering functions from `wc2026.py`, so `wc2026.py` must be present in the same folder. The two bracket simulation files also depend on `wc2026_knockout_model.pkl`, so train the knockout model first.

---

## Getting Started

### 1. Clone the repo

```
git clone https://github.com/Shyam-Parmar/World-Cup-2026-Predictions.git
cd World-Cup-2026-Predictions
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train a model

Group stage model (home win / draw / away win):

```
python wc2026.py
```

Knockout stage model (home win / away win, no draw):

```
python wc2026_knockout.py
```

Each script fetches the latest data, trains its model, and saves a `.pkl` file locally (`wc2026_model.pkl` or `wc2026_knockout_model.pkl`).

### 4. Run a Streamlit app

Group stage dashboard:

```
streamlit run wc2026_dashboard.py
```

Knockout stage dashboard:

```
streamlit run wc2026_knockout_dashboard.py
```

The knockout dashboard trains fresh on startup, then shows upcoming fixtures, played matches with prediction accuracy, and overall model summary across three tabs.

### 5. Run the full bracket simulation

Console version:

```
python wc2026_bracket_simulation.py
```

This trains (or loads) `wc2026_knockout_model.pkl`, then walks every round from Round of 32 to the Final, printing each matchup, win probabilities, and the resolved winner, ending with a predicted champion.

Dashboard version:

```
streamlit run wc2026_bracket_simulation_dashboard.py
```

Same bracket walk logic, displayed as an interactive bracket in the browser instead of console output.

---

## Requirements

- Python 3.9+
- xgboost
- scikit-learn
- pandas
- numpy
- matplotlib
- streamlit

---

## Notes

- Friendlies are excluded from training data
- Both models use a time based train/test split (cutoff: 2018-01-01)
- Probabilities are calibrated using isotonic regression
- The knockout model drops all regular time draws before training, since knockout matches are decided by extra time or penalties
- The bracket simulation hardcodes only the Round of 32 pairings, since bracket shape is not present in a results only dataset. Everything else (results, fixtures, predictions) stays dynamic
- Model `.pkl` files are excluded from version control via `.gitignore`

---

## Data Source

[martj42/international_results](https://github.com/martj42/international_results) — open dataset of international football results.

---

## License

MIT
