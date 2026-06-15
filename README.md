# ⚽ FIFA World Cup 2026 Match Predictor

A machine learning model that predicts World Cup 2026 match outcomes (home win / draw / away win) using historical international results, ELO ratings, rolling form, and head-to-head records. A Streamlit app displays live predictions and probabilities.

---

## How It Works

1. Historical match data is pulled from [martj42/international_results](https://github.com/martj42/international_results)
2. Features are engineered per match (ELO, rolling form, H2H, stage)
3. An XGBoost classifier is trained on pre-2018 data and calibrated on a holdout slice
4. Probabilities are generated for upcoming fixtures

---

## Features Used

| Category | Features |
|---|---|
| ELO | Pre-match home/away ratings, difference |
| Rolling form (5 & 10 games) | Goals for/against, goal diff per team |
| Head-to-head | Win/draw/loss rates, match count |
| Context | Neutral venue, tournament stage |

---

## Project Structure

```
├── wc2026.py            # Feature engineering, training, and prediction pipeline
├── wc2026_dashboard.py  # Streamlit app for displaying predictions
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes wc2026_model.pkl
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/wc2026-predictor.git
cd wc2026-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python model.py
```

This fetches the latest data, trains the model, and saves `wc2026_model.pkl` locally.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

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
- The model uses a time-based train/test split (cutoff: 2018-01-01)
- Probabilities are calibrated using isotonic regression
- `wc2026_model.pkl` is excluded from version control via `.gitignore`

---

## Data Source

[martj42/international_results](https://github.com/martj42/international_results) — open dataset of international football results.

---

## License

MIT
