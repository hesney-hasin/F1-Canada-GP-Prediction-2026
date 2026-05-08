# 🏎️ F1 Canada 2026 Race Predictor

A machine learning predictor for the **2026 Canadian Grand Prix** (May 22, Montreal 🇨🇦) 
built with FastF1, scikit-learn, and real F1 timing data.

## 📊 How It Works

1. Pulls historical Canada GP race + qualifying data (2022–2025) via **FastF1**
2. Engineers features: qualifying pace, grid position, Canada circuit history, 2026 season form
3. Trains an **ensemble model** (Gradient Boosting + Random Forest)
4. Predicts the 2026 finishing order

## 🔧 Tech Stack

- `fastf1` — Official F1 timing API
- `scikit-learn` — ML models (GradientBoosting, RandomForest)
- `pandas` / `numpy` — Data processing
- Python 3.14+

## ⚡ Setup

```bash
git clone https://github.com/YOUR_USERNAME/f1-canada-predictor
cd f1-canada-predictor
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python predictor.py
```

## 📦 Requirements
fastf1
scikit-learn
pandas
numpy
matplotlib
seaborn

text

## 🎯 Features Used

| Feature | Importance | Description |
|---|---|---|
| QualiNorm | 0.355 | Normalized qualifying pace |
| CanadaAvgPos | 0.283 | Driver's avg finish at Canada |
| GridPosition | 0.182 | Starting grid position |
| CanadaBestPos | 0.106 | Best ever finish at Canada |
| Form2026 | 0.074 | Current 2026 championship standing |

## 📅 Best Accuracy

Run **after qualifying on May 24, 2026** — replace the estimated grid 
with real Q3 lap times for maximum prediction accuracy.

## ⚠️ Disclaimer

Predictions are based on historical patterns. F1 races are unpredictable — 
safety cars, weather, and strategy can change everything!

## 👩‍💻 Author

Hasnay Hasin 