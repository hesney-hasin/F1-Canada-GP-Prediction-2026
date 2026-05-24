# STEP 1: Setup
import warnings
warnings.filterwarnings("ignore")
import fastf1
import pandas as pd
import numpy as np
import os

os.makedirs("./f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("./f1_cache")

#STEP 2: Race Results
all_results = []
for year in [2022, 2023, 2024, 2025]:
    print(f" Loading {year} Canada GP race...")
    session = fastf1.get_session(year, "Canada", "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results[["Abbreviation","TeamName","GridPosition","Position","Points"]].copy()
    results = results.rename(columns={"Abbreviation": "Driver"})
    results["Year"]         = year
    results["Position"]     = pd.to_numeric(results["Position"],     errors="coerce")
    results["GridPosition"] = pd.to_numeric(results["GridPosition"], errors="coerce")
    results["Points"]       = pd.to_numeric(results["Points"],       errors="coerce")
    all_results.append(results)
df_race = pd.concat(all_results, ignore_index=True)
print(f" Race data: {len(df_race)} rows")

#STEP 3: Qualifying Times
all_quali = []
for year in [2022, 2023, 2024, 2025]:
    print(f" Loading {year} Canada GP qualifying...")
    session = fastf1.get_session(year, "Canada", "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    best_laps = (
        session.laps.pick_quicklaps()
        .groupby("Driver")["LapTime"].min()
        .reset_index()
        .rename(columns={"LapTime": "BestQualiTime"})
    )
    best_laps["QualiTime_s"] = best_laps["BestQualiTime"].dt.total_seconds()
    best_laps["Year"] = year
    all_quali.append(best_laps)
df_quali = pd.concat(all_quali, ignore_index=True)
print(f" Qualifying data: {len(df_quali)} rows")

# STEP 4: Merge & Feature Engineering
# 4a. Merging race + qualifying
df = df_race.merge(df_quali[["Driver","Year","QualiTime_s"]],
                 on=["Driver","Year"], how="left")

# 4b. Normalize qualifying time within each year
df["QualiNorm"] = df.groupby("Year")["QualiTime_s"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# 4c. Canada circuit history — prior years only (no leakage)
records = []
for _, row in df.iterrows():
    prior = df[(df["Driver"] == row["Driver"]) & (df["Year"] < row["Year"])]
    records.append({
        "Driver":       row["Driver"],
        "Year":         row["Year"],
        "CanadaAvgPos": prior["Position"].mean() if len(prior) > 0 else np.nan,
        "CanadaBestPos":prior["Position"].min()  if len(prior) > 0 else np.nan,
        "CanadaRaces":  len(prior),
    })
prior_hist = pd.DataFrame(records)
df = df.merge(prior_hist, on=["Driver","Year"], how="left")
df["CanadaAvgPos"]  = df["CanadaAvgPos"].fillna(12.0)
df["CanadaBestPos"] = df["CanadaBestPos"].fillna(12.0)

# Full history kept separately for 2026 prediction only
canada_hist = (
    df.groupby("Driver")
    .agg(CanadaAvgPos=("Position","mean"), CanadaBestPos=("Position","min"), CanadaRaces=("Year","count"))
    .reset_index()
)

# 4d. 2026 standings
standings_2026 = {
    "ANT": 1,  "RUS": 2,  "LEC": 3,  "NOR": 4,  "HAM": 5,
    "PIA": 6,  "VER": 7,  "BEA": 8,  "GAS": 9,  "LAW": 10,
    "COL": 11, "LIN": 12, "HAD": 13, "SAI": 14, "BOR": 15,
    "OCO": 16, "ALB": 17, "HUL": 18, "BOT": 19, "PER": 20,
    "ALO": 21, "STR": 22,
}
df["Form2026"] = df["Driver"].map(standings_2026)
df["Form2026"] = df["Form2026"].fillna(15)

df_clean = df.dropna(subset=["Position","GridPosition","QualiTime_s"])
print(f" Features ready! {len(df_clean)} clean rows")
print(df_clean[["Driver","Year","GridPosition","QualiTime_s","QualiNorm","CanadaAvgPos","Form2026","Position"]].head(8).to_string(index=False))

# STEP 5: Training the ML Model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Define features and target
FEATURES = ["GridPosition", "QualiNorm", "CanadaAvgPos", "CanadaBestPos", "Form2026"]
TARGET   = "Position"

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

print(f" Training on {len(X)} samples with {len(FEATURES)} features")
print(f"   Features: {FEATURES}\n")

# Model 1: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X, y)
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
groups = df_clean["Year"].values
gb_scores = cross_val_score(gb_model, X, y, cv=logo, groups=groups, scoring="neg_mean_absolute_error")
print(f" Gradient Boosting — MAE: {-gb_scores.mean():.2f} positions (±{gb_scores.std():.2f})")

# Model 2: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X, y)
rf_scores = cross_val_score(rf_model, X, y, cv=logo, groups=groups, scoring="neg_mean_absolute_error")
print(f" Random Forest     — MAE: {-rf_scores.mean():.2f} positions (±{rf_scores.std():.2f})")

# ── Feature importance (from Gradient Boosting)
print(f"\nFeature Importance:")
for feat, imp in sorted(zip(FEATURES, gb_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {bar} {imp:.3f}")
    
# STEP 6: Predicting 2026 Canada GP

QUALI_MODE = "post"

# Full 2022–2025 history for 2026 prediction
canada_hist_full = (
    df.groupby("Driver")
    .agg(CanadaAvgPos=("Position","mean"), CanadaBestPos=("Position","min"), CanadaRaces=("Year","count"))
    .reset_index()
)

if QUALI_MODE == "post":
    print("  Fetching 2026 Canada qualifying session...")
    quali_live = fastf1.get_session(2026, "Canada", "Q")
    quali_live.load(laps=True, telemetry=False, weather=False, messages=False)
    best_q = (
        quali_live.laps.pick_quicklaps()
        .groupby("Driver")["LapTime"].min()
        .reset_index()
        .rename(columns={"LapTime": "BestQualiTime"})
    )
    best_q["QualiTime_s"] = best_q["BestQualiTime"].dt.total_seconds()
    grid_2026 = quali_live.results[["Abbreviation","TeamName"]].copy()
    grid_2026 = grid_2026.rename(columns={"Abbreviation":"Driver"})
    grid_2026 = grid_2026.merge(best_q[["Driver","QualiTime_s"]], on="Driver", how="left")

    # ── ADD HERE ──────────────────────────────────────────────
    grid_2026["GridPosition"] = grid_2026["QualiTime_s"].rank(method="first").astype(float)
    grid_2026["GridPosition"] = grid_2026["GridPosition"].fillna(float(len(grid_2026)))
    # ──────────────────────────────────────────────────────────

    qt_min = grid_2026["QualiTime_s"].min()
    qt_max = grid_2026["QualiTime_s"].max()
    grid_2026["QualiNorm"] = (grid_2026["QualiTime_s"] - qt_min) / (qt_max - qt_min)
    mode_label = "POST-QUALIFYING (real Q3 lap times)"

else:
    grid_2026 = pd.DataFrame({
        "Driver": ["ANT","RUS","LEC","NOR","HAM","PIA","VER","BEA","GAS","LAW",
                   "COL","LIN","HAD","SAI","BOR","OCO","ALB","HUL","BOT","PER","ALO","STR"],
        "GridPosition": list(range(1, 23)),
        "TeamName": [
            "Mercedes","Mercedes","Ferrari","McLaren","Ferrari","McLaren",
            "Red Bull Racing","Haas F1 Team","Alpine","Racing Bulls",
            "Alpine","Racing Bulls","Red Bull Racing","Williams","Audi",
            "Haas F1 Team","Williams","Audi","Cadillac","Cadillac",
            "Aston Martin","Aston Martin"
        ]
    })
    grid_2026["QualiNorm"] = 0.5
    mode_label = "PRE-QUALIFYING ⏳ (estimated grid)"

grid_2026 = grid_2026.merge(canada_hist_full, on="Driver", how="left")
grid_2026["CanadaAvgPos"]  = grid_2026["CanadaAvgPos"].fillna(12.0)
grid_2026["CanadaBestPos"] = grid_2026["CanadaBestPos"].fillna(12.0)
grid_2026["Form2026"]      = grid_2026["Driver"].map(standings_2026).fillna(15)

X_pred = grid_2026[FEATURES].values

grid_2026["GB_Pred"]  = gb_model.predict(X_pred)
grid_2026["RF_Pred"]  = rf_model.predict(X_pred)
grid_2026["Ensemble"] = grid_2026["GB_Pred"] * 0.6 + grid_2026["RF_Pred"] * 0.4
grid_2026 = grid_2026.sort_values("Ensemble").reset_index(drop=True)
grid_2026["PredictedPos"] = range(1, len(grid_2026) + 1)
print("\n" + "=" * 55)
print("  🏁  PREDICTED 2026 CANADA GP RESULT")
print(f"  {mode_label}")
print("=" * 55)
print(f"  {'Pos':<4} {'Driver':<6} {'Team':<22} {'Score'}")
print("  " + "-" * 45)

medals = {1:"🥇", 2:"🥈", 3:"🥉"}
for _, row in grid_2026.head(10).iterrows():
    pos  = int(row["PredictedPos"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon}  {row['Driver']:<6} {str(row['TeamName']):<22} {row['Ensemble']:.2f}")

print("=" * 55)
