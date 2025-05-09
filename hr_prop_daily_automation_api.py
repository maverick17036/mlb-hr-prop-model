
# DAILY HR PROP MODEL AUTOMATION SCRIPT (NO PYBASEBALL)

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- API KEY ---
ODDS_API_KEY = "8e075e101b8f14749df689272a1168e3"

# --- PLAYER LIST ---
players = {
    "Shohei Ohtani": 660271,
    "Aaron Judge": 592450,
    "Pete Alonso": 624413,
    "Yordan Alvarez": 670541,
    "Kyle Schwarber": 656941
}

# --- FETCH GAME LOGS FROM MLB STATS API ---
def fetch_game_logs(player_id):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=hitting&season=2024"
    res = requests.get(url)
    if res.status_code != 200:
        return pd.DataFrame()
    data = res.json()
    games = data.get("stats", [])[0].get("splits", [])
    rows = []
    for g in games:
        row = g["stat"]
        row["game_date"] = g["date"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["home_run"] = df["homeRuns"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

# --- PROCESS + MODEL ---
def prepare_data(df, player_name):
    df = df.copy()
    df["player"] = player_name
    df = df.sort_values("game_date")
    df["hr_5g_avg"] = df["home_run"].rolling(5).mean().fillna(0)
    df["hr_10g_avg"] = df["home_run"].rolling(10).mean().fillna(0)
    return df

def train_model(df):
    features = ["hr_5g_avg", "hr_10g_avg"]
    X = df[features]
    y = df["home_run"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    df["predicted_hr_prob"] = model.predict_proba(X)[:, 1]
    return df

# --- FETCH ODDS ---
def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=player_home_run&apiKey={ODDS_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

def match_odds(df, odds_data):
    results = []
    for player in df["player"].unique():
        row = df[df["player"] == player].iloc[-1]
        prob = row["predicted_hr_prob"]
        name_match = next((o for o in odds_data if player.lower() in o["name"].lower()), None)
        if name_match:
            try:
                decimal_odds = float(name_match["bookmakers"][0]["markets"][0]["outcomes"][0]["price"])
                implied_prob = 1 / decimal_odds
                value_edge = prob - implied_prob
                results.append({
                    "player": player,
                    "predicted_prob": round(prob, 4),
                    "odds": decimal_odds,
                    "implied_prob": round(implied_prob, 4),
                    "value_edge": round(value_edge, 4)
                })
            except:
                continue
    return pd.DataFrame(results)

def save_results(df):
    today = datetime.today().strftime('%Y-%m-%d')
    df.to_csv(f"daily_hr_prop_predictions_{today}.csv", index=False)
    print(f"Saved daily predictions to daily_hr_prop_predictions_{today}.csv")

# === RUN SCRIPT ===
all_data = []
for name, pid in players.items():
    logs = fetch_game_logs(pid)
    if not logs.empty:
        data = prepare_data(logs, name)
        all_data.append(data)

if all_data:
    df_all = pd.concat(all_data)
    df_all = train_model(df_all)
    odds = fetch_odds()
    results = match_odds(df_all, odds)
    save_results(results)
else:
    print("No data available.")
