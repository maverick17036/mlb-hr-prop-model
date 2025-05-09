
# COMPLETE STREAMLIT DASHBOARD FOR DAILY MLB HR PROP MODEL

import streamlit as st
import pandas as pd
import numpy as np
import requests
from pybaseball import statcast_batter_game_logs, playerid_lookup
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from datetime import datetime

# === API KEYS ===
ODDS_API_KEY = "8e075e101b8f14749df689272a1168e3"
WEATHER_API_KEY = "mOmfXqJsMkG1C7Tr9mlTkAAOVAqnJn5q"

# === PLAYER SETUP ===
top_players = ['Shohei Ohtani', 'Aaron Judge', 'Pete Alonso', 'Yordan Alvarez', 'Kyle Schwarber']

@st.cache_data
def get_player_ids(names):
    ids = {}
    for name in names:
        try:
            first, last = name.split(" ", 1)
            result = playerid_lookup(last, first)
            if not result.empty:
                ids[name] = int(result.iloc[0]['key_mlbam'])
        except:
            ids[name] = None
    return ids

@st.cache_data
def get_statcast_logs(player_ids):
    logs = []
    for name, pid in player_ids.items():
        if pid:
            try:
                data = statcast_batter_game_logs(2024, 2025, pid)
                data['player'] = name
                logs.append(data)
            except:
                continue
    return pd.concat(logs)

def prepare_features(df):
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['hr'] = df['home_run'].astype(int)
    df.sort_values(by=['player', 'game_date'], inplace=True)
    df['hr_5g_avg'] = df.groupby('player')['hr'].transform(lambda x: x.rolling(5).mean().fillna(0))
    df['hr_10g_avg'] = df.groupby('player')['hr'].transform(lambda x: x.rolling(10).mean().fillna(0))
    return df

def train_model(df):
    features = ['hr_5g_avg', 'hr_10g_avg']
    X = df[features]
    y = df['hr']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    df['predicted_hr_prob'] = model.predict_proba(X)[:, 1]
    return df, model

def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=player_home_run&apiKey={ODDS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def match_odds_to_model(df, odds_data):
    records = []
    for player in df['player'].unique():
        player_row = df[df['player'] == player].iloc[-1]
        prob = player_row['predicted_hr_prob']
        name_match = next((o for o in odds_data if player.lower() in o['name'].lower()), None)
        if name_match:
            try:
                decimal_odds = float(name_match['bookmakers'][0]['markets'][0]['outcomes'][0]['price'])
                implied_prob = 1 / decimal_odds
                value = prob - implied_prob
                records.append({
                    'player': player,
                    'predicted_prob': round(prob, 4),
                    'odds': decimal_odds,
                    'implied_prob': round(implied_prob, 4),
                    'value_edge': round(value, 4)
                })
            except:
                continue
    return pd.DataFrame(records)

# === STREAMLIT DASHBOARD ===

st.title("MLB Daily Home Run Prop Model")
st.markdown("Integrated with live odds, player stats, and betting value calculations.")

with st.spinner("Loading model..."):
    ids = get_player_ids(top_players)
    logs = get_statcast_logs(ids)
    logs = prepare_features(logs)
    logs, model = train_model(logs)
    odds = fetch_odds()
    final_df = match_odds_to_model(logs, odds)

min_edge = st.slider("Minimum value edge", 0.00, 0.10, 0.02, 0.01)
filtered_df = final_df[final_df['value_edge'] >= min_edge]
st.dataframe(filtered_df.sort_values('value_edge', ascending=False))

st.download_button("Download CSV", filtered_df.to_csv(index=False), "hr_prop_values.csv", "text/csv")
