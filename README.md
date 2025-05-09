
# MLB Home Run Prop Model Dashboard

This is a complete Streamlit dashboard application that predicts MLB player home run probabilities using Statcast data and overlays real sportsbook odds to calculate betting value.

## Features
- Predictive modeling with XGBoost
- Rolling HR metrics from Statcast
- Sportsbook odds integration via The Odds API
- Value edge calculations (model prob - implied prob)
- Streamlit dashboard with filtering and CSV export

## How to Deploy (Streamlit Cloud)

1. Sign up for a free account at [streamlit.io](https://streamlit.io)
2. Fork or upload this project to a public GitHub repository
3. In Streamlit Cloud:
   - Click **New App**
   - Connect your GitHub
   - Select the repository and set the main file as `Complete_HR_Prop_Model_Dashboard.py`
4. Add your API keys as environment secrets in Streamlit:
   - `ODDS_API_KEY`
   - `WEATHER_API_KEY`
5. Click **Deploy**

Enjoy your daily HR prop prediction dashboard!

---

## Credits
Developed using Python, Streamlit, pybaseball, and real-time sports/weather APIs.
