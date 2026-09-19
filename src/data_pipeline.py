"""
Premier League data pipeline for the StepOut portfolio project.

Step 1: Download 7 seasons (2019/20 - 2025/26) of match data from
        football-data.co.uk (mirrored on datahub.io with stable URLs).
Step 2: Merge into one clean DataFrame with a Season column.
Step 3: Compute rolling attack/defense strength ratings per team, the
        core input to a Dixon-Coles / Poisson match simulator.

Run with: python pl_pipeline.py
Requires: pandas, requests  (pip install pandas requests)
"""

import pandas as pd
import requests
from io import StringIO

# --- Step 1: download -------------------------------------------------

SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
BASE_URL = "https://datahub.io/football/english-premier-league/_r/-/season-{}.csv"

def fetch_season(season_code: str) -> pd.DataFrame:
    url = BASE_URL.format(season_code)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # Human-readable season label, e.g. "1920" -> "2019/20"
    yr1 = 2000 + int(season_code[:2])
    yr2_suffix = season_code[2:]
    df["Season"] = f"{yr1}/{yr2_suffix}"
    return df

def build_dataset() -> pd.DataFrame:
    frames = []
    for code in SEASONS:
        print(f"Fetching season {code}...")
        frames.append(fetch_season(code))
    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["Date"] = pd.to_datetime(all_matches["Date"])
    all_matches = all_matches.sort_values("Date").reset_index(drop=True)
    return all_matches

# --- Step 2: team strength ratings --------------------------------------
# Simple Dixon-Coles-style attack/defense strength, relative to league
# average goals scored/conceded, split by home/away. This is the
# standard input to a Poisson match simulator: expected goals for a
# fixture = league_avg_goals * attack_strength_A * defense_weakness_B.

def compute_team_strength(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Compute attack/defense strength for one season using that
    season's matches only. Call this once per season, or restrict to a
    trailing window (e.g. last 38 games) for an in-season rolling version."""
    df = matches[matches["Season"] == season]

    league_avg_home_goals = df["FTHG"].mean()
    league_avg_away_goals = df["FTAG"].mean()

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    rows = []
    for team in teams:
        home_games = df[df["HomeTeam"] == team]
        away_games = df[df["AwayTeam"] == team]

        home_attack = home_games["FTHG"].mean() / league_avg_home_goals
        home_defense = home_games["FTAG"].mean() / league_avg_away_goals
        away_attack = away_games["FTAG"].mean() / league_avg_away_goals
        away_defense = away_games["FTHG"].mean() / league_avg_home_goals

        rows.append({
            "Team": team,
            "Season": season,
            "HomeAttack": home_attack,     # >1 = scores more than average at home
            "HomeDefense": home_defense,   # >1 = concedes more than average at home
            "AwayAttack": away_attack,
            "AwayDefense": away_defense,
        })
    return pd.DataFrame(rows)

# --- Step 3: expected goals for a fixture -------------------------------

def expected_goals(strength: pd.DataFrame, home_team: str, away_team: str,
                    league_avg_home_goals: float, league_avg_away_goals: float):
    h = strength[strength["Team"] == home_team].iloc[0]
    a = strength[strength["Team"] == away_team].iloc[0]
    home_xg = league_avg_home_goals * h["HomeAttack"] * a["AwayDefense"]
    away_xg = league_avg_away_goals * a["AwayAttack"] * h["HomeDefense"]
    return home_xg, away_xg

if __name__ == "__main__":
    matches = build_dataset()
    matches.to_csv("pl_2019_2026_matches.csv", index=False)
    print(f"Saved {len(matches)} matches across {matches['Season'].nunique()} seasons "
          f"to pl_2019_2026_matches.csv")

    # Example: strength ratings for the current (or most recent) season
    all_strength = []
    for season in matches["Season"].unique():
        strength = compute_team_strength(matches, season)
        all_strength.append(strength)
    team_strength = pd.concat(all_strength, ignore_index=True)
    team_strength.to_csv("team_strength_all_seasons.csv", index=False)
    print("\nTeam strength ratings:")
    print(team_strength.to_string(index=False))