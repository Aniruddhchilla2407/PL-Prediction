"""
strength_ratings.py

Computes team attack/defense strength ratings from historical match
data. Includes a time-decay weighted version: instead of a flat
average over a window of games, each match is weighted by
exp(-days_ago / half_life), so recent form counts more than older
results without a hard cutoff. This is the standard fix for a model
being slow to react to in-season swings.
"""

import numpy as np
import pandas as pd


def compute_team_strength(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    df = matches[matches["Season"] == season]
    return _strength_from_matches(df)


def compute_rolling_strength(matches: pd.DataFrame, as_of_date, window_games: int = 38) -> pd.DataFrame:
    df = matches[matches["Date"] < pd.Timestamp(as_of_date)].copy()
    df = df.sort_values("Date")
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    trimmed_frames = []
    for team in teams:
        team_games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        trimmed_frames.append(team_games.tail(window_games))
    trimmed = pd.concat(trimmed_frames).drop_duplicates()
    return _strength_from_matches(trimmed)


def compute_decayed_strength(matches: pd.DataFrame, as_of_date, half_life_days: int = 180,
                              lookback_days: int = 730) -> pd.DataFrame:
    """
    Weight each match by exp(-days_ago / half_life_days) -- a match
    half_life_days old counts for half a full-weight match, one
    2*half_life_days old counts for a quarter, etc. lookback_days
    caps how far back we even look, mostly to keep computation
    reasonable, since very old games get near-zero weight anyway.
    """
    cutoff = pd.Timestamp(as_of_date)
    df = matches[(matches["Date"] < cutoff) & (matches["Date"] >= cutoff - pd.Timedelta(days=lookback_days))].copy()
    days_ago = (cutoff - df["Date"]).dt.days
    df["Weight"] = np.exp(-days_ago / half_life_days)

    league_avg_home_goals = np.average(df["FTHG"], weights=df["Weight"])
    league_avg_away_goals = np.average(df["FTAG"], weights=df["Weight"])

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    rows = []
    for team in teams:
        home_games = df[df["HomeTeam"] == team]
        away_games = df[df["AwayTeam"] == team]

        home_attack = (np.average(home_games["FTHG"], weights=home_games["Weight"]) / league_avg_home_goals
                       if len(home_games) else 1.0)
        home_defense = (np.average(home_games["FTAG"], weights=home_games["Weight"]) / league_avg_away_goals
                        if len(home_games) else 1.0)
        away_attack = (np.average(away_games["FTAG"], weights=away_games["Weight"]) / league_avg_away_goals
                       if len(away_games) else 1.0)
        away_defense = (np.average(away_games["FTHG"], weights=away_games["Weight"]) / league_avg_home_goals
                        if len(away_games) else 1.0)

        rows.append({
            "Team": team, "HomeAttack": home_attack, "HomeDefense": home_defense,
            "AwayAttack": away_attack, "AwayDefense": away_defense,
        })
    return pd.DataFrame(rows)


def _strength_from_matches(df: pd.DataFrame) -> pd.DataFrame:
    league_avg_home_goals = df["FTHG"].mean()
    league_avg_away_goals = df["FTAG"].mean()
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    rows = []
    for team in teams:
        home_games = df[df["HomeTeam"] == team]
        away_games = df[df["AwayTeam"] == team]
        home_attack = home_games["FTHG"].mean() / league_avg_home_goals if len(home_games) else 1.0
        home_defense = home_games["FTAG"].mean() / league_avg_away_goals if len(home_games) else 1.0
        away_attack = away_games["FTAG"].mean() / league_avg_away_goals if len(away_games) else 1.0
        away_defense = away_games["FTHG"].mean() / league_avg_home_goals if len(away_games) else 1.0
        rows.append({
            "Team": team, "HomeAttack": home_attack, "HomeDefense": home_defense,
            "AwayAttack": away_attack, "AwayDefense": away_defense,
        })
    return pd.DataFrame(rows)


def expected_goals(strength: pd.DataFrame, home_team: str, away_team: str,
                    league_avg_home_goals: float, league_avg_away_goals: float):
    h = strength[strength["Team"] == home_team].iloc[0]
    a = strength[strength["Team"] == away_team].iloc[0]
    home_xg = league_avg_home_goals * h["HomeAttack"] * a["AwayDefense"]
    away_xg = league_avg_away_goals * a["AwayAttack"] * h["HomeDefense"]
    return home_xg, away_xg