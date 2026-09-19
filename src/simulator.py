"""
simulator.py

Poisson match simulator + Monte Carlo season simulation, vectorized
with numpy.
"""

import numpy as np
import pandas as pd


def current_standings(played_matches: pd.DataFrame) -> pd.Series:
    points = {}
    for _, row in played_matches.iterrows():
        if row["FTHG"] > row["FTAG"]:
            home_pts, away_pts = 3, 0
        elif row["FTHG"] < row["FTAG"]:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1
        points[row["HomeTeam"]] = points.get(row["HomeTeam"], 0) + home_pts
        points[row["AwayTeam"]] = points.get(row["AwayTeam"], 0) + away_pts
    return pd.Series(points)


def run_monte_carlo(played_matches: pd.DataFrame, remaining_fixtures: pd.DataFrame,
                     strength: pd.DataFrame, league_avg_home_goals: float,
                     league_avg_away_goals: float, n_simulations: int = 10000):
    """Returns (points_matrix, all_teams) -- points_matrix has shape
    (n_simulations, n_teams), raw final points for every simulated season."""

    base_points_series = current_standings(played_matches)
    all_teams = list(pd.unique(pd.concat([
        played_matches["HomeTeam"], played_matches["AwayTeam"],
        remaining_fixtures["HomeTeam"], remaining_fixtures["AwayTeam"],
    ])))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    n_teams = len(all_teams)

    base_points = np.zeros(n_teams)
    for team, pts in base_points_series.items():
        base_points[team_to_idx[team]] = pts

    strength_dict = strength.set_index("Team").to_dict("index")

    home_idx, away_idx, home_xg, away_xg = [], [], [], []
    for _, fx in remaining_fixtures.iterrows():
        h, a = fx["HomeTeam"], fx["AwayTeam"]
        h_s, a_s = strength_dict[h], strength_dict[a]
        home_idx.append(team_to_idx[h])
        away_idx.append(team_to_idx[a])
        home_xg.append(league_avg_home_goals * h_s["HomeAttack"] * a_s["AwayDefense"])
        away_xg.append(league_avg_away_goals * a_s["AwayAttack"] * h_s["HomeDefense"])

    home_idx, away_idx = np.array(home_idx), np.array(away_idx)
    home_xg, away_xg = np.array(home_xg), np.array(away_xg)
    n_fixtures = len(home_idx)

    home_goals = np.random.poisson(home_xg[None, :], size=(n_simulations, n_fixtures))
    away_goals = np.random.poisson(away_xg[None, :], size=(n_simulations, n_fixtures))

    home_pts = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
    away_pts = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))

    points = np.tile(base_points, (n_simulations, 1))
    for f in range(n_fixtures):
        points[:, home_idx[f]] += home_pts[:, f]
        points[:, away_idx[f]] += away_pts[:, f]

    return points, all_teams


def probability_table(points: np.ndarray, all_teams: list) -> pd.DataFrame:
    n_simulations, n_teams = points.shape
    order = np.argsort(-points, axis=1)

    title_count = np.bincount(order[:, 0], minlength=n_teams)
    top4_count = np.bincount(order[:, :4].ravel(), minlength=n_teams)
    relegation_count = np.bincount(order[:, -3:].ravel(), minlength=n_teams)

    results = pd.DataFrame({
        "Team": all_teams,
        "TitleProbability": title_count / n_simulations,
        "Top4Probability": top4_count / n_simulations,
        "RelegationProbability": relegation_count / n_simulations,
    })
    return results.sort_values("TitleProbability", ascending=False).reset_index(drop=True)


def predicted_table(points: np.ndarray, all_teams: list) -> pd.DataFrame:
    """A single deterministic final table: average points per team
    across all simulations, ranked -- this is 'the prediction' rather
    than a probability distribution."""
    mean_points = points.mean(axis=0)
    df = pd.DataFrame({"Team": all_teams, "PredictedPoints": mean_points})
    df = df.sort_values("PredictedPoints", ascending=False).reset_index(drop=True)
    df["PredictedRank"] = df.index + 1

    def outcome(rank):
        if rank == 1:
            return "Champion"
        elif rank <= 4:
            return "Top 4"
        elif rank > len(df) - 3:
            return "Relegated"
        return "Mid-table"

    df["Outcome"] = df["PredictedRank"].apply(outcome)
    return df


def simulate_season(played_matches, remaining_fixtures, strength,
                     league_avg_home_goals, league_avg_away_goals, n_simulations=10000):
    """Kept for backward compatibility -- returns the probability table."""
    points, all_teams = run_monte_carlo(played_matches, remaining_fixtures, strength,
                                         league_avg_home_goals, league_avg_away_goals, n_simulations)
    return probability_table(points, all_teams)