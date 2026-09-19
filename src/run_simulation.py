"""
run_simulation.py

Ties the pipeline together end-to-end:
  1. Load full historical matches (2019/20-2025/26) + this season's
     played matches (2026/27 so far).
  2. Compute rolling team strength as of today, using that combined
     history.
  3. Patch in a fallback rating for any team with no history at all
     (newly promoted sides) -- discounted below league average,
     reflecting the historical base rate that promoted teams tend
     to underperform average, rather than assuming they're exactly
     average until real 2026/27 data accumulates.
  4. Simulate the rest of the 2026/27 season thousands of times.
  5. Save both the probability table and a single predicted final
     table.
"""

import pandas as pd
from strength_ratings import compute_decayed_strength
from simulator import run_monte_carlo, probability_table, predicted_table

HIST_MATCHES_PATH = "data/raw/pl_2019_2026_matches.csv"
CURRENT_PLAYED_PATH = "data/raw/pl_2026_27_played.csv"
REMAINING_FIXTURES_PATH = "data/raw/pl_2026_27_remaining.csv"
PROBABILITY_OUTPUT_PATH = "outputs/season_forecast.csv"
PREDICTED_TABLE_OUTPUT_PATH = "outputs/predicted_final_table.csv"

COLUMNS_NEEDED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

PROMOTED_DISCOUNT = 0.85  # tune this based on your backtest results


def load_combined_history() -> pd.DataFrame:
    historical = pd.read_csv(HIST_MATCHES_PATH, parse_dates=["Date"])[COLUMNS_NEEDED]
    current = pd.read_csv(CURRENT_PLAYED_PATH, parse_dates=["Date"])[COLUMNS_NEEDED]
    combined = pd.concat([historical, current], ignore_index=True)
    return combined.sort_values("Date").reset_index(drop=True)


def fill_missing_teams(strength: pd.DataFrame, all_teams, promoted_discount: float = PROMOTED_DISCOUNT) -> pd.DataFrame:
    """Newly promoted teams (no historical data) get a rating below
    league average by default, reflecting the historical base rate
    that promoted sides tend to underperform average -- not a
    penalty, just a more honest prior than assuming they're exactly
    average until real 2026/27 data accumulates."""
    known = set(strength["Team"])
    missing = [t for t in all_teams if t not in known]
    if missing:
        print(f"No historical data for: {missing} -- using discounted fallback rating ({promoted_discount})")
        fallback_rows = pd.DataFrame([{
            "Team": t,
            "HomeAttack": promoted_discount, "HomeDefense": 1 / promoted_discount,
            "AwayAttack": promoted_discount, "AwayDefense": 1 / promoted_discount,
        } for t in missing])
        strength = pd.concat([strength, fallback_rows], ignore_index=True)
    return strength


if __name__ == "__main__":
    combined_history = load_combined_history()
    current_played = pd.read_csv(CURRENT_PLAYED_PATH, parse_dates=["Date"])
    remaining = pd.read_csv(REMAINING_FIXTURES_PATH, parse_dates=["Date"])

    today = pd.Timestamp.today()
    strength = compute_decayed_strength(combined_history, today, half_life_days=180)

    all_teams = pd.unique(pd.concat([
        current_played["HomeTeam"], current_played["AwayTeam"],
        remaining["HomeTeam"], remaining["AwayTeam"],
    ]))
    strength = fill_missing_teams(strength, all_teams)

    league_avg_home_goals = combined_history["FTHG"].mean()
    league_avg_away_goals = combined_history["FTAG"].mean()

    points_matrix, all_teams = run_monte_carlo(
        played_matches=current_played,
        remaining_fixtures=remaining,
        strength=strength,
        league_avg_home_goals=league_avg_home_goals,
        league_avg_away_goals=league_avg_away_goals,
        n_simulations=10000,
    )

    forecast = probability_table(points_matrix, all_teams)
    forecast.to_csv(PROBABILITY_OUTPUT_PATH, index=False)
    print(forecast.to_string(index=False))

    final_table = predicted_table(points_matrix, all_teams)
    final_table.to_csv(PREDICTED_TABLE_OUTPUT_PATH, index=False)
    print("\nPredicted final table:")
    print(final_table.to_string(index=False))