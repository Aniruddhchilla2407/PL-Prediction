"""
backtest.py

Backtests the model across multiple seasons AND multiple cutoff
points within each season (early/mid/late), rather than one cutoff
per season -- a handful of test cases can't estimate accuracy
precisely, so this multiplies the sample size to make the metrics
mean something.
"""

import pandas as pd
from strength_ratings import compute_decayed_strength
from simulator import run_monte_carlo, predicted_table

HIST_MATCHES_PATH = "data/raw/pl_2019_2026_matches.csv"
BACKTEST_SEASONS = ["2021/22", "2022/23", "2023/24", "2024/25"]
CUTOFF_FRACTIONS = [0.15, 0.35, 0.55]  # fraction of the season played before predicting the rest
PROMOTED_DISCOUNT = 0.85


def actual_final_outcome(all_matches: pd.DataFrame, season: str):
    season_matches = all_matches[all_matches["Season"] == season]
    points = {}
    for _, row in season_matches.iterrows():
        if row["FTHG"] > row["FTAG"]:
            h, a = 3, 0
        elif row["FTHG"] < row["FTAG"]:
            h, a = 0, 3
        else:
            h, a = 1, 1
        points[row["HomeTeam"]] = points.get(row["HomeTeam"], 0) + h
        points[row["AwayTeam"]] = points.get(row["AwayTeam"], 0) + a
    ranked = pd.Series(points).sort_values(ascending=False)
    return {
        "champion": ranked.index[0],
        "top4": set(ranked.index[:4]),
        "relegated": set(ranked.index[-3:]),
    }


def prior_season_of(season: str, all_seasons: list) -> str:
    idx = all_seasons.index(season)
    return all_seasons[idx - 1]


def fill_missing_teams(strength: pd.DataFrame, all_teams, promoted_discount: float = PROMOTED_DISCOUNT) -> pd.DataFrame:
    known = set(strength["Team"])
    missing = [t for t in all_teams if t not in known]
    if missing:
        fallback_rows = pd.DataFrame([{
            "Team": t,
            "HomeAttack": promoted_discount, "HomeDefense": 1 / promoted_discount,
            "AwayAttack": promoted_discount, "AwayDefense": 1 / promoted_discount,
        } for t in missing])
        strength = pd.concat([strength, fallback_rows], ignore_index=True)
    return strength


def run_backtest():
    all_matches = pd.read_csv(HIST_MATCHES_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    all_seasons = sorted(all_matches["Season"].unique(), key=lambda s: s[:4])

    champion_hits = 0
    top4_recalls, relegation_recalls = [], []
    n_cases = 0

    for season in BACKTEST_SEASONS:
        prior_season = prior_season_of(season, all_seasons)
        season_matches = all_matches[all_matches["Season"] == season].sort_values("Date").reset_index(drop=True)
        actual = actual_final_outcome(all_matches, season)

        for frac in CUTOFF_FRACTIONS:
            cutoff_idx = int(len(season_matches) * frac)
            played = season_matches.iloc[:cutoff_idx]
            remaining = season_matches.iloc[cutoff_idx:][["HomeTeam", "AwayTeam"]]
            if played.empty or remaining.empty:
                continue
            cutoff_date = played["Date"].max() + pd.Timedelta(days=1)

            strength = compute_decayed_strength(all_matches, cutoff_date, half_life_days=180)

            all_teams = pd.unique(pd.concat([played["HomeTeam"], played["AwayTeam"],
                                              remaining["HomeTeam"], remaining["AwayTeam"]]))
            strength = fill_missing_teams(strength, all_teams)

            history_so_far = all_matches[all_matches["Date"] < cutoff_date]
            league_avg_home = history_so_far["FTHG"].mean()
            league_avg_away = history_so_far["FTAG"].mean()

            points_matrix, teams = run_monte_carlo(played, remaining, strength, league_avg_home, league_avg_away, n_simulations=3000)
            pred = predicted_table(points_matrix, teams)

            predicted_champion = pred.iloc[0]["Team"]
            predicted_top4 = set(pred.iloc[:4]["Team"])
            predicted_relegated = set(pred.iloc[-3:]["Team"])

            is_correct = predicted_champion == actual["champion"]
            top4_recall = len(predicted_top4 & actual["top4"]) / len(actual["top4"])
            relegation_recall = len(predicted_relegated & actual["relegated"]) / len(actual["relegated"])

            champion_hits += int(is_correct)
            top4_recalls.append(top4_recall)
            relegation_recalls.append(relegation_recall)
            n_cases += 1

            print(f"{season} @ {int(frac*100)}% played -- champion: {predicted_champion} ({'hit' if is_correct else 'miss'}) | "
                  f"top4 recall: {top4_recall:.2f} | relegation recall: {relegation_recall:.2f}")

    print(f"\n=== Summary across {n_cases} test cases ({len(BACKTEST_SEASONS)} seasons x {len(CUTOFF_FRACTIONS)} cutoffs) ===")
    print(f"Champion accuracy: {champion_hits}/{n_cases} = {champion_hits/n_cases:.2f}")
    print(f"Mean top-4 recall: {sum(top4_recalls)/n_cases:.2f}")
    print(f"Mean relegation recall: {sum(relegation_recalls)/n_cases:.2f}")


if __name__ == "__main__":
    run_backtest()