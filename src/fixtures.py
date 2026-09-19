"""
fixtures.py

Loads the current-season fixture list (played + remaining) from a
manually downloaded CSV (fixturedownload.com blocks automated
requests, so this file is grabbed by hand once per matchweek from
https://fixturedownload.com/results/epl-2026 -> Download as CSV).
"""

import pandas as pd

# fixturedownload.com name -> football-data.co.uk name
NAME_MAP = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}


def load_fixtures(path: str = "data/raw/pl_2026_27_fixtures.csv"):
    df = pd.read_csv(path)

    df["HomeTeam"] = df["Home Team"].replace(NAME_MAP)
    df["AwayTeam"] = df["Away Team"].replace(NAME_MAP)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")

    # Pull two numbers out of the Result column regardless of exact
    # formatting ("2-1", "2 - 1", etc). Rows with no valid score
    # (blank, NaN, "-") end up with NaN in both columns -> unplayed.
    scores = df["Result"].astype(str).str.extract(r"(\d+)\D+(\d+)")
    df["FTHG"] = pd.to_numeric(scores[0], errors="coerce")
    df["FTAG"] = pd.to_numeric(scores[1], errors="coerce")

    played = df[df["FTHG"].notna() & df["FTAG"].notna()].copy()
    remaining = df[df["FTHG"].isna() | df["FTAG"].isna()].copy()

    played["FTHG"] = played["FTHG"].astype(int)
    played["FTAG"] = played["FTAG"].astype(int)

    return played[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]], remaining[["Date", "HomeTeam", "AwayTeam"]]


if __name__ == "__main__":
    played, remaining = load_fixtures()
    played.to_csv("data/raw/pl_2026_27_played.csv", index=False)
    remaining.to_csv("data/raw/pl_2026_27_remaining.csv", index=False)
    print(f"{len(played)} matches played so far, {len(remaining)} remaining this season")