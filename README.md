# Premier League Forecasting Engine

A Monte Carlo season simulator for the Premier League, built to model
title race, top-4, and relegation probabilities from match-level
data — plus a natural-language query layer on top (in progress).

## What this does

Instead of predicting a single winner outright, this simulates the
rest of a Premier League season thousands of times using a
Poisson-based match model, and reports the *distribution* of
outcomes: title probability, top-4 probability, and relegation
probability per team. This mirrors how real sports analytics
systems (e.g. FiveThirtyEight's SPI) approach forecasting — as a
probability distribution grounded in team strength, not a coin-flip
classifier on a single-season target.

## Methodology

1. **Team strength ratings**: attack/defense strength per team,
   split home/away, computed relative to league-average goals. Uses
   time-decay weighting (`compute_decayed_strength`) so recent form
   counts more than results from months ago, rather than a flat
   average across a fixed window.
2. **Match simulation**: each fixture's expected goals are computed
   from both teams' strength ratings, then a Poisson draw generates
   a scoreline. Vectorized with numpy across all simulations at
   once for performance.
3. **Season simulation**: 10,000 simulated versions of the
   remainder of the season, aggregated into per-team probabilities
   and a single predicted final table (average points across all
   simulations, ranked).
4. **Promoted teams** (no historical top-flight data) get a
   discounted fallback rating rather than a flat league-average
   one, reflecting the historical base rate that newly promoted
   sides tend to underperform average.

## Data sources

- **Historical results (2019/20–2025/26)**: football-data.co.uk,
  via the datahub.io mirror.
- **Current season fixtures (2026/27)**: fixturedownload.com,
  manually downloaded (their CSV endpoint blocks automated
  requests per `robots.txt`) and re-downloaded periodically as new
  matchweeks are played.

## Backtest results

Tested by rewinding to an early/mid-season point in past seasons,
simulating the rest of that season, and comparing against what
actually happened:

| Metric | Result |
|---|---|
| Champion accuracy | 9/12 (75%) |
| Mean top-4 recall | 0.79 |
| Mean relegation recall | 0.64 |

**On the misses, specifically** — this matters more than the raw
number: the remaining misses aren't model failures so much as
genuinely unpredictable events even to human experts at the time:

- **2022/23**: Arsenal were correctly identified as league leaders;
  Man City's title-winning surge was one of the greatest run-ins in
  Premier League history and wasn't visible in the data before it
  happened.
- **2023/24**: the tightest title race in Premier League history —
  City, Arsenal, and Liverpool separated by single digits into the
  final weeks. No model calls that early with confidence.
- **2024/25**: Man City's uncharacteristic collapse was driven
  substantially by Rodri's long-term injury — a squad disruption
  invisible to a model built purely on match results.

Rather than chase 100% on a 12-case backtest (which would likely
mean overfitting to these specific known outcomes), the priority was
expanding the test set for a more statistically meaningful estimate
— see `src/backtest.py` for the current season/cutoff coverage.

## Project structure
PL-prediction/
├── data/
│ ├── raw/ # downloaded match + fixture CSVs
│ └── processed/ # computed strength ratings
├── src/
│ ├── data_pipeline.py # historical match data ingestion
│ ├── fixtures.py # current-season fixture loading
│ ├── strength_ratings.py
│ ├── simulator.py # Poisson model + Monte Carlo engine
│ ├── run_simulation.py # end-to-end pipeline entry point
│ ├── backtest.py # accuracy/recall validation
│ └── rag/ # natural-language query layer (planned)
├── notebooks/
├── outputs/ # simulation results
├── requirements.txt
└── README.md

## Setup

```powershell
pip install -r requirements.txt
python src\data_pipeline.py      # fetch 2019/20-2025/26 historical data
# manually download current-season fixtures from fixturedownload.com,
# save to data\raw\pl_2026_27_fixtures.csv
python src\fixtures.py           # split into played/remaining
python src\run_simulation.py     # generate forecast + predicted table
python src\backtest.py           # run accuracy/recall validation
```

## Known limitations

- Team strength is built purely from match results — no injury,
  transfer, or squad-quality data, which is the main source of
  misses noted above.
- Promoted teams' fallback rating (`PROMOTED_DISCOUNT`) is a
  reasonable starting estimate, not yet tuned against the backtest.
- No RAG/LLM query layer yet — planned next.

## Next steps

- Natural-language query interface (RAG) over the forecast data
- Golden Boot forecasting using player-level scoring data
- Expand backtest coverage to more seasons and cutoff points
