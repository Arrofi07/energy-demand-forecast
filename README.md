# Energy Demand Forecasting

Day-ahead household electricity demand forecasting, built as a rigorous,
end-to-end comparison of forecasting paradigms — classical statistics
(SARIMA), decomposition (Prophet), gradient boosting (LightGBM), and deep
learning (LSTM) — evaluated on identical rolling-origin backtests, with
statistical significance testing, seasonal breakdowns, and a dedicated
horizon-sensitivity investigation, not just a single leaderboard table.

## The question

*Can a trustworthy 24-hour-ahead demand forecast be built for a real
household power series, and which forecasting paradigm actually holds up
under scrutiny — not just on average, but across time, season, and horizon?*

## Headline result

| Method | MAE (kW) | RMSE (kW) | vs. baseline |
|---|---|---|---|
| **LightGBM** (direct multi-horizon) | **0.429** | **0.591** | −19% MAE |
| LSTM (direct seq2seq) | 0.440 | 0.605 | −17% MAE |
| Prophet | 0.485 | 0.637 | −9% MAE |
| Daily seasonal naive (baseline) | 0.530 | 0.786 | — |
| SARIMA(3,1,0)(2,0,0)[24] | 0.560 | 0.776 | +6% MAE (worse) |

LightGBM and LSTM are the clear top tier — but a Diebold-Mariano
significance test found the gap *between them* is **not** statistically
significant (p = 0.098). Every other pairwise comparison is significant at
p < 0.001. That nuance, and several others like it, is the point of this
project: a single MAE column hides more than it reveals.

**A few findings that only showed up once the analysis went past the
aggregate table:**

- **The 24h winner isn't the whole story.** Re-running models at 1h and 7d
  horizons found LightGBM/LSTM stay on top throughout, but SARIMA goes from
  competitive at 1h to the single *worst* method of any kind at 7d (0.800
  MAE — worse than flat persistence), while Prophet moves the opposite
  direction, climbing from one of the weakest methods at 1h to 2nd-best at
  7d.

  ![MAE at 1h, 24h, and 7d horizons for every method — LightGBM/LSTM stay lowest throughout, SARIMA rises sharply, Prophet's rank improves](assets/horizon_sensitivity.png)

- **The naive baseline beats SARIMA and Prophet outright in summer.** Time-
  series cross-validation across 4 seasonal folds found low, stable summer
  demand makes yesterday's value nearly unbeatable — added model complexity
  doesn't pay off there.
- **A real data-leakage bug was caught and fixed mid-project**, not
  glossed over: a naive baseline's fixed 24h lag silently looked up
  future values once asked for a horizon longer than 24h. Caught before it
  could distort the 7-day results, fixed with a guard clause and a
  regression test.
- **LightGBM's engineered anomaly-flag feature measurably pays off**:
  failure-case analysis found its worst predictions land on already-known
  anomalous days half as often as LSTM's or Prophet's.

Full reasoning, caveats, and every finding above (plus a dozen more) are in
[`ROADMAP.md`](ROADMAP.md) and the notebooks themselves — every notebook
ends with a `Findings` section written against real, executed output, not
aspirational claims.

## Project structure

```
energy-demand-forecast/
├── data/
│   ├── energy.duckdb          # analytical DB: minute / hourly / daily / hourly_features tables
│   ├── processed/              # cleaned Parquet at each resolution
│   └── results/                 # saved backtest results per model (long-format, reused across notebooks)
├── notebooks/
│   ├── 01_eda_trends_seasonality.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_forecasting_baselines.ipynb
│   ├── 05_model_sarima.ipynb
│   ├── 06_model_prophet_lightgbm.ipynb
│   ├── 07_model_lstm.ipynb
│   ├── 08_model_comparison.ipynb          # Phase 9A: 24h benchmark, CV, DM tests, seasonality, failure cases
│   └── 09_horizon_sensitivity.ipynb        # Phase 9B: 1h / 24h / 7d comparison
├── src/
│   ├── data/              # download, cleaning, schema validation, DuckDB loading
│   ├── features/          # leak-free lag/rolling/calendar feature engineering
│   ├── models/            # sarima.py, prophet_model.py, lightgbm_direct.py, lstm.py
│   └── evaluation/        # shared train/test split, backtest harness, metrics, significance testing
├── tests/                  # pytest suite mirroring src/, 79 tests
├── ROADMAP.md              # phase-by-phase plan, decisions, and findings (the project's real changelog)
└── pyproject.toml
```

## Methodology at a glance

1. **Data engineering** — the UCI Individual Household Electric Power
   Consumption dataset (minute-level, 2006–2010, ~2M rows), cleaned,
   schema-validated with Pandera, and loaded into DuckDB at minute/hourly/
   daily resolutions.
2. **EDA, anomaly detection, feature engineering** — trend/seasonality
   decomposition, four independent anomaly-detection methods cross-checked
   against each other, leak-free lag/rolling/cyclical features (verified by
   tests, not just by inspection).
3. **A shared evaluation harness, built once, reused by every model** —
   the same train/test split, the same rolling forecast origins, the same
   metrics (MAE/RMSE/MAPE/sMAPE), so every model in the project is judged
   on identical terms.
4. **Four forecasting paradigms, same 24h-ahead benchmark**: seasonal
   naive baselines → SARIMA → Prophet & direct multi-horizon LightGBM →
   direct sequence-to-sequence LSTM (PyTorch, Optuna-tuned).
5. **Evaluation that goes past a leaderboard**: time-series
   cross-validation, Diebold-Mariano significance testing, per-season
   breakdowns, failure-case analysis, and a dedicated 1h/24h/7d horizon-
   sensitivity study.

See [`ROADMAP.md`](ROADMAP.md) for the full phase-by-phase breakdown,
including deferred/optional items and the reasoning behind every design
decision.

## Tech stack

- **Data**: DuckDB, Parquet, Pandera (schema validation)
- **Modeling**: statsmodels/pmdarima (SARIMA), Prophet, LightGBM, PyTorch
  (LSTM)
- **Tuning & tracking**: Optuna, MLflow
- **Evaluation**: scipy (Diebold-Mariano), custom rolling-origin backtest
  harness
- **Tooling**: uv (dependency management), pytest, Jupyter

## Getting started

```bash
# Install dependencies (Python 3.12)
uv sync

# Build the dataset (download → clean/resample → DuckDB → features)
uv run python -m src.data.download
uv run python -m src.data.load
uv run python -m src.data.duckdb_setup
uv run python -m src.features.build_features

# Run the notebooks in order (01 through 09), or open them in Jupyter/VS Code

# Run the test suite
uv run pytest

# View MLflow experiment tracking (SARIMA/Prophet/LightGBM/LSTM runs)
uv run mlflow ui
```

## Testing

79 tests covering the data pipeline, feature engineering, every model's
core logic, the evaluation harness, and the statistical significance
testing utilities — including regression tests for the data-leakage bug
mentioned above. Run with `uv run pytest`; slow tests (fitting real
Prophet/SARIMA/LightGBM models) are marked and can be excluded with
`uv run pytest -m "not slow"`.

## Data source

[UCI Machine Learning Repository — Individual Household Electric Power
Consumption](https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip),
one household's minute-level electricity consumption, December 2006 –
November 2010.

## Roadmap

This is Phases 1–9 of a 16-phase plan. Phases 1–9 form a complete,
story-worthy portfolio piece on their own (data → EDA → anomalies →
features → baselines → classical → ML/DL → rigorous evaluation). Phase 10
(business impact) and Phases 11–16 (production pipeline, API, dashboard,
MLOps, deployment) are the natural next milestone — see
[`ROADMAP.md`](ROADMAP.md) for the full plan.
