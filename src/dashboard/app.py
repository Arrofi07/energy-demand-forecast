"""Energy Demand Forecast dashboard -- home page.

Run with: `uv run streamlit run src/dashboard/app.py`

Deliberately thin, matching `src/api/main.py`'s philosophy: every number
shown here is loaded live from `data/results/` (via
`src/evaluation/results_store.py`) or computed live from the production
model bundle (`src/pipeline/`) -- nothing is hardcoded from the README or
ROADMAP, so this page can't silently drift out of sync with the actual
saved artifacts.
"""

import sys
from pathlib import Path

# Streamlit runs this file directly (not via `python -m`), so Python only
# puts src/dashboard/ on sys.path, not the project root -- without this,
# every `from src...` import below fails with ModuleNotFoundError, caught
# by actually launching the app rather than just writing it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.data_access import summarize_24h_benchmark

st.set_page_config(page_title="Energy Demand Forecast", page_icon="⚡", layout="wide")

st.title("⚡ Energy Demand Forecasting")
st.caption(
    "Day-ahead household electricity demand forecasting -- classical statistics, "
    "decomposition, gradient boosting, and deep learning, compared on identical "
    "rolling-origin backtests."
)

st.markdown(
    """
    This dashboard serves the models and results built in Phases 1-12 of the
    project (see [`ROADMAP.md`](https://github.com) for the full phase-by-phase
    writeup). Use the sidebar to explore:

    - **Live Forecast** -- pick any timestamp in the dataset and get a real
      24h-ahead forecast from the production LightGBM model, compared
      against the actual values where known, with an empirical
      residual-based prediction interval.
    - **Model Comparison** -- the full 24h benchmark leaderboard, plus the
      1h / 24h / 7d horizon-sensitivity comparison from Phase 9B.
    - **Anomaly Detection** -- the 27 days Phase 3 flagged across four
      independent anomaly-detection methods, shown against the full series.
    - **Business Impact** -- an interactive version of Phase 10's
      illustrative cost-savings estimate; change the portfolio size and
      imbalance premium assumptions and watch the estimate update live.
    """
)

st.subheader("24h-ahead benchmark -- live from `data/results/`")

summary = summarize_24h_benchmark()
st.dataframe(
    summary.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.1f}%", "sMAPE": "{:.1f}%", "n": "{:.0f}"})
    .highlight_min(subset=["MAE", "RMSE"], color="#d4f4dd"),
    width="stretch",
)

best_method = summary["MAE"].idxmin()
baseline_mae = summary.loc["daily_seasonal_naive", "MAE"]
best_mae = summary.loc[best_method, "MAE"]
improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100

st.metric(
    label=f"Best method ({best_method}) vs. daily seasonal naive baseline",
    value=f"{best_mae:.3f} kW MAE",
    delta=f"-{improvement_pct:.1f}%",
    delta_color="inverse",
)
