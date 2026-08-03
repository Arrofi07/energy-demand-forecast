"""Model comparison dashboard -- the 24h benchmark leaderboard (Phase 9A)
and the 1h/24h/7d horizon-sensitivity comparison (Phase 9B), both computed
live from the saved backtest results in `data/results/`.
"""

import sys
from pathlib import Path

# See the matching comment in src/dashboard/app.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.data_access import load_24h_benchmark_results, load_horizon_sensitivity_results, summarize_24h_benchmark

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")

view = st.radio("View", ["24h benchmark", "Horizon sensitivity (1h / 24h / 7d)"], horizontal=True)

if view == "24h benchmark":
    summary = summarize_24h_benchmark()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Leaderboard")
        st.dataframe(
            summary.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.1f}%", "sMAPE": "{:.1f}%", "n": "{:.0f}"})
            .highlight_min(subset=["MAE", "RMSE"], color="#d4f4dd"),
            width="stretch",
        )
    with col2:
        st.subheader("MAE by method")
        fig = px.bar(summary.reset_index(names="method"), x="method", y="MAE", color="method")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Error vs. horizon step")
    all_24h = load_24h_benchmark_results()
    horizon_metrics = (
        all_24h[all_24h["method"].isin(summary.index)]
        .groupby(["method", "horizon_step"])
        .apply(lambda d: pd.Series({"MAE": (d["actual"] - d["forecast"]).abs().mean()}), include_groups=False)
        .reset_index()
    )
    fig = px.line(horizon_metrics, x="horizon_step", y="MAE", color="method", markers=True)
    fig.update_layout(xaxis_title="Horizon step (hours ahead)", yaxis_title="MAE (kW)", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Notice the double peak around h=7 (7am) and h=19-21 (7-9pm) for every method -- "
        "Phase 7's finding that morning/evening consumption swings are the hardest to "
        "predict regardless of model paradigm."
    )

else:
    sensitivity = load_horizon_sensitivity_results()
    pivot = sensitivity.pivot_table(index="method", columns="horizon", values="MAE")
    pivot = pivot[[c for c in ["1h", "24h", "7d"] if c in pivot.columns]]

    st.subheader("MAE across horizons")
    st.dataframe(pivot.style.format("{:.3f}"), width="stretch")

    fig = px.line(
        sensitivity, x="horizon", y="MAE", color="method", markers=True,
        category_orders={"horizon": ["1h", "24h", "7d"]},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "LightGBM/LSTM stay near the top at every horizon. SARIMA is competitive at 1h "
        "but becomes the single worst method of any kind at 7d -- worse than flat "
        "persistence. Prophet moves the opposite direction: weak at 1h, 2nd-best at 7d."
    )
