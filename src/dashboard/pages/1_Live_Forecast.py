"""Interactive 24h-ahead forecast, actual-vs-prediction comparison, and an
empirical prediction interval -- all computed live against the production
LightGBM bundle (`src/pipeline/forecaster.py`), the same code path
`src/api/main.py`'s `/forecast` endpoint uses.
"""

import datetime as dt
import sys
from pathlib import Path

# See the matching comment in src/dashboard/app.py -- pages are one
# directory deeper, so they need one more .parent to reach the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_access import get_empirical_error_bands, get_forecaster, load_hourly_series
from src.pipeline.inference_features import LOOKBACK_HOURS, get_latest_features

st.set_page_config(page_title="Live Forecast", page_icon="📈", layout="wide")
st.title("📈 Live Forecast")

series = load_hourly_series()
min_valid = (series.index.min() + pd.Timedelta(hours=LOOKBACK_HOURS)).to_pydatetime()
max_valid = series.index.max().to_pydatetime()

st.caption(
    f"Pick any forecast origin between **{min_valid:%Y-%m-%d %H:%M}** and "
    f"**{max_valid:%Y-%m-%d %H:%M}** (the full dataset range, minus the "
    f"~{LOOKBACK_HOURS}h of history every forecast needs as input)."
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    picked_date = st.date_input(
        "Forecast origin date", value=max_valid.date(), min_value=min_valid.date(), max_value=max_valid.date(),
    )
with col2:
    picked_hour = st.selectbox("Hour", options=list(range(24)), index=max_valid.hour)
with col3:
    st.write("")
    st.write("")
    generate = st.button("Generate forecast", type="primary")

as_of = pd.Timestamp(dt.datetime.combine(picked_date, dt.time(hour=picked_hour)))
as_of = min(max(as_of, pd.Timestamp(min_valid)), pd.Timestamp(max_valid))

if generate:
    try:
        features_row = get_latest_features(as_of)
    except ValueError as exc:
        st.error(f"Can't forecast from this origin: {exc}")
        st.stop()

    forecaster = get_forecaster()
    result = forecaster.predict(None, features_row)
    result["target_timestamp"] = as_of + pd.to_timedelta(result["horizon_step"], unit="h")
    result["actual"] = series.reindex(result["target_timestamp"]).values

    bands = get_empirical_error_bands()
    result = result.merge(bands, left_on="horizon_step", right_index=True, how="left")
    result["lower"] = result["forecast"] + result["lower_q"]
    result["upper"] = result["forecast"] + result["upper_q"]

    context_start = as_of - pd.Timedelta(hours=72)
    context = series.loc[context_start:as_of]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=context.index, y=context.values, name="actual (last 72h)", line=dict(color="#555")))
    fig.add_trace(go.Scatter(
        x=result["target_timestamp"], y=result["upper"], line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=result["target_timestamp"], y=result["lower"], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(220,20,60,0.15)", name="empirical 80% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=result["target_timestamp"], y=result["forecast"], name="forecast (next 24h)",
        line=dict(color="crimson"), mode="lines+markers",
    ))
    if result["actual"].notna().any():
        fig.add_trace(go.Scatter(
            x=result["target_timestamp"], y=result["actual"], name="actual (if known)",
            line=dict(color="#2ca02c", dash="dot"), mode="lines+markers",
        ))
    fig.add_vline(x=as_of, line_dash="dash", line_color="black", annotation_text="forecast origin")
    fig.update_layout(
        title=f"24h-ahead forecast issued at {as_of}",
        yaxis_title="Global active power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    if result["actual"].notna().any():
        scored = result.dropna(subset=["actual"])
        mae = (scored["actual"] - scored["forecast"]).abs().mean()
        st.metric("MAE for this forecast (where actuals are known)", f"{mae:.3f} kW")
    else:
        st.info(
            "No actual values exist yet for this origin's forecast window -- "
            "this is a genuine forward forecast, not a backtest."
        )

    st.caption(
        "The shaded band is an **empirical** 10th-90th percentile interval, built from "
        "the real (actual - forecast) residuals of the Phase 7 backtest at each horizon "
        "step -- not a model-native prediction interval (this LightGBM setup produces "
        "point forecasts only). Widens exactly where Phase 7/9 found forecasting hardest: "
        "the morning and evening consumption peaks."
    )

    with st.expander("Raw forecast table"):
        st.dataframe(
            result[["horizon_step", "target_timestamp", "forecast", "lower", "upper", "actual"]],
            width="stretch",
        )
else:
    st.info("Pick a forecast origin and click **Generate forecast**.")
