"""Anomaly visualization -- the 27 days Phase 3 flagged (by 2+ of 4
independent detection methods: z-score, rolling IQR, STL residual,
Isolation Forest), shown against the full daily series.

Reuses the already-investigated flagged-date list from
`src/features/build_features.py` rather than re-running detection live --
those 27 dates are the result of Phase 3's full cross-method investigation
(`02_anomaly_detection.ipynb`), not something to silently recompute
differently here.
"""

import sys
from pathlib import Path

# See the matching comment in src/dashboard/app.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_access import get_flagged_anomaly_dates, load_hourly_series

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection")

st.caption(
    "27 days flagged by 2+ of Phase 3's four independent anomaly-detection methods "
    "(z-score, rolling IQR, STL residual, Isolation Forest) -- none overlap the known "
    "data-gap dates, a clean separation between \"missing data\" and \"unusual-but-present data\"."
)

series = load_hourly_series()
daily = series.resample("D").mean()
flagged_dates = get_flagged_anomaly_dates()
overall_mean = daily.mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="daily mean (kW)", line=dict(color="#555", width=1)))

flagged_values = daily.reindex(flagged_dates).dropna()
fig.add_trace(go.Scatter(
    x=flagged_values.index, y=flagged_values.values, mode="markers", name="flagged anomaly day",
    marker=dict(color="crimson", size=10, symbol="x"),
))
fig.add_hline(y=overall_mean, line_dash="dot", line_color="gray", annotation_text="overall mean")
fig.update_layout(
    title="Daily mean consumption, with flagged anomaly days highlighted",
    yaxis_title="Global active power (kW)",
    height=500,
)
st.plotly_chart(fig, width="stretch")

st.subheader("Flagged days in detail")
detail = pd.DataFrame({
    "date": flagged_dates,
    "daily_mean_kw": daily.reindex(flagged_dates).values,
})
detail["vs_overall_mean"] = detail["daily_mean_kw"] - overall_mean
detail["direction"] = detail["vs_overall_mean"].apply(lambda x: "high" if x > 0 else "low")
detail = detail.sort_values("daily_mean_kw", ascending=False)

st.dataframe(
    detail.style.format({"daily_mean_kw": "{:.3f}", "vs_overall_mean": "{:+.3f}"}),
    width="stretch",
    hide_index=True,
)

n_high = (detail["direction"] == "high").sum()
n_low = (detail["direction"] == "low").sum()
st.caption(
    f"{n_high} high-power days (cold snaps / gatherings) and {n_low} low-power days "
    "(likely short absences) -- the two anomaly flavors Phase 3 identified. "
    "`is_flagged_anomaly` (built from this same list) is a Phase 4 feature every "
    "LightGBM model in this project has access to."
)
