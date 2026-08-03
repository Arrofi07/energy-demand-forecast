"""Cached data/model loaders shared across dashboard pages.

Streamlit reruns the whole script on every user interaction, so anything
expensive (loading the 24-model LightGBM bundle, querying DuckDB, reading
every saved backtest parquet) is wrapped in `st.cache_resource` /
`st.cache_data` here rather than repeated inside each page. No forecasting
or evaluation logic lives in this module -- it only loads what
`src/pipeline/` and `src/evaluation/` already produce.
"""

import duckdb
import pandas as pd
import streamlit as st

from src.evaluation.baselines import compute_metrics
from src.evaluation.results_store import load_results
from src.features.build_features import FLAGGED_ANOMALY_DATES
from src.pipeline.forecaster import DEFAULT_BUNDLE_PATH, DirectMultiHorizonForecaster, ModelBundle, load_bundle
from src.pipeline.inference_features import DB_PATH

# The 24h-horizon methods every model was benchmarked on (Phases 5-8) --
# excludes the "_7d" / horizon-sensitivity result files, which
# 2_Model_Comparison.py loads separately.
CANONICAL_24H_METHODS = ["daily_seasonal_naive", "sarima", "prophet", "lightgbm", "lstm"]

# Result file -> method name(s) it contains, for the 24h benchmark.
RESULT_FILES_24H = ["baselines", "sarima_arima", "prophet", "lightgbm_direct", "lstm"]


@st.cache_resource
def get_bundle() -> ModelBundle:
    return load_bundle(DEFAULT_BUNDLE_PATH)


def get_forecaster() -> DirectMultiHorizonForecaster:
    return DirectMultiHorizonForecaster(get_bundle())


@st.cache_data
def load_hourly_series() -> pd.Series:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT datetime, global_active_power FROM hourly ORDER BY datetime").df()
    con.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")["global_active_power"].asfreq("h")


@st.cache_data
def load_24h_benchmark_results() -> pd.DataFrame:
    """Concatenates every 24h-benchmark result file, renaming
    'daily_seasonal_naive' consistently -- matches
    `10_business_impact.ipynb`'s `all_24h` exactly.
    """
    frames = [load_results(name) for name in RESULT_FILES_24H]
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def summarize_24h_benchmark() -> pd.DataFrame:
    all_24h = load_24h_benchmark_results()
    return pd.DataFrame({
        m: compute_metrics(all_24h[all_24h["method"] == m]) for m in CANONICAL_24H_METHODS
    }).T.sort_values("MAE")


@st.cache_data
def load_horizon_sensitivity_results() -> pd.DataFrame:
    """1h / 24h / 7d comparison, matching `09_horizon_sensitivity.ipynb`:
    1h reuses the 24h files at horizon_step == 1, 7d needs the dedicated
    "_7d" result files.
    """
    records = []
    all_24h = load_24h_benchmark_results()
    for method in CANONICAL_24H_METHODS:
        subset = all_24h[all_24h["method"] == method]
        one_h = subset[subset["horizon_step"] == 1]
        if len(one_h):
            records.append({"method": method, "horizon": "1h", **compute_metrics(one_h)})
        if len(subset):
            records.append({"method": method, "horizon": "24h", **compute_metrics(subset)})

    for name in ["baselines_7d", "sarima_7d", "prophet_7d", "lightgbm_7d"]:
        df = load_results(name)
        for method in df["method"].unique():
            subset = df[df["method"] == method]
            records.append({"method": method, "horizon": "7d", **compute_metrics(subset)})

    return pd.DataFrame.from_records(records)


@st.cache_data
def get_empirical_error_bands(lower_q: float = 0.1, upper_q: float = 0.9) -> pd.DataFrame:
    """Empirical (actual - forecast) error quantiles per horizon step, from
    the real LightGBM backtest -- used to shade an approximate prediction
    interval around a live forecast. Not a model-native prediction interval
    (LightGBM here is trained for point forecasts only); this is a
    residual-based empirical band, and is labeled as such wherever it's shown.
    """
    results = load_results("lightgbm_direct").dropna(subset=["actual", "forecast"])
    results = results.copy()
    results["error"] = results["actual"] - results["forecast"]
    bands = results.groupby("horizon_step")["error"].quantile([lower_q, upper_q]).unstack()
    bands.columns = ["lower_q", "upper_q"]
    return bands


def get_flagged_anomaly_dates() -> pd.DatetimeIndex:
    return FLAGGED_ANOMALY_DATES
