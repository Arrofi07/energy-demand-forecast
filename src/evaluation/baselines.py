"""Baseline forecasting methods and backtesting harness for the 24h-ahead benchmark.

All three baselines are pure lookups -- no fitting -- so "backtesting" here
means repeated evaluation across many forecast origins, not training. These
numbers are the bar every later model (SARIMA, Prophet, LightGBM, LSTM) has
to beat.
"""

import numpy as np
import pandas as pd

HORIZON = 24


def naive_forecast(series: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """Flat-line forecast: repeat the last observed value for all `horizon` steps.
    Absolute minimum bar -- if a model can't beat this, it's not adding value.
    """
    last_val = series.loc[origin]
    idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.Series(last_val, index=idx)


def _lagged_forecast(series: pd.Series, origin: pd.Timestamp, horizon: int, lag_hours: int) -> pd.Series:
    """Forecast T+h using the value observed exactly `lag_hours` earlier, for each h.

    Requires `horizon <= lag_hours`: the lookback for the farthest step is
    `origin + horizon - lag_hours`, which is only guaranteed to be at or
    before `origin` (i.e. actually known at forecast time) when the lag is
    at least as long as the horizon. Calling this with a longer horizon than
    lag would silently look up values *after* the origin -- future data
    leakage rather than a legitimate forecast.
    """
    if horizon > lag_hours:
        raise ValueError(
            f"horizon ({horizon}) exceeds lag_hours ({lag_hours}) -- the farthest "
            "forecast step would look up a value after the origin (future leakage)."
        )
    idx_future = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
    idx_lookback = idx_future - pd.Timedelta(hours=lag_hours)
    values = series.loc[idx_lookback].values
    return pd.Series(values, index=idx_future)


def daily_seasonal_naive(series: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """'Tomorrow looks like today's same hours' -- captures daily rhythm, misses day-of-week shifts."""
    return _lagged_forecast(series, origin, horizon, lag_hours=24)


def weekly_seasonal_naive(series: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """'Next day looks like the same day last week' -- captures daily rhythm AND day-of-week effect."""
    return _lagged_forecast(series, origin, horizon, lag_hours=168)


def extended_daily_seasonal_naive(series: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.Series:
    """Repeat the most recently observed 24h profile for the entire horizon.

    Unlike `daily_seasonal_naive` (each step looks up its own specific day-ago
    value, and is only leak-free up to horizon=24 -- see `_lagged_forecast`'s
    guard), this looks up exactly one 24h block ending at `origin` and tiles
    it forward, so every value it ever reads is at or before `origin` --
    leak-free at *any* horizon. It's mathematically identical to
    `daily_seasonal_naive` whenever horizon<=24 (both reduce to "the value
    24h before this step"), and is the extension of that same "recency"
    baseline to horizons where the original isn't valid -- testing whether
    recency (Phase 5's finding: daily lag beat weekly lag at 24h) still helps
    at longer horizons, or was an artifact specific to the 24h horizon where
    it was first measured.
    """
    last_24 = series.loc[origin - pd.Timedelta(hours=23): origin].values
    if len(last_24) != 24:
        raise ValueError(f"Expected 24h of history before {origin}, got {len(last_24)}.")
    idx_future = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
    tiled = np.tile(last_24, int(np.ceil(horizon / 24)))[:horizon]
    return pd.Series(tiled, index=idx_future)


def get_forecast_origins(series: pd.Series, test_start: pd.Timestamp, horizon: int = HORIZON,
                          origin_freq: str = "24h") -> pd.DatetimeIndex:
    """One forecast origin per day across the test period -- matches issuing
    one day-ahead forecast per day in production, and keeps forecast windows
    non-overlapping for cleaner backtesting statistics.
    """
    test_end = series.index.max()
    last_valid_origin = test_end - pd.Timedelta(hours=horizon)
    return pd.date_range(start=test_start, end=last_valid_origin, freq=origin_freq)


def run_backtest(series: pd.Series, methods: dict, origins: pd.DatetimeIndex,
                  horizon: int = HORIZON) -> pd.DataFrame:
    """Run every method at every origin, return a long-format results table:
    one row per (origin, horizon_step, method) with actual vs. forecast.
    """
    records = []
    for origin in origins:
        idx_future = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        actual = series.reindex(idx_future)
        for name, fn in methods.items():
            try:
                forecast = fn(series, origin, horizon)
            except KeyError:
                continue  # lookback window falls outside available data -- skip this origin
            for h, (a, f) in enumerate(zip(actual.values, forecast.values), start=1):
                records.append({"origin": origin, "horizon_step": h, "method": name,
                                 "actual": a, "forecast": f})
    return pd.DataFrame.from_records(records)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(forecast)
    mask = denom != 0
    return float(np.mean(2 * np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100)


def compute_metrics(df: pd.DataFrame, actual_col: str = "actual", forecast_col: str = "forecast") -> dict:
    """Drop any (actual, forecast) pair touching a data gap before scoring -- NaNs
    would otherwise silently poison the aggregate metrics.
    """
    d = df.dropna(subset=[actual_col, forecast_col])
    a, f = d[actual_col].values, d[forecast_col].values
    return {"MAE": mae(a, f), "RMSE": rmse(a, f), "MAPE": mape(a, f), "sMAPE": smape(a, f), "n": len(d)}