"""Convert continuous forecasts into the common backtest result format.

Some forecasting methods (e.g., Prophet or MSTL) generate a continuous
forecast timeline directly instead of updating model state at each forecast
origin. This utility converts those forecasts into the same long-format
structure used by rolling-origin models, allowing all models to be compared
with identical evaluation metrics.
"""

import pandas as pd


def to_backtest_format(
    forecast_series: pd.Series,
    actual_series: pd.Series,
    origins: pd.DatetimeIndex,
    horizon: int,
    method_name: str,
) -> pd.DataFrame:
    """Transform continuous forecasts into long-format backtest results."""

    records = []

    # Each origin represents a point in time where a forecast evaluation
    # begins. Extract the following `horizon` hours for comparison.
    for origin in origins:

        # Create timestamps for the forecast window after the origin.
        idx_future = pd.date_range(
            origin + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )

        # Retrieve predicted and observed values for this forecast window.
        # reindex() keeps the timestamps aligned even if some values are missing.
        f = forecast_series.reindex(idx_future)
        a = actual_series.reindex(idx_future)

        # Store each forecast step separately.
        # This creates the same format as SARIMA rolling backtesting:
        # one row = one prediction at one horizon step.
        for h, (av, fv) in enumerate(
            zip(a.values, f.values),
            start=1,
        ):
            records.append(
                {
                    "origin": origin,
                    "horizon_step": h,
                    "method": method_name,
                    "actual": av,
                    "forecast": fv,
                }
            )

    # Convert the list of records into a DataFrame for metric calculation,
    # visualization, and model comparison.
    return pd.DataFrame.from_records(records)