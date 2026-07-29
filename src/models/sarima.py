"""SARIMA/ARIMA modeling utilities for the 24h-ahead centerpiece benchmark.

Key design decisions:

- Seasonal period m=24 (daily), not m=168 (weekly). Weekly seasonality is
  computationally expensive in statsmodels' state-space implementation,
  while earlier baseline experiments showed that daily seasonality provides
  a better trade-off between accuracy and training time.
- Rolling-origin backtesting updates the fitted model using
  `.append(refit=False)`. Model parameters remain fixed after the initial
  fit, while only the internal state is updated with new observations.
  This is much faster than re-estimating the model at every forecast origin.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tests(series: pd.Series) -> dict:
    """Evaluate whether a time series is stationary using ADF and KPSS tests."""

    # Remove missing values because both statistical tests require
    # complete observations.
    clean = series.dropna()

    # Augmented Dickey-Fuller:
    # H0 = the series contains a unit root (non-stationary).
    adf_stat, adf_p, *_ = adfuller(clean)

    # KPSS:
    # H0 = the series is stationary.
    # Using both tests provides stronger evidence because they
    # test opposite null hypotheses.
    kpss_stat, kpss_p, *_ = kpss(clean, nlags="auto")

    # Print an easy-to-read interpretation of both tests.
    print(
        f"ADF:  statistic={adf_stat:.4f}, p-value={adf_p:.4f} "
        f"({'stationary' if adf_p < 0.05 else 'non-stationary'} per ADF)"
    )

    print(
        f"KPSS: statistic={kpss_stat:.4f}, p-value={kpss_p:.4f} "
        f"({'stationary' if kpss_p > 0.05 else 'non-stationary'} per KPSS)"
    )

    # Return the test statistics for later reporting.
    return {
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_p,
        "kpss_statistic": kpss_stat,
        "kpss_pvalue": kpss_p,
    }


def fit_model(
    train: pd.Series,
    order: tuple,
    seasonal_order: tuple = (0, 0, 0, 0),
):
    """Fit an ARIMA or SARIMA model using statsmodels."""

    # Construct the state-space model.
    # Setting both constraints to False allows the optimizer to explore
    # a wider range of parameter values.
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # Fit the model parameters.
    #
    # Statsmodels' Kalman filter naturally handles missing observations,
    # so short gaps do not need to be imputed before fitting.
    return model.fit(disp=False)


def rolling_backtest(
    series: pd.Series,
    fitted_results,
    origins: pd.DatetimeIndex,
    horizon: int,
    method_name: str,
) -> pd.DataFrame:
    """Perform rolling-origin forecasting over the test period."""

    records = []

    # Start from the initially fitted model.
    current_results = fitted_results

    # Keep track of the most recent observation already incorporated
    # into the model state.
    last_end = fitted_results.data.row_labels[-1]

    # Process each forecast origin sequentially.
    for origin in origins:

        # Identify observations that became available since
        # the previous forecast origin.
        new_obs = series.loc[
            last_end + pd.Timedelta(hours=1): origin
        ]

        if len(new_obs) > 0:
            # Update the model state with new observations
            # without re-estimating model parameters.
            current_results = current_results.append(
                new_obs,
                refit=False,
            )
            last_end = origin

        # Forecast the requested number of future time steps.
        forecast = (
            current_results
            .get_forecast(steps=horizon)
            .predicted_mean
        )

        # Generate timestamps corresponding to the forecast horizon.
        idx_future = pd.date_range(
            origin + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )

        # Retrieve the true observations for evaluation.
        actual = series.reindex(idx_future)

        # Store forecasts in the same long-format structure used
        # by earlier baseline experiments.
        for h, (a, f) in enumerate(
            zip(actual.values, forecast.values),
            start=1,
        ):
            records.append(
                {
                    "origin": origin,
                    "horizon_step": h,
                    "method": method_name,
                    "actual": a,
                    "forecast": f,
                }
            )

    return pd.DataFrame.from_records(records)