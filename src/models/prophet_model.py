"""Prophet forecasting for the 24h-ahead centerpiece benchmark.

Key design decision: unlike SARIMA, Prophet does not depend on recent lag
values (trend + yearly/weekly/daily seasonality only), so it doesn't need to
be walked forward origin-by-origin to backtest. Fit once on the training
period, then predict a single continuous forecast across the entire test
range -- `src/evaluation/continuous.py` slices that continuous forecast into
the same (origin, horizon_step) long-format table used by every other model,
so metrics stay directly comparable.
"""

import logging

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def fit_model(train: pd.Series) -> Prophet:
    """Fit Prophet on an hourly-indexed training series.

    Rows with missing values are dropped before fitting -- Prophet doesn't
    require a fully regular index (it treats `ds` as arbitrary timestamps),
    so the known Phase 1 data gaps don't need interpolation here.
    """
    train_df = train.dropna().reset_index()
    train_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.fit(train_df)
    return model


def forecast_continuous(model: Prophet, periods: int, freq: str = "h") -> pd.Series:
    """Predict one continuous forecast covering `periods` steps past training end.

    `make_future_dataframe` appends `periods` future timestamps to the
    training history; only the future portion is needed downstream, but
    returning the full `yhat` series keeps this function a simple wrapper.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"]
