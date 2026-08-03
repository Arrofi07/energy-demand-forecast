import numpy as np
import pandas as pd
import pytest

from src.models.prophet_model import fit_model, forecast_continuous


@pytest.fixture
def synthetic_series():
    """30 days of hourly data with a clear daily rhythm -- enough history
    for Prophet to fit yearly/weekly/daily seasonality without erroring,
    without paying the cost of fitting on years of real data.
    """
    idx = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(0)
    hours = idx.hour.to_numpy()
    daily_pattern = 1.5 + np.sin(2 * np.pi * (hours - 6) / 24)
    noise = rng.normal(0, 0.05, size=len(idx))
    return pd.Series(daily_pattern + noise, index=idx, name="global_active_power")


@pytest.mark.slow
class TestFitModel:
    def test_fits_without_error(self, synthetic_series):
        model = fit_model(synthetic_series)
        assert model is not None

    def test_drops_nan_rows_before_fitting(self, synthetic_series):
        with_nan = synthetic_series.copy()
        with_nan.iloc[10] = np.nan

        model = fit_model(with_nan)

        assert model.history.shape[0] == len(synthetic_series) - 1


@pytest.mark.slow
class TestForecastContinuous:
    def test_returns_series_covering_history_plus_requested_periods(self, synthetic_series):
        model = fit_model(synthetic_series)

        forecast = forecast_continuous(model, periods=24)

        assert len(forecast) == len(synthetic_series) + 24

    def test_future_portion_starts_immediately_after_training_data(self, synthetic_series):
        model = fit_model(synthetic_series)

        forecast = forecast_continuous(model, periods=24)
        future_part = forecast.iloc[-24:]

        assert future_part.index[0] == synthetic_series.index[-1] + pd.Timedelta(hours=1)
        assert future_part.index[-1] == synthetic_series.index[-1] + pd.Timedelta(hours=24)
