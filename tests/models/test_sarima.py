import numpy as np
import pandas as pd
import pytest

from src.models.sarima import fit_model, rolling_backtest, stationarity_tests


@pytest.fixture
def stationary_series():
    idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(2.0, 0.2, size=len(idx)), index=idx).asfreq("h")


@pytest.fixture
def trending_series():
    idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    rng = np.random.default_rng(0)
    trend = np.linspace(0, 10, len(idx))
    return pd.Series(trend + rng.normal(0, 0.1, size=len(idx)), index=idx).asfreq("h")


class TestStationarityTests:
    def test_returns_expected_keys(self, stationary_series):
        result = stationarity_tests(stationary_series)
        assert set(result.keys()) == {"adf_statistic", "adf_pvalue", "kpss_statistic", "kpss_pvalue"}

    def test_stationary_series_has_low_adf_pvalue(self, stationary_series):
        result = stationarity_tests(stationary_series)
        assert result["adf_pvalue"] < 0.05

    def test_trending_series_has_high_adf_pvalue(self, trending_series):
        result = stationarity_tests(trending_series)
        assert result["adf_pvalue"] > 0.05

    def test_drops_nan_before_testing(self, stationary_series):
        with_nan = stationary_series.copy()
        with_nan.iloc[5] = np.nan
        result = stationarity_tests(with_nan)  # should not raise
        assert "adf_statistic" in result


@pytest.mark.slow
class TestFitModel:
    def test_fits_and_produces_a_forecastable_result(self, stationary_series):
        fitted = fit_model(stationary_series, order=(1, 0, 0))
        forecast = fitted.get_forecast(steps=5).predicted_mean
        assert len(forecast) == 5

    def test_seasonal_term_changes_aic(self, stationary_series):
        plain = fit_model(stationary_series, order=(1, 0, 0))
        seasonal = fit_model(stationary_series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
        assert plain.aic != seasonal.aic


@pytest.mark.slow
class TestRollingBacktest:
    def test_produces_expected_long_format_shape(self, stationary_series):
        fitted = fit_model(stationary_series, order=(1, 0, 0))
        origins = stationary_series.index[100:103]

        result = rolling_backtest(stationary_series, fitted, origins, horizon=3, method_name="test_method", window_hours=96)

        assert list(result.columns) == ["origin", "horizon_step", "method", "actual", "forecast"]
        assert len(result) == len(origins) * 3
        assert set(result["method"]) == {"test_method"}
        assert set(result["horizon_step"]) == {1, 2, 3}
