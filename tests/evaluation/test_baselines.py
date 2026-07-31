import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (
    compute_metrics,
    daily_seasonal_naive,
    get_forecast_origins,
    mae,
    mape,
    naive_forecast,
    rmse,
    run_backtest,
    smape,
    weekly_seasonal_naive,
)


@pytest.fixture
def long_series():
    """10 days hourly, values = position index, so lagged lookups are easy to check by hand."""
    idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


class TestNaiveForecast:
    def test_repeats_last_observed_value(self, long_series):
        origin = long_series.index[100]
        forecast = naive_forecast(long_series, origin, horizon=24)

        assert len(forecast) == 24
        assert (forecast == long_series.loc[origin]).all()
        assert forecast.index[0] == origin + pd.Timedelta(hours=1)
        assert forecast.index[-1] == origin + pd.Timedelta(hours=24)


class TestDailySeasonalNaive:
    def test_uses_value_24h_earlier_for_each_step(self, long_series):
        origin = long_series.index[100]
        forecast = daily_seasonal_naive(long_series, origin, horizon=24)

        expected = long_series.loc[origin - pd.Timedelta(hours=23):origin + pd.Timedelta(hours=1)]
        # value at future step h is the value 24h before that future timestamp
        for h in range(1, 25):
            future_ts = origin + pd.Timedelta(hours=h)
            assert forecast.loc[future_ts] == long_series.loc[future_ts - pd.Timedelta(hours=24)]

    def test_raises_keyerror_when_lookback_out_of_range(self, long_series):
        origin = long_series.index[5]  # too close to start for a 24h lookback
        with pytest.raises(KeyError):
            daily_seasonal_naive(long_series, origin, horizon=24)


class TestWeeklySeasonalNaive:
    def test_uses_value_168h_earlier(self, long_series):
        origin = long_series.index[200]
        forecast = weekly_seasonal_naive(long_series, origin, horizon=24)

        for h in range(1, 25):
            future_ts = origin + pd.Timedelta(hours=h)
            assert forecast.loc[future_ts] == long_series.loc[future_ts - pd.Timedelta(hours=168)]


class TestGetForecastOrigins:
    def test_origins_stay_within_valid_horizon_window(self, long_series):
        test_start = long_series.index[24]
        origins = get_forecast_origins(long_series, test_start, horizon=24, origin_freq="24h")

        assert origins[0] == test_start
        last_valid = long_series.index.max() - pd.Timedelta(hours=24)
        assert origins[-1] <= last_valid
        assert (origins.to_series().diff().dropna() == pd.Timedelta(hours=24)).all()


class TestRunBacktest:
    def test_produces_one_row_per_origin_step_method(self, long_series):
        origins = get_forecast_origins(long_series, long_series.index[48], horizon=24)
        methods = {"naive": naive_forecast, "daily_seasonal": daily_seasonal_naive}

        result = run_backtest(long_series, methods, origins, horizon=24)

        assert set(result["method"].unique()) == set(methods.keys())
        assert len(result) == len(origins) * 24 * len(methods)
        assert list(result.columns) == ["origin", "horizon_step", "method", "actual", "forecast"]

    def test_skips_origins_where_lookback_raises_keyerror(self, long_series):
        # first origin is too early for a 24h lookback, so daily_seasonal should
        # silently skip it while naive (no lookback) still produces rows for it
        origins = pd.DatetimeIndex([long_series.index[5], long_series.index[100]])
        methods = {"naive": naive_forecast, "daily_seasonal": daily_seasonal_naive}

        result = run_backtest(long_series, methods, origins, horizon=24)

        naive_origins = set(result.loc[result["method"] == "naive", "origin"])
        seasonal_origins = set(result.loc[result["method"] == "daily_seasonal", "origin"])
        assert origins[0] in naive_origins
        assert origins[0] not in seasonal_origins


class TestMetrics:
    def test_mae(self):
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.0, 4.0, 0.0])
        assert mae(actual, forecast) == pytest.approx(5 / 3)

    def test_rmse(self):
        actual = np.array([0.0, 0.0])
        forecast = np.array([3.0, 4.0])
        assert rmse(actual, forecast) == pytest.approx(np.sqrt((9 + 16) / 2))

    def test_mape_masks_zero_actuals(self):
        actual = np.array([0.0, 10.0])
        forecast = np.array([5.0, 12.0])
        # the zero-actual row must be excluded, not divide-by-zero
        assert mape(actual, forecast) == pytest.approx(20.0)

    def test_smape_masks_rows_where_both_are_zero(self):
        actual = np.array([0.0, 10.0])
        forecast = np.array([0.0, 8.0])
        expected = 2 * abs(10 - 8) / (10 + 8) * 100
        assert smape(actual, forecast) == pytest.approx(expected)

    def test_compute_metrics_drops_nan_pairs(self):
        df = pd.DataFrame({
            "actual": [1.0, np.nan, 3.0, 4.0],
            "forecast": [1.0, 2.0, np.nan, 5.0],
        })

        metrics = compute_metrics(df)

        assert metrics["n"] == 2  # only rows (1,1) and (4,5) have both values present
        assert metrics["MAE"] == pytest.approx(np.mean([0.0, 1.0]))
