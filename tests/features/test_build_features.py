import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FLAGGED_ANOMALY_DATES,
    LAG_HOURS,
    ROLLING_WINDOWS,
    add_anomaly_flag,
    add_cyclical_encoding,
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
)


class TestAddTimeFeatures:
    def test_calendar_columns_match_known_timestamps(self):
        idx = pd.DatetimeIndex([
            "2020-01-06 03:00",  # Monday, winter
            "2020-01-11 15:00",  # Saturday -> weekend, winter
            "2020-07-04 12:00",  # Saturday, summer
        ])
        df = pd.DataFrame({"global_active_power": [1.0, 2.0, 3.0]}, index=idx)

        out = add_time_features(df)

        assert out["hour"].tolist() == [3, 15, 12]
        assert out["dayofweek"].tolist() == [0, 5, 5]
        assert out["is_weekend"].tolist() == [0, 1, 1]
        assert out["season"].tolist() == ["winter", "winter", "summer"]
        assert out["quarter"].tolist() == [1, 1, 3]

    def test_does_not_mutate_input(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="h")
        df = pd.DataFrame({"global_active_power": range(5)}, index=idx)

        add_time_features(df)

        assert "hour" not in df.columns


class TestAddCyclicalEncoding:
    def test_hour_0_and_23_are_close_in_cyclical_space(self):
        idx = pd.date_range("2020-01-01 00:00", periods=24, freq="h")
        df = pd.DataFrame({"global_active_power": range(24)}, index=idx)
        df = add_time_features(df)

        out = add_cyclical_encoding(df)

        h0 = out.iloc[0][["hour_sin", "hour_cos"]].to_numpy()
        h23 = out.iloc[23][["hour_sin", "hour_cos"]].to_numpy()
        h12 = out.iloc[12][["hour_sin", "hour_cos"]].to_numpy()

        assert np.linalg.norm(h0 - h23) < np.linalg.norm(h0 - h12)

    def test_known_values_at_hour_zero(self):
        idx = pd.date_range("2020-01-06 00:00", periods=1, freq="h")  # Monday, January
        df = pd.DataFrame({"global_active_power": [1.0]}, index=idx)
        df = add_time_features(df)

        out = add_cyclical_encoding(df)

        assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)


class TestAddLagFeatures:
    def test_lag_values_match_shifted_series(self, synthetic_hourly_df):
        out = add_lag_features(synthetic_hourly_df)

        for lag in LAG_HOURS:
            col = f"global_active_power_lag_{lag}h"
            expected = synthetic_hourly_df["global_active_power"].shift(lag)
            pd.testing.assert_series_equal(out[col], expected, check_names=False)

    def test_first_rows_are_nan(self, synthetic_hourly_df):
        out = add_lag_features(synthetic_hourly_df)
        max_lag = max(LAG_HOURS)
        assert out[f"global_active_power_lag_{max_lag}h"].iloc[:max_lag].isna().all()


class TestAddRollingFeatures:
    def test_excludes_current_row_from_its_own_window(self, synthetic_hourly_df):
        """The rolling window at row i must be computed strictly from rows
        before i. Since global_active_power == row position here, the mean of
        the last N rows *ending at* i-1 has a known closed form -- if the
        current row's own value leaked in, this would be off by exactly
        row_value / N.
        """
        out = add_rolling_features(synthetic_hourly_df)
        window = ROLLING_WINDOWS[0]
        i = 300  # comfortably past the warm-up window

        col = f"global_active_power_roll_mean_{window}h"
        expected = np.mean(np.arange(i - window, i))  # positions i-window .. i-1
        assert out[col].iloc[i] == pytest.approx(expected)

    def test_produces_mean_std_min_max_for_each_window(self, synthetic_hourly_df):
        out = add_rolling_features(synthetic_hourly_df)
        for window in ROLLING_WINDOWS:
            for stat in ["mean", "std", "min", "max"]:
                assert f"global_active_power_roll_{stat}_{window}h" in out.columns


class TestAddDiffFeatures:
    def test_diff_columns_use_only_already_lagged_values(self, synthetic_hourly_df):
        out = add_diff_features(synthetic_hourly_df)
        i = 200

        # global_active_power[i] == i, so lag_1 = i-1, lag_2 = i-2, lag_25 = i-25
        assert out["global_active_power_diff_1h"].iloc[i] == pytest.approx((i - 1) - (i - 2))
        assert out["global_active_power_diff_24h"].iloc[i] == pytest.approx((i - 1) - (i - 25))

    def test_pct_change_guards_against_division_by_zero(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="h")
        values = np.zeros(30)
        values[10:] = 5.0  # lag_2 will be exactly 0 for the row right after the jump
        df = pd.DataFrame({"global_active_power": values}, index=idx)

        out = add_diff_features(df)

        assert not np.isinf(out["global_active_power_pct_change_1h"]).any()

    def test_current_row_target_never_affects_its_own_diff_features(self, synthetic_hourly_df):
        """Leakage guard: perturbing only row i's target value must not change
        row i's own diff features (they're built from lag_1/lag_2/lag_25, i.e.
        strictly earlier rows), but it MUST change row i+1's, since row i+1's
        lag_1 is row i.
        """
        baseline = add_diff_features(synthetic_hourly_df)

        perturbed_df = synthetic_hourly_df.copy()
        i = 150
        perturbed_df.iloc[i, perturbed_df.columns.get_loc("global_active_power")] += 1000.0
        perturbed = add_diff_features(perturbed_df)

        cols = [c for c in baseline.columns if c.startswith("global_active_power_diff")
                or c.startswith("global_active_power_pct_change")]

        for col in cols:
            assert baseline[col].iloc[i] == pytest.approx(perturbed[col].iloc[i]), col
            assert baseline[col].iloc[i + 1] != pytest.approx(perturbed[col].iloc[i + 1]), col


class TestAddAnomalyFlag:
    def test_flags_every_hour_of_a_flagged_day(self):
        flagged_day = FLAGGED_ANOMALY_DATES[0]
        idx = pd.date_range(flagged_day, periods=48, freq="h")  # flagged day + the next day
        df = pd.DataFrame({"global_active_power": range(48)}, index=idx)

        out = add_anomaly_flag(df)

        assert out["is_flagged_anomaly"].iloc[:24].tolist() == [1] * 24
        assert out["is_flagged_anomaly"].iloc[24:].tolist() == [0] * 24
