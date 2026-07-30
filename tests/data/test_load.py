import numpy as np
import pandas as pd

from src.data.load import (
    MAX_GAP_FOR_INTERPOLATION,
    MEAN_COLS,
    SUM_COLS,
    interpolate_short_gaps,
    resample,
)


class TestInterpolateShortGaps:
    def test_short_gap_gets_filled(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="min")
        values = [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({"datetime": idx, "global_active_power": values,
                            "global_reactive_power": values, "voltage": values,
                            "global_intensity": values, "sub_metering_1": values,
                            "sub_metering_2": values, "sub_metering_3": values})

        out = interpolate_short_gaps(df)

        assert not out["global_active_power"].isna().any()
        # linear interpolation between 2.0 (idx1) and 5.0 (idx4)
        assert out["global_active_power"].iloc[2] == 3.0
        assert out["global_active_power"].iloc[3] == 4.0

    def test_gap_longer_than_limit_stays_nan(self):
        """interpolate(limit=N) fills up to N consecutive NaNs walking forward
        from the last valid value, so a gap longer than the limit is only
        partially filled -- the tail nearest the *next* valid value, beyond
        reach of the limit, stays NaN.
        """
        limit_minutes = int(MAX_GAP_FOR_INTERPOLATION / pd.Timedelta(minutes=1))
        gap_minutes = limit_minutes + 30
        idx = pd.date_range("2020-01-01", periods=gap_minutes + 2, freq="min")
        values = [np.nan] * (gap_minutes + 2)
        values[0] = 1.0
        values[-1] = 2.0
        df = pd.DataFrame({"datetime": idx, "global_active_power": values,
                            "global_reactive_power": values, "voltage": values,
                            "global_intensity": values, "sub_metering_1": values,
                            "sub_metering_2": values, "sub_metering_3": values})

        out = interpolate_short_gaps(df)

        # within reach of the limit (right after the leading valid value): filled
        assert not np.isnan(out["global_active_power"].iloc[limit_minutes])
        # beyond the limit, closer to the trailing valid value: stays NaN
        assert np.isnan(out["global_active_power"].iloc[limit_minutes + 10])

    def test_gap_at_the_edge_is_not_interpolated(self):
        """limit_area='inside' means leading/trailing NaNs are left alone."""
        idx = pd.date_range("2020-01-01", periods=5, freq="min")
        values = [np.nan, np.nan, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"datetime": idx, "global_active_power": values,
                            "global_reactive_power": values, "voltage": values,
                            "global_intensity": values, "sub_metering_1": values,
                            "sub_metering_2": values, "sub_metering_3": values})

        out = interpolate_short_gaps(df)

        assert out["global_active_power"].iloc[:2].isna().all()


class TestResample:
    def test_mean_cols_are_averaged_and_sum_cols_summed(self):
        idx = pd.date_range("2020-01-01 00:00", periods=120, freq="min")
        df = pd.DataFrame({"datetime": idx})
        for col in MEAN_COLS:
            df[col] = 2.0
        for col in SUM_COLS:
            df[col] = 1.0

        out = resample(df, "h")

        assert (out[MEAN_COLS[0]] == 2.0).all()
        assert (out[SUM_COLS[0]] == 60.0).all()  # 60 one-minute rows summed per hour

    def test_all_nan_period_sums_to_nan_not_zero(self):
        """min_count=1 must make an all-missing hour NaN, not silently 0 --
        otherwise a data outage would look like zero energy consumption.
        """
        idx = pd.date_range("2020-01-01 00:00", periods=60, freq="min")
        df = pd.DataFrame({"datetime": idx})
        for col in MEAN_COLS:
            df[col] = np.nan
        for col in SUM_COLS:
            df[col] = np.nan

        out = resample(df, "h")

        assert out[SUM_COLS[0]].isna().all()
        assert out[MEAN_COLS[0]].isna().all()
