import pandas as pd
import pytest

from src.pipeline.inference_features import LOOKBACK_HOURS, build_latest_features


class TestBuildLatestFeatures:
    def test_returns_single_row_at_as_of(self, synthetic_hourly_df):
        as_of = synthetic_hourly_df.index[-1]
        row = build_latest_features(synthetic_hourly_df, as_of)
        assert list(row.index) == [as_of]

    def test_no_nans_when_window_is_long_enough(self, synthetic_hourly_df):
        # fixture provides 30 days (720h) of history, comfortably more than
        # LOOKBACK_HOURS -- every lag/rolling window should be fully populated.
        as_of = synthetic_hourly_df.index[-1]
        row = build_latest_features(synthetic_hourly_df, as_of)
        assert not row.isna().any(axis=1).item()

    def test_raises_if_as_of_not_in_window(self, synthetic_hourly_df):
        missing_ts = synthetic_hourly_df.index[-1] + pd.Timedelta(hours=1)
        with pytest.raises(ValueError, match="not found"):
            build_latest_features(synthetic_hourly_df, missing_ts)

    def test_raises_if_window_too_short_for_full_lookback(self, synthetic_hourly_df):
        short_df = synthetic_hourly_df.iloc[-50:]  # well under the 168h rolling window
        as_of = short_df.index[-1]
        with pytest.raises(ValueError, match="NaNs"):
            build_latest_features(short_df, as_of)

    def test_lag_1h_matches_previous_hours_actual_value(self, synthetic_hourly_df):
        as_of = synthetic_hourly_df.index[-1]
        row = build_latest_features(synthetic_hourly_df, as_of)
        expected = synthetic_hourly_df["global_active_power"].loc[as_of - pd.Timedelta(hours=1)]
        assert row["global_active_power_lag_1h"].item() == pytest.approx(expected)

    def test_lag_24h_matches_value_one_day_earlier(self, synthetic_hourly_df):
        as_of = synthetic_hourly_df.index[-1]
        row = build_latest_features(synthetic_hourly_df, as_of)
        expected = synthetic_hourly_df["global_active_power"].loc[as_of - pd.Timedelta(hours=24)]
        assert row["global_active_power_lag_24h"].item() == pytest.approx(expected)

    def test_lookback_hours_constant_is_comfortably_sufficient(self, synthetic_hourly_df):
        """LOOKBACK_HOURS includes a 24h margin above the true minimum (168h,
        the longest rolling/lag window) -- this pins that it's actually
        enough, not just documented as such.
        """
        as_of = synthetic_hourly_df.index[-1]
        window = synthetic_hourly_df.loc[as_of - pd.Timedelta(hours=LOOKBACK_HOURS):as_of]
        row = build_latest_features(window, as_of)
        assert not row.isna().any(axis=1).item()

    def test_exactly_168_hours_is_the_true_minimum_sufficient_window(self, synthetic_hourly_df):
        """The longest window anywhere (roll_*_168h, lag_168h) needs the raw
        value at exactly `as_of - 168h` -- so a window spanning exactly 168
        hours before `as_of` (169 rows including `as_of` itself) is the tight
        boundary, one hour tighter than the `LOOKBACK_HOURS` constant's
        deliberate margin.
        """
        as_of = synthetic_hourly_df.index[-1]
        window = synthetic_hourly_df.loc[as_of - pd.Timedelta(hours=168):as_of]
        row = build_latest_features(window, as_of)
        assert not row.isna().any(axis=1).item()

    def test_one_hour_short_of_the_true_minimum_is_not_sufficient(self, synthetic_hourly_df):
        as_of = synthetic_hourly_df.index[-1]
        window = synthetic_hourly_df.loc[as_of - pd.Timedelta(hours=167):as_of]
        with pytest.raises(ValueError, match="NaNs"):
            build_latest_features(window, as_of)


@pytest.mark.slow
class TestBuildLatestFeaturesAgainstFullHistory:
    """Integration check: a bounded recent window should reproduce exactly
    what the full-history `hourly_features` DuckDB table already has for the
    same timestamp -- the assumption the whole inference-features module is
    built on, verified against real data rather than just asserted.
    """

    def test_matches_hourly_features_table(self):
        import duckdb

        from src.pipeline.inference_features import DB_PATH, load_recent_hourly

        con = duckdb.connect(str(DB_PATH), read_only=True)
        full = con.execute("SELECT * FROM hourly_features ORDER BY datetime").df()
        con.close()
        full["datetime"] = pd.to_datetime(full["datetime"])
        full = full.set_index("datetime")

        # A timestamp confirmed (by direct inspection) to have zero NaNs in
        # its preceding 200 hours -- picking an arbitrary "safely clear of
        # gaps" date isn't reliable, since Phase 1 only enumerated the 3
        # *longest* of 7 total long gaps, not all of them: an earlier version
        # of this test picked 2009-06-15 assuming it was clean, and it
        # wasn't (a real, previously-unlisted ~51h gap sits right before it,
        # 2009-06-13 04:00 to 2009-06-15 06:00), correctly triggering
        # build_latest_features's NaN guard rather than a bug in the guard.
        as_of = pd.Timestamp("2008-06-01 00:00:00")
        expected = full.loc[[as_of]]

        recent_df = load_recent_hourly(as_of)
        actual = build_latest_features(recent_df, as_of)

        compare_cols = [
            "global_active_power_lag_1h",
            "global_active_power_lag_24h",
            "global_active_power_lag_168h",
            "global_active_power_roll_mean_24h",
            "global_active_power_roll_std_168h",
            "global_active_power_diff_24h",
            "hour_sin",
            "dayofweek_cos",
            "is_weekend",
        ]
        for col in compare_cols:
            assert actual[col].item() == pytest.approx(expected[col].item()), col
