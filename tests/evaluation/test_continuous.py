import numpy as np
import pandas as pd

from src.evaluation.continuous import to_backtest_format


def test_long_format_alignment_with_origins_and_horizon():
    idx = pd.date_range("2020-01-01", periods=24 * 5, freq="h")
    forecast = pd.Series(np.arange(len(idx), dtype=float) * 10, index=idx)
    actual = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    origins = pd.DatetimeIndex([idx[24], idx[48]])

    result = to_backtest_format(forecast, actual, origins, horizon=24, method_name="prophet")

    assert list(result.columns) == ["origin", "horizon_step", "method", "actual", "forecast"]
    assert len(result) == len(origins) * 24
    assert (result["method"] == "prophet").all()
    assert result["horizon_step"].tolist() == list(range(1, 25)) * len(origins)

    # spot check one row: origin[0] + step 1 -> actual/forecast at that timestamp
    row = result.iloc[0]
    ts = origins[0] + pd.Timedelta(hours=1)
    assert row["actual"] == actual.loc[ts]
    assert row["forecast"] == forecast.loc[ts]


def test_missing_timestamps_become_nan_via_reindex():
    idx = pd.date_range("2020-01-01", periods=24 * 2, freq="h")  # 48 hourly points
    forecast = pd.Series(np.arange(len(idx)), index=idx)
    actual = pd.Series(np.arange(len(idx)), index=idx)
    # future window (origin+1 .. origin+24) runs past idx's end, so reindex
    # must produce NaN for the timestamps that fall outside the series
    origin = idx[40]

    result = to_backtest_format(forecast, actual, pd.DatetimeIndex([origin]), horizon=24, method_name="m")

    assert result["actual"].isna().sum() > 0
    assert result["forecast"].isna().sum() > 0
    # the steps that do fall within range should still be populated
    assert result.iloc[0]["actual"] == actual.iloc[41]
