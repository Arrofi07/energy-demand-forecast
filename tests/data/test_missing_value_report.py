import numpy as np
import pandas as pd

from src.data.missing_value_report import analyze_gaps


def test_identifies_and_sizes_distinct_gaps():
    idx = pd.date_range("2020-01-01", periods=10, freq="min")
    # rows 2-3 missing (2 rows), row 7 missing (1 row)
    values = [1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0]
    df = pd.DataFrame({
        "datetime": idx,
        "global_active_power": values,
        "global_reactive_power": [1.0] * 10,
        "voltage": [230.0] * 10,
        "global_intensity": [1.0] * 10,
        "sub_metering_1": [0.0] * 10,
        "sub_metering_2": [0.0] * 10,
        "sub_metering_3": [0.0] * 10,
    })

    gaps = analyze_gaps(df)

    assert len(gaps) == 2
    assert set(gaps["rows"]) == {2, 1}
    # sorted longest-first: the 2-row gap should come before the 1-row gap
    assert gaps["rows"].iloc[0] == 2

    two_row_gap = gaps.loc[gaps["rows"] == 2].iloc[0]
    assert two_row_gap["start"] == idx[2]
    assert two_row_gap["end"] == idx[3]
    assert two_row_gap["duration"] == pd.Timedelta(minutes=2)


def test_no_missing_values_returns_empty():
    idx = pd.date_range("2020-01-01", periods=5, freq="min")
    df = pd.DataFrame({
        "datetime": idx,
        "global_active_power": [1.0] * 5,
        "global_reactive_power": [1.0] * 5,
        "voltage": [230.0] * 5,
        "global_intensity": [1.0] * 5,
        "sub_metering_1": [0.0] * 5,
        "sub_metering_2": [0.0] * 5,
        "sub_metering_3": [0.0] * 5,
    })

    gaps = analyze_gaps(df)

    assert len(gaps) == 0


def test_row_counted_as_missing_if_any_measurement_column_is_nan():
    idx = pd.date_range("2020-01-01", periods=3, freq="min")
    df = pd.DataFrame({
        "datetime": idx,
        "global_active_power": [1.0, 1.0, 1.0],
        "global_reactive_power": [1.0, np.nan, 1.0],  # only this column missing at row 1
        "voltage": [230.0] * 3,
        "global_intensity": [1.0] * 3,
        "sub_metering_1": [0.0] * 3,
        "sub_metering_2": [0.0] * 3,
        "sub_metering_3": [0.0] * 3,
    })

    gaps = analyze_gaps(df)

    assert len(gaps) == 1
    assert gaps.iloc[0]["rows"] == 1
