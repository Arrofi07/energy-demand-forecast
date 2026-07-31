import pandas as pd

from src.evaluation.splits import get_train_test_split


def test_split_point_matches_expected_fraction():
    idx = pd.date_range("2020-01-01", periods=1000, freq="h")
    series = pd.Series(range(1000), index=idx)

    split = get_train_test_split(series, test_frac=0.2)

    expected_idx = int(1000 * 0.8)
    assert split == idx[expected_idx].normalize()


def test_split_point_is_normalized_to_midnight():
    idx = pd.date_range("2020-01-01 00:00", periods=500, freq="h")
    series = pd.Series(range(500), index=idx)

    split = get_train_test_split(series, test_frac=0.3)

    assert split == split.normalize()
    assert split.hour == 0 and split.minute == 0 and split.second == 0


def test_default_test_frac_is_point_two():
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    series = pd.Series(range(100), index=idx)

    assert get_train_test_split(series) == get_train_test_split(series, test_frac=0.2)
