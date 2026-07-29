"""Shared train/test split, so every phase (baselines through Phase 9)
evaluates on the exact same held-out period for a fair, consistent comparison.
"""

import pandas as pd


def get_train_test_split(series: pd.Series, test_frac: float = 0.2) -> pd.Timestamp:
    """Return the timestamp marking the start of the test period.

    Rounded down to midnight so forecast origins land on clean day
    boundaries -- matching the day-ahead business framing (one forecast
    issued per day, at a fixed time).
    """
    split_idx = int(len(series) * (1 - test_frac))
    split_point = series.index[split_idx]
    return split_point.normalize()