"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_index():
    """30 days of hourly timestamps, no gaps."""
    return pd.date_range("2020-01-01", periods=24 * 30, freq="h")


@pytest.fixture
def synthetic_power_series(hourly_index):
    """Deterministic synthetic hourly power series with a daily rhythm.

    Seeded RNG so tests are reproducible. Values are strictly positive,
    mimicking the always-on-appliances floor in the real dataset.
    """
    rng = np.random.default_rng(42)
    hours = hourly_index.hour.to_numpy()
    daily_pattern = 1.5 + np.sin(2 * np.pi * (hours - 6) / 24)
    noise = rng.normal(0, 0.05, size=len(hourly_index))
    values = daily_pattern + noise
    return pd.Series(values, index=hourly_index, name="global_active_power")


@pytest.fixture
def synthetic_hourly_df(hourly_index):
    """Minimal stand-in for the 'hourly' DuckDB table: datetime index +
    global_active_power, using a simple increasing sequence so lag/rolling/
    diff relationships are easy to reason about by hand in tests.
    """
    n = len(hourly_index)
    df = pd.DataFrame(
        {"global_active_power": np.arange(n, dtype=float)},
        index=hourly_index,
    )
    df.index.name = "datetime"
    return df
