"""Model comparison utilities for Phase 9 Part A -- the 24h benchmark evaluation.

Everything here operates on long-format forecast tables with the columns:

    origin, horizon_step, method, actual, forecast

The forecasts are already produced and saved by Phases 5-8. This module
does not refit any models. Instead, it evaluates existing forecasts from
several perspectives:

- seasonal performance
- chronological stability across test-period folds
- pairwise statistical significance using Diebold-Mariano tests
- worst-case forecast errors
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# Map each calendar month to the corresponding meteorological season.
# Keeping this mapping consistent with the feature-engineering pipeline
# ensures that evaluation results use the same season definitions.
SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}


def add_season(
    df: pd.DataFrame,
    date_col: str = "origin",
) -> pd.DataFrame:
    """Attach a season label based on the month of each forecast origin."""

    # Work on a copy so that adding the evaluation column does not
    # unexpectedly modify the original results DataFrame.
    df = df.copy()

    # Convert the origin timestamp into a categorical season label.
    df["season"] = df[date_col].dt.month.map(SEASON_MAP)

    return df


def assign_cv_folds(
    origins: pd.DatetimeIndex,
    n_folds: int = 4,
) -> pd.Series:
    """Split chronological forecast origins into contiguous test-period folds.

    This is a time-series stability check rather than traditional random
    k-fold cross-validation.

    The models have already been trained on the original training period.
    The test period is divided into chronological blocks to determine
    whether model rankings remain consistent over time.
    """

    # Sort and deduplicate origins to establish a canonical chronological
    # sequence before assigning folds.
    sorted_origins = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(origins).unique())
    )

    fold_of = {}

    # Split the test-period origins into contiguous chronological blocks.
    # Earlier timestamps always belong to earlier folds.
    for fold, chunk in enumerate(
        np.array_split(sorted_origins, n_folds)
    ):
        for ts in chunk:
            fold_of[ts] = fold

    return pd.Series(fold_of, name="fold")


def _loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    kind: str = "absolute",
) -> np.ndarray:
    """Calculate the selected pointwise forecast loss."""

    # Absolute error is robust to large individual errors and corresponds
    # to the MAE-style evaluation used elsewhere in the benchmark.
    if kind == "absolute":
        return np.abs(actual - forecast)

    # Squared error penalizes large forecasting mistakes more strongly
    # and corresponds to the MSE/RMSE family of metrics.
    if kind == "squared":
        return (actual - forecast) ** 2

    raise ValueError(f"Unknown loss kind: {kind!r}")


def per_origin_loss(
    df: pd.DataFrame,
    method: str,
    loss: str = "absolute",
) -> pd.Series:
    """Calculate the mean forecast loss for each origin for one method.

    The 24 horizon-specific errors are averaged into one loss value per
    origin. This makes each forecast origin one observation for subsequent
    chronological fold analysis and statistical testing.
    """

    # Keep only results belonging to the requested forecasting method
    # and remove predictions where either the actual or forecast is missing.
    d = (
        df[df["method"] == method]
        .dropna(subset=["actual", "forecast"])
        .copy()
    )

    # Calculate the selected loss for every forecasted value.
    d["loss"] = _loss(
        d["actual"].to_numpy(),
        d["forecast"].to_numpy(),
        loss,
    )

    # Average the horizon-specific errors so that each origin contributes
    # exactly one observation to the subsequent statistical analysis.
    return d.groupby("origin")["loss"].mean()


@dataclass
class DMResult:
    """Container for the result of a pairwise Diebold-Mariano test."""

    method_a: str
    method_b: str
    mean_loss_diff: float
    dm_statistic: float
    p_value: float
    n_origins: int


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Compare two aligned forecast-loss series using the DM test.

    The null hypothesis is that the two forecasting methods have equal
    predictive accuracy.

    A positive mean loss difference means method A has higher average
    loss than method B. The test uses a Harvey/Leybourne/Newbold-style
    small-sample correction and a Student's-t reference distribution.
    """

    # Construct the loss differential:
    #
    # d_t > 0  -> method B performed better at origin t
    # d_t < 0  -> method A performed better at origin t
    d = np.asarray(loss_a) - np.asarray(loss_b)

    n = len(d)
    d_bar = d.mean()

    # Estimate the variance of the loss differential.
    # This is the lag-0 component of the long-run variance.
    gamma_0 = np.var(d, ddof=0)
    long_run_var = gamma_0

    # Add autocovariance terms when forecast windows overlap.
    # For h=1, this loop is skipped and only the contemporaneous
    # variance is used.
    for k in range(1, h):
        gamma_k = np.mean(
            (d[k:] - d_bar) * (d[:-k] - d_bar)
        )
        long_run_var += 2 * gamma_k

    # Standard Diebold-Mariano statistic.
    dm_stat = d_bar / np.sqrt(long_run_var / n)

    # Harvey/Leybourne/Newbold small-sample correction.
    # This adjusts the statistic when the number of forecast origins
    # is not large enough for the asymptotic approximation to be ideal.
    correction = np.sqrt(
        (
            n
            + 1
            - 2 * h
            + h * (h - 1) / n
        )
        / n
    )

    dm_stat_corrected = dm_stat * correction

    # Use a two-sided Student's-t distribution to test whether the
    # corrected statistic differs significantly from zero.
    p_value = 2 * (
        1
        - stats.t.cdf(
            np.abs(dm_stat_corrected),
            df=n - 1,
        )
    )

    return float(dm_stat_corrected), float(p_value)


def pairwise_dm_tests(
    df: pd.DataFrame,
    methods: list[str],
    loss: str = "absolute",
    h: int = 1,
) -> pd.DataFrame:
    """Run pairwise Diebold-Mariano tests for the selected methods.

    Each pair is evaluated only on forecast origins available for both
    methods. This keeps the comparison paired and avoids comparing
    methods on different samples.
    """

    # Calculate one loss series per forecasting method.
    # The resulting index is the forecast origin.
    losses = {
        method: per_origin_loss(df, method, loss)
        for method in methods
    }

    records = []

    # Generate every unique method pair without comparing a method
    # against itself or testing the same pair twice.
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:

            # Keep only forecast origins available for both models.
            # This is important when some models drop origins because
            # of missing features or observations.
            common = losses[method_a].index.intersection(
                losses[method_b].index
            )

            a = losses[method_a].loc[common].to_numpy()
            b = losses[method_b].loc[common].to_numpy()

            # Test whether the two methods have significantly different
            # predictive accuracy on their shared forecast origins.
            dm_stat, p_value = diebold_mariano_test(
                a,
                b,
                h=h,
            )

            records.append(
                DMResult(
                    method_a=method_a,
                    method_b=method_b,

                    # Positive value means method A has higher average
                    # loss than method B.
                    mean_loss_diff=float(
                        a.mean() - b.mean()
                    ),

                    dm_statistic=dm_stat,
                    p_value=p_value,
                    n_origins=len(common),
                )
            )

    return pd.DataFrame.from_records(
        [result.__dict__ for result in records]
    )


def worst_cases(
    df: pd.DataFrame,
    method: str,
    n: int = 10,
    loss: str = "absolute",
) -> pd.DataFrame:
    """Return the n largest individual forecast errors for one method."""

    # Select valid predictions for the requested method.
    d = (
        df[df["method"] == method]
        .dropna(subset=["actual", "forecast"])
        .copy()
    )

    # Calculate the error associated with every
    # origin/horizon-step prediction.
    d["loss"] = _loss(
        d["actual"].to_numpy(),
        d["forecast"].to_numpy(),
        loss,
    )

    # Sort from largest to smallest error and return the worst cases.
    return (
        d.sort_values("loss", ascending=False)
        .head(n)
    )