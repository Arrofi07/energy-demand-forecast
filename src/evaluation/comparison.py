"""Model comparison utilities for Phase 9 Part A -- the 24h benchmark evaluation.

Everything here operates on the long-format (origin, horizon_step, method,
actual, forecast) tables already produced and saved by Phases 5-8
(`src/evaluation/results_store.py`). No model is refit here -- this module
only re-analyzes forecasts that already exist, from a few different angles:
seasonal breakdown, chronological stability (a CV-style check), pairwise
statistical significance (Diebold-Mariano), and worst-case inspection.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def add_season(df: pd.DataFrame, date_col: str = "origin") -> pd.DataFrame:
    """Attach a `season` column derived from the origin's calendar month.

    Same month->season mapping `src/features/build_features.py` uses, so
    "winter" here means the same thing it did back in the EDA/feature phases.
    """
    df = df.copy()
    df["season"] = df[date_col].dt.month.map(SEASON_MAP)
    return df


def assign_cv_folds(origins: pd.DatetimeIndex, n_folds: int = 4) -> pd.Series:
    """Bucket a canonical, sorted origin index into `n_folds` contiguous,
    chronological blocks (fold 0 = earliest).

    This is a walk-forward-style check, not classic k-fold CV: models were
    already fit once on the single Phase 5 train/test split, so folds here
    slice the *test period* into independent time chunks to check whether
    the model ranking found on the whole test period is stable across
    different chunks of it, rather than an artifact of averaging over the
    entire window.
    """
    sorted_origins = pd.DatetimeIndex(sorted(pd.DatetimeIndex(origins).unique()))
    fold_of = {}
    for fold, chunk in enumerate(np.array_split(sorted_origins, n_folds)):
        for ts in chunk:
            fold_of[ts] = fold
    return pd.Series(fold_of, name="fold")


def _loss(actual: np.ndarray, forecast: np.ndarray, kind: str = "absolute") -> np.ndarray:
    if kind == "absolute":
        return np.abs(actual - forecast)
    if kind == "squared":
        return (actual - forecast) ** 2
    raise ValueError(f"Unknown loss kind: {kind!r}")


def per_origin_loss(df: pd.DataFrame, method: str, loss: str = "absolute") -> pd.Series:
    """Mean loss per origin (averaged across the 24 horizon steps), for one method.

    Averaging across horizon steps collapses each origin's forecast to a
    single number, which is what both the CV-fold breakdown and the
    Diebold-Mariano test below treat as "one observation in time".
    """
    d = df[df["method"] == method].dropna(subset=["actual", "forecast"]).copy()
    d["loss"] = _loss(d["actual"].to_numpy(), d["forecast"].to_numpy(), loss)
    return d.groupby("origin")["loss"].mean()


@dataclass
class DMResult:
    method_a: str
    method_b: str
    mean_loss_diff: float
    dm_statistic: float
    p_value: float
    n_origins: int


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test statistic and p-value for two aligned loss series.

    Standard DM statistic with the Harvey/Leybourne/Newbold (1997) small-
    sample correction and a Student's-t reference distribution (their
    recommendation over the asymptotic normal for the origin counts this
    project has, n~70-290). `h` truncates the long-run variance estimator's
    autocovariance sum at h-1 lags -- the correct choice when successive
    loss observations come from overlapping forecast windows. It's left at
    the default of 1 (no autocovariance correction beyond lag 0) because
    `get_forecast_origins` spaces origins exactly 24h apart with a 24h
    horizon, so windows never overlap.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_bar = d.mean()

    gamma_0 = np.var(d, ddof=0)
    long_run_var = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        long_run_var += 2 * gamma_k

    dm_stat = d_bar / np.sqrt(long_run_var / n)

    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    return float(dm_stat_corrected), float(p_value)


def pairwise_dm_tests(df: pd.DataFrame, methods: list[str], loss: str = "absolute", h: int = 1) -> pd.DataFrame:
    """Run the Diebold-Mariano test for every pair of methods in `methods`.

    Each pair is compared only on the origins both methods actually cover
    (an inner join) -- LightGBM/LSTM drop origins near data gaps that
    SARIMA/Prophet don't, so this keeps every pairwise test properly paired
    rather than silently comparing mismatched samples.
    """
    losses = {m: per_origin_loss(df, m, loss) for m in methods}

    records = []
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            common = losses[method_a].index.intersection(losses[method_b].index)
            a = losses[method_a].loc[common].to_numpy()
            b = losses[method_b].loc[common].to_numpy()

            dm_stat, p_value = diebold_mariano_test(a, b, h=h)
            records.append(DMResult(
                method_a=method_a,
                method_b=method_b,
                mean_loss_diff=float(a.mean() - b.mean()),
                dm_statistic=dm_stat,
                p_value=p_value,
                n_origins=len(common),
            ))

    return pd.DataFrame.from_records([r.__dict__ for r in records])


def worst_cases(df: pd.DataFrame, method: str, n: int = 10, loss: str = "absolute") -> pd.DataFrame:
    """The `n` largest-error (origin, horizon_step) predictions for one method."""
    d = df[df["method"] == method].dropna(subset=["actual", "forecast"]).copy()
    d["loss"] = _loss(d["actual"].to_numpy(), d["forecast"].to_numpy(), loss)
    return d.sort_values("loss", ascending=False).head(n)
