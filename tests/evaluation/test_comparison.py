import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import (
    add_season,
    assign_cv_folds,
    diebold_mariano_test,
    pairwise_dm_tests,
    per_origin_loss,
    worst_cases,
)


@pytest.fixture
def results_df():
    """Two methods over the same 10 daily origins x 3 horizon steps, method_b
    deliberately built to have a much larger, consistent error than method_a
    so pairwise comparisons have an unambiguous direction to check against.
    Method 'partial' only covers a subset of origins, mimicking LightGBM/LSTM
    dropping origins near data gaps.
    """
    origins = pd.date_range("2020-01-01", periods=10, freq="24h")
    rows = []
    for origin in origins:
        for h in [1, 2, 3]:
            actual = 1.0
            rows.append({"origin": origin, "horizon_step": h, "method": "method_a", "actual": actual, "forecast": actual + 0.1})
            rows.append({"origin": origin, "horizon_step": h, "method": "method_b", "actual": actual, "forecast": actual + 1.0})
    for origin in origins[:4]:
        for h in [1, 2, 3]:
            rows.append({"origin": origin, "horizon_step": h, "method": "partial", "actual": 1.0, "forecast": 1.05})
    return pd.DataFrame.from_records(rows)


class TestAddSeason:
    def test_maps_month_to_expected_season(self):
        df = pd.DataFrame({"origin": pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"])})
        result = add_season(df)
        assert list(result["season"]) == ["winter", "spring", "summer", "fall"]


class TestAssignCvFolds:
    def test_folds_are_contiguous_and_chronological(self):
        origins = pd.date_range("2020-01-01", periods=12, freq="24h")
        folds = assign_cv_folds(origins, n_folds=4)

        assert set(folds.unique()) == {0, 1, 2, 3}
        # Fold 0 must contain only earlier timestamps than fold 3.
        assert folds[origins[0]] == 0
        assert folds[origins[-1]] == 3
        # Monotonic non-decreasing fold index as time advances.
        ordered = folds.loc[sorted(folds.index)]
        assert list(ordered) == sorted(ordered)

    def test_covers_every_origin_exactly_once(self):
        origins = pd.date_range("2020-01-01", periods=10, freq="24h")
        folds = assign_cv_folds(origins, n_folds=3)
        assert set(folds.index) == set(origins)
        assert len(folds) == len(origins)


class TestPerOriginLoss:
    def test_averages_across_horizon_steps(self, results_df):
        loss = per_origin_loss(results_df, "method_a", loss="absolute")
        assert loss.tolist() == pytest.approx([0.1] * len(loss))

    def test_only_includes_requested_method(self, results_df):
        loss = per_origin_loss(results_df, "partial", loss="absolute")
        assert len(loss) == 4


class TestDieboldMariano:
    def test_no_systematic_difference_is_not_significant(self):
        """Two independent, identically-distributed loss series -- any
        observed difference is sampling noise, so the test should not
        reject the null of equal predictive accuracy.
        """
        rng = np.random.default_rng(0)
        loss_a = rng.normal(1.0, 0.1, size=200)
        loss_b = rng.normal(1.0, 0.1, size=200)
        stat, p_value = diebold_mariano_test(loss_a, loss_b)
        assert p_value > 0.05

    def test_identical_series_has_undefined_statistic(self):
        """Zero variance in the loss differential makes the DM statistic
        mathematically undefined (division by zero), not zero.
        """
        a = np.array([1.0, 2.0, 3.0, 4.0])
        stat, p_value = diebold_mariano_test(a, a.copy())
        assert np.isnan(stat)

    def test_systematically_better_model_gives_significant_negative_statistic(self):
        rng = np.random.default_rng(0)
        loss_a = rng.normal(0.1, 0.01, size=200)  # consistently small error
        loss_b = rng.normal(1.0, 0.01, size=200)  # consistently large error
        stat, p_value = diebold_mariano_test(loss_a, loss_b)

        assert stat < 0  # method_a's loss is significantly lower
        assert p_value < 0.05


class TestPairwiseDmTests:
    def test_pairs_only_common_origins(self, results_df):
        result = pairwise_dm_tests(results_df, ["method_a", "method_b", "partial"])
        row = result[
            ((result["method_a"] == "method_a") & (result["method_b"] == "partial"))
            | ((result["method_a"] == "partial") & (result["method_b"] == "method_a"))
        ].iloc[0]
        assert row["n_origins"] == 4

    def test_returns_one_row_per_pair(self, results_df):
        result = pairwise_dm_tests(results_df, ["method_a", "method_b", "partial"])
        assert len(result) == 3  # 3 choose 2


class TestWorstCases:
    def test_returns_n_rows_sorted_descending_by_loss(self, results_df):
        result = worst_cases(results_df, "method_b", n=5)
        assert len(result) == 5
        assert (result["loss"].diff().dropna() <= 0).all()

    def test_only_includes_requested_method(self, results_df):
        result = worst_cases(results_df, "partial", n=100)
        assert (result["method"] == "partial").all()
        assert len(result) == 12  # 4 origins x 3 horizon steps
