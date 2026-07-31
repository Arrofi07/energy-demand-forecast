import pandas as pd
import pytest

from src.evaluation import results_store


@pytest.fixture(autouse=True)
def isolated_results_dir(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp dir so tests never touch data/results."""
    monkeypatch.setattr(results_store, "RESULTS_DIR", tmp_path / "results")


def test_save_and_load_roundtrip():
    df = pd.DataFrame({
        "origin": pd.date_range("2020-01-01", periods=3, freq="h"),
        "horizon_step": [1, 2, 3],
        "method": ["naive"] * 3,
        "actual": [1.0, 2.0, 3.0],
        "forecast": [1.1, 2.1, 2.9],
    })

    results_store.save_results(df, "naive")
    loaded = results_store.load_results("naive")

    pd.testing.assert_frame_equal(loaded, df)


def test_save_creates_results_dir_if_missing():
    assert not results_store.RESULTS_DIR.exists()
    df = pd.DataFrame({"a": [1]})

    results_store.save_results(df, "x")

    assert results_store.RESULTS_DIR.exists()
    assert (results_store.RESULTS_DIR / "x.parquet").exists()


def test_load_all_results_concatenates_every_file():
    df_a = pd.DataFrame({"origin": [1], "method": ["a"]})
    df_b = pd.DataFrame({"origin": [2], "method": ["b"]})
    results_store.save_results(df_a, "model_a")
    results_store.save_results(df_b, "model_b")

    combined = results_store.load_all_results()

    assert len(combined) == 2
    assert set(combined["method"]) == {"a", "b"}
