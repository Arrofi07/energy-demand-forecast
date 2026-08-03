import duckdb
import pandas as pd
import pytest

from src.data import duckdb_setup


@pytest.fixture
def tiny_processed_dir(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, freq in [("minute", "min"), ("hourly", "h"), ("daily", "D")]:
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=5, freq=freq),
            "value": range(5),
        })
        df.to_parquet(processed / f"{name}.parquet", index=False)
    return processed


@pytest.fixture
def patched_paths(tmp_path, tiny_processed_dir, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setattr(duckdb_setup, "PROCESSED_DIR", tiny_processed_dir)
    monkeypatch.setattr(duckdb_setup, "DB_PATH", db_path)
    return db_path


class TestBuildDuckdb:
    def test_creates_all_three_tables_with_expected_row_counts(self, patched_paths):
        duckdb_setup.build_duckdb()

        con = duckdb.connect(str(patched_paths), read_only=True)
        for table in ["minute", "hourly", "daily"]:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 5
        con.close()

    def test_rerunning_is_idempotent_not_additive(self, patched_paths):
        duckdb_setup.build_duckdb()
        duckdb_setup.build_duckdb()

        con = duckdb.connect(str(patched_paths), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM hourly").fetchone()[0]
        con.close()

        assert count == 5  # not 10 -- DROP TABLE IF EXISTS before each CREATE

    def test_reflects_updated_source_parquet_on_rerun(self, patched_paths, tiny_processed_dir):
        duckdb_setup.build_duckdb()

        updated = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "value": range(3),
        })
        updated.to_parquet(tiny_processed_dir / "hourly.parquet", index=False)
        duckdb_setup.build_duckdb()

        con = duckdb.connect(str(patched_paths), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM hourly").fetchone()[0]
        con.close()

        assert count == 3
