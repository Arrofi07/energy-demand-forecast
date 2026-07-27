"""Build a DuckDB database from the processed Parquet files."""

from pathlib import Path

import duckdb

PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/energy.duckdb")


def build_duckdb() -> None:
    con = duckdb.connect(str(DB_PATH))

    for table in ["minute", "hourly", "daily"]:
        parquet_path = PROCESSED_DIR / f"{table}.parquet"
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{parquet_path}')"
        )
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"Loaded table '{table}': {count:,} rows")

    con.close()
    print(f"DuckDB database ready at {DB_PATH}")


if __name__ == "__main__":
    build_duckdb()