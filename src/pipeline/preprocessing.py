"""Reusable end-to-end preprocessing pipeline.

Phases 1-4 built this pipeline as four separate manual steps (see the
`Getting started` section of README.md, still valid for anyone who wants to
run them individually). This module wraps the same four steps -- download,
clean/resample, load into DuckDB, engineer features -- into a single
orchestrated call, so a scheduled or ad-hoc "refresh the data" job doesn't
need to know the internal step order. No preprocessing logic is duplicated
here; each step just calls the existing Phase 1-4 module directly.

Those Phase 1-4 modules all use paths relative to the project root (e.g.
`Path("data/raw")`), matching this project's documented "always run from
repo root" convention -- true for direct CLI use (`uv run python -m
src.data.download`), but not guaranteed for a caller like a notebook, whose
own working directory is wherever *it* lives. This module's entire purpose
is to be a single callable entrypoint regardless of caller, so it owns
making that convention hold (see `_in_project_root` below) rather than
silently writing a second copy of the dataset whever the caller happened to
be -- exactly the bug this fix replaced, caught by running this pipeline
from `notebooks/11_production_pipeline.ipynb` and finding a stray, fully
duplicated `notebooks/data/` afterwards.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from src.data.download import download_dataset
from src.data.duckdb_setup import build_duckdb
from src.data.load import build_all
from src.features.build_features import save_feature_set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@contextmanager
def _in_project_root():
    previous_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def run_preprocessing_pipeline(force_download: bool = False) -> None:
    """Run the full raw-to-features pipeline: download -> clean/resample ->
    DuckDB -> engineered features. Idempotent -- safe to re-run; each step
    overwrites its own output rather than appending to it.
    """
    with _in_project_root():
        print("=== Step 1/4: download ===")
        download_dataset(force=force_download)

        print("=== Step 2/4: clean, resample, write Parquet ===")
        build_all()

        print("=== Step 3/4: load Parquet into DuckDB ===")
        build_duckdb()

        print("=== Step 4/4: build engineered feature set ===")
        save_feature_set()

    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
