from pathlib import Path

import pytest

from src.pipeline.preprocessing import PROJECT_ROOT, _in_project_root


class TestInProjectRoot:
    """Regression coverage for a real bug: `run_preprocessing_pipeline()`
    orchestrates Phase 1-4 modules that use paths relative to the project
    root, so calling it from a different cwd (e.g. a notebook under
    `notebooks/`) silently wrote a second, fully duplicated copy of the
    dataset there instead of touching the real `data/` directory. These
    tests target the exact mechanism (`_in_project_root`) that fixes it,
    without paying the cost of running the full pipeline.
    """

    def test_cwd_is_project_root_inside_the_context(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with _in_project_root():
            assert Path.cwd() == PROJECT_ROOT

    def test_original_cwd_is_restored_after_the_context(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with _in_project_root():
            pass
        assert Path.cwd() == tmp_path

    def test_original_cwd_is_restored_even_if_the_block_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            with _in_project_root():
                raise ValueError("boom")
        assert Path.cwd() == tmp_path
