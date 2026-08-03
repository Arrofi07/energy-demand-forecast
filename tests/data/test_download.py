import zipfile

import pytest

from src.data import download


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    zip_path = raw_dir / "household_power_consumption.zip"
    txt_path = raw_dir / "household_power_consumption.txt"
    monkeypatch.setattr(download, "RAW_DIR", raw_dir)
    monkeypatch.setattr(download, "ZIP_PATH", zip_path)
    monkeypatch.setattr(download, "TXT_PATH", txt_path)
    return raw_dir, zip_path, txt_path


def _fake_zip_response(tmp_path, arcname, content=b"fake data"):
    source = tmp_path / "source.txt"
    source.write_bytes(content)
    zip_path = tmp_path / "fake.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(source, arcname=arcname)

    class FakeResponse:
        def __init__(self):
            self.content = zip_path.read_bytes()

        def raise_for_status(self):
            pass

    return FakeResponse()


class TestDownloadDataset:
    def test_skips_download_if_file_already_exists(self, patched_paths, monkeypatch):
        raw_dir, _, txt_path = patched_paths
        raw_dir.mkdir(parents=True)
        txt_path.write_text("existing data")

        def fail_if_called(*a, **k):
            raise AssertionError("should not call requests.get when the file already exists")

        monkeypatch.setattr(download.requests, "get", fail_if_called)

        result = download.download_dataset()

        assert result == txt_path
        assert txt_path.read_text() == "existing data"

    def test_downloads_and_extracts_when_missing(self, patched_paths, monkeypatch, tmp_path):
        _, _, txt_path = patched_paths
        fake_response = _fake_zip_response(tmp_path, "household_power_consumption.txt", b"real content")

        calls = {}

        def fake_get(url, timeout):
            calls["url"], calls["timeout"] = url, timeout
            return fake_response

        monkeypatch.setattr(download.requests, "get", fake_get)

        result = download.download_dataset()

        assert result == txt_path
        assert txt_path.read_bytes() == b"real content"
        assert calls["url"] == download.DATA_URL

    def test_force_redownloads_even_if_file_exists(self, patched_paths, monkeypatch, tmp_path):
        raw_dir, _, txt_path = patched_paths
        raw_dir.mkdir(parents=True)
        txt_path.write_text("stale data")
        fake_response = _fake_zip_response(tmp_path, "household_power_consumption.txt", b"fresh data")
        monkeypatch.setattr(download.requests, "get", lambda url, timeout: fake_response)

        download.download_dataset(force=True)

        assert txt_path.read_bytes() == b"fresh data"

    def test_raises_if_extraction_does_not_produce_expected_file(self, patched_paths, monkeypatch, tmp_path):
        fake_response = _fake_zip_response(tmp_path, "wrong_name.txt")
        monkeypatch.setattr(download.requests, "get", lambda url, timeout: fake_response)

        with pytest.raises(FileNotFoundError):
            download.download_dataset()
