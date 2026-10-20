"""Locating the CDR torsion reference data without shipping it in the package."""

import pytest

from rfproteina.metrics import cdr_torsions


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("RFD4_CDR_TORSION_DATA", raising=False)
    monkeypatch.setattr(cdr_torsions, "__file__", str(tmp_path / "module" / "cdr_torsions.py"))
    cdr_torsions.reference_data_dir.cache_clear()
    yield
    cdr_torsions.reference_data_dir.cache_clear()


def test_environment_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("RFD4_CDR_TORSION_DATA", str(tmp_path / "custom"))
    assert cdr_torsions.reference_data_dir() == tmp_path / "custom"


def test_installed_directory_is_used_without_downloading(monkeypatch, tmp_path):
    (tmp_path / "module" / "cdr_torsion_data").mkdir(parents=True)

    def _fail(*args, **kwargs):
        raise AssertionError("must not download when the data is installed")

    monkeypatch.setattr(cdr_torsions, "snapshot_download", _fail)
    assert cdr_torsions.reference_data_dir() == tmp_path / "module" / "cdr_torsion_data"


def test_missing_data_is_downloaded_from_the_dataset_repository(monkeypatch, tmp_path):
    calls = []

    def _download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path / "cache")

    monkeypatch.setattr(cdr_torsions, "snapshot_download", _download)
    path = cdr_torsions.reference_data_dir()
    assert path == tmp_path / "cache" / "metrics" / "cdr_torsion_data"
    assert calls == [
        (
            "nvidia/RFD4-Proteina",
            {"repo_type": "dataset", "allow_patterns": ["metrics/cdr_torsion_data/*"]},
        )
    ]


def test_download_failure_names_the_override(monkeypatch):
    def _offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(cdr_torsions, "snapshot_download", _offline)
    with pytest.raises(FileNotFoundError, match="RFD4_CDR_TORSION_DATA"):
        cdr_torsions.reference_data_dir()
