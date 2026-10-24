"""Tests for the Colab bundle exporter.

The archive is built from a throwaway repository tree, so the test proves the
allowlist *structurally*: secrets, checkpoints, the SQLite cache and the raw
data are present in the fake checkout and must not appear in the zip. The
requirements file is checked against its real contents.
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from munasib.core.exceptions import DataError

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_colab_bundle.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENTS = _REPO_ROOT / "requirements-colab.txt"
_REVIEW_COLUMNS = ("review_id", "asin", "split", "review_text_clean", "sentiment_label_weak")


@pytest.fixture(scope="module")
def script():
    """Import ``scripts/export_colab_bundle.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_export_colab_bundle", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses defined in the script resolve ``cls.__module__`` through
    # ``sys.modules``, so the module must be registered before execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_repo(tmp_path: Path) -> Path:
    """Build a repository tree containing every file the bundle must skip.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The fake repository root.
    """
    root = tmp_path / "repo"
    for directory in (
        "src/munasib",
        "scripts",
        "configs",
        "docs",
        "data/processed",
        "data/raw",
        "data/db",
        "models",
        "tests",
        ".git",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    (root / "src/munasib/__init__.py").write_text("", encoding="utf-8")
    (root / "src/munasib/mod.py").write_text("x = 1\n", encoding="utf-8")
    (root / "scripts/train_sentiment.py").write_text("", encoding="utf-8")
    (root / "configs/config.yaml").write_text("project: {}\n", encoding="utf-8")
    (root / "configs/about.yaml").write_text("name: munasib\n", encoding="utf-8")
    (root / "configs/criteria_weights.yaml").write_text("price: 0.2\n", encoding="utf-8")
    (root / "docs/sentiment.md").write_text("# sentiment\n", encoding="utf-8")
    (root / "docs/training_on_colab.md").write_text("# colab\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'munasib'\n", encoding="utf-8")
    (root / "requirements-colab.txt").write_text("transformers==5.19.0\n", encoding="utf-8")
    (root / "tests/test_thing.py").write_text("", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / ".git/config").write_text("[core]\n", encoding="utf-8")
    (root / "models/checkpoint.bin").write_text("weights\n", encoding="utf-8")
    (root / "data/raw/reviews.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "data/db/munasib.sqlite").write_text("db\n", encoding="utf-8")

    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "asin": ["a", "a", "b"],
            "split": ["train", "val", "test"],
            "review_text_clean": ["good", "bad", "okay"],
            "sentiment_label_weak": [2, 0, 1],
        }
    ).to_parquet(root / "data/processed/reviews.parquet", index=False)
    return root


def _build(script, root: Path, output: Path) -> int:
    """Run the exporter CLI against a repository tree.

    Args:
        script: Loaded exporter module.
        root: Fake repository root.
        output: Destination zip.

    Returns:
        The process exit code.
    """
    return script.main(["--repo-root", str(root), "--output", str(output)])


def test_bundle_contains_only_allowlisted_files(script, tmp_path: Path) -> None:
    """Secrets, checkpoints, the database and the raw data never enter the zip."""
    root = _fake_repo(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    assert _build(script, root, archive_path) == 0

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()

    assert "configs/config.yaml" in names
    assert "src/munasib/mod.py" in names
    assert "scripts/train_sentiment.py" in names
    assert "pyproject.toml" in names
    assert "requirements-colab.txt" in names
    assert "data/processed/reviews.parquet" in names

    forbidden = (
        "configs/about.yaml",
        "configs/criteria_weights.yaml",
        "tests/test_thing.py",
        ".env",
        "models/checkpoint.bin",
        "data/raw/reviews.jsonl",
        "data/db/munasib.sqlite",
    )
    for name in forbidden:
        assert name not in names
    assert not any(name.startswith(".git/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert len([name for name in names if name.startswith("configs/")]) == 1


def test_bundle_carries_a_slim_review_table(script, tmp_path: Path) -> None:
    """Only the five columns the trainers read travel with the bundle."""
    root = _fake_repo(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    assert _build(script, root, archive_path) == 0
    with zipfile.ZipFile(archive_path) as archive:
        frame = pd.read_parquet(io.BytesIO(archive.read("data/processed/reviews.parquet")))
    assert list(frame.columns) == list(_REVIEW_COLUMNS)
    assert len(frame) == 3


def test_bundle_reports_its_size_and_members(script, tmp_path: Path, capsys) -> None:
    """The printed report is what the user relies on to know what was shipped."""
    root = _fake_repo(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    assert _build(script, root, archive_path) == 0
    output = capsys.readouterr().out
    assert "compressed:" in output
    assert "files:" in output
    assert "configs/config.yaml" in output
    assert archive_path.is_file()


def test_missing_review_table_fails_with_an_explanation(script, tmp_path: Path) -> None:
    """A bundle without data would fail later, in Colab, far from the cause."""
    root = _fake_repo(tmp_path)
    (root / "data/processed/reviews.parquet").unlink()
    assert _build(script, root, tmp_path / "bundle.zip") == 1


def test_missing_allowlisted_input_fails(script, tmp_path: Path) -> None:
    """A missing config/doc must not silently produce an incomplete bundle."""
    root = _fake_repo(tmp_path)
    (root / "pyproject.toml").unlink()
    assert _build(script, root, tmp_path / "bundle.zip") == 1


def test_forbidden_path_guard_rejects_a_widened_allowlist(script, tmp_path: Path, monkeypatch) -> None:
    """If someone adds a secret to the allowlist, the build fails instead of leaking."""
    root = _fake_repo(tmp_path)
    monkeypatch.setattr(script, "CONFIG_FILES", (".env",))
    with pytest.raises(DataError, match="forbidden"):
        script.collect_files(root)


def test_requirements_are_exactly_pinned_and_exclude_torch() -> None:
    """Every package line pins ``==`` and torch is left to the Colab runtime."""
    lines = [
        line.strip()
        for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "requirements-colab.txt must list the training dependencies"
    pattern = re.compile(r"^[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9_.\-]*$")
    for line in lines:
        assert pattern.match(line), f"not an exact pin: {line!r}"
        assert line.split("==")[0].lower() != "torch"
    packages = {line.split("==")[0].lower() for line in lines}
    assert packages == {"transformers", "tokenizers", "scikit-learn", "pandas", "numpy", "pyarrow"}
