"""Tests for the ``scripts/build_dataset.py`` CLI wrapper.

The heavy lifting lives in :mod:`munasib.data.preprocessing` (tested separately);
these tests cover argument parsing and the failure exit codes that do not touch
the raw data. Hermetic: no network, no GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_dataset.py"


def _load_module():
    """Import ``scripts/build_dataset.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_build_dataset_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults() -> None:
    """Defaults point at the master config and the loader's chunk size."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.chunk_size == 50_000


def test_parse_args_overrides() -> None:
    """Flags override config path and chunk size."""
    module = _load_module()
    args = module.parse_args(["--config", "other.yaml", "--chunk-size", "10"])
    assert args.config == "other.yaml"
    assert args.chunk_size == 10


def test_main_rejects_invalid_chunk_size() -> None:
    """A non-positive chunk size exits 1 before any file is opened."""
    module = _load_module()
    assert module.main(["--chunk-size", "0"]) == 1
    assert module.main(["--chunk-size", "-5"]) == 1
