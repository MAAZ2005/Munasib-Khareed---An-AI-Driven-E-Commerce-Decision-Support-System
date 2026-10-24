"""Tests for the ``scripts/train_sentiment.py`` CLI.

Covers the argument contradictions (``--smoke`` with ``--final``/``--resume``),
the offline smoke run through the real entry point, and the rule that the test
split is only read when ``--final`` is passed.

The smoke tests need ``data/processed/reviews.parquet`` and are skipped without
it; the rest is hermetic. No network, no GPU, and no pretrained weights.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import Settings, load_settings

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_sentiment.py"
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
_REVIEWS = Path(__file__).resolve().parents[2] / "data" / "processed" / "reviews.parquet"

_TEXT_BY_LABEL = {
    2: "good great works well",
    0: "bad broke terrible",
    1: "okay average fine",
}


@pytest.fixture(scope="module")
def script():
    """Import ``scripts/train_sentiment.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_train_sentiment_cli", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_frame(prefix: str, per_class: int = 2) -> pd.DataFrame:
    """Small balanced review frame for the hermetic CLI test.

    Args:
        prefix: Unique review-id prefix.
        per_class: Reviews per class.

    Returns:
        Frame with the columns the trainer reads.
    """
    return pd.DataFrame(
        [
            {
                "review_id": f"{prefix}-{label}-{index}",
                "asin": f"p{index % 2}",
                "review_text_clean": text,
                "sentiment_label_weak": label,
                "split": "train",
            }
            for label, text in _TEXT_BY_LABEL.items()
            for index in range(per_class)
        ]
    )


def _tiny_settings(tmp_path: Path) -> Settings:
    """Project settings with tiny model and reporting paths.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Settings whose sentiment block cannot write into the repository.
    """
    settings = load_settings(_REPO_CONFIG)
    sentiment = settings.review_intel.sentiment.model_copy(
        update={
            "max_length_tokens": 10,
            "class_weighting": "none",
            "seed": 1,
            "report_dir": tmp_path / "reports",
            "rnn": settings.review_intel.sentiment.rnn.model_copy(
                update={
                    "embedding_dim": 8,
                    "hidden_size": 8,
                    "vocab_size": 100,
                    "min_freq": 1,
                    "learning_rate": 1e-2,
                    "output_dir": tmp_path / "run",
                }
            ),
            "transformer": settings.review_intel.sentiment.transformer.model_copy(
                update={"batch_size": 4, "epochs": 1, "output_dir": tmp_path / "run"}
            ),
        }
    )
    return settings.model_copy(
        update={"review_intel": settings.review_intel.model_copy(update={"sentiment": sentiment})}
    )


def test_smoke_and_final_are_rejected_together(script, caplog) -> None:
    """A smoke run must never reach the test split, so the flags cannot combine."""
    with caplog.at_level("ERROR"):
        assert script.main(["--smoke", "--final"]) == 2
    assert "--final cannot be combined with --smoke" in caplog.text


def test_smoke_and_resume_are_rejected_together(script, caplog) -> None:
    """A smoke run always starts in a fresh temp directory, so resume is moot."""
    with caplog.at_level("ERROR"):
        assert script.main(["--smoke", "--resume"]) == 2
    assert "--resume cannot be combined with --smoke" in caplog.text


@pytest.mark.skipif(not _REVIEWS.is_file(), reason="processed reviews are not built")
def test_transformer_smoke_runs_end_to_end(script, tmp_path: Path, monkeypatch) -> None:
    """The offline CPU smoke produces checkpoints and a report in a temp dir."""
    destination = tmp_path / "smoke"
    monkeypatch.setattr(script.tempfile, "mkdtemp", lambda **_: str(destination))
    assert script.main(["--model", "transformer", "--smoke"]) == 0

    assert (destination / "best" / "model_meta.json").is_file()
    assert (destination / "best" / "model.safetensors").is_file()
    assert (destination / "last" / "trainer_state.pt").is_file()
    assert (destination / "training_log.json").is_file()
    payload = json.loads(
        (destination / "reports" / "sentiment_transformer_val.json").read_text(encoding="utf-8")
    )
    assert payload["model"]["architecture"] == "transformer"
    assert payload["model"]["base_model"] == "distilbert-base-uncased"
    assert payload["training"]["full_report_split"] is False
    assert payload["training"]["train_rows"] == script.SMOKE_ROWS
    assert payload["evaluated_rows"] == script.SMOKE_VAL_ROWS
    assert payload["metrics"]["n"] == script.SMOKE_VAL_ROWS


@pytest.mark.skipif(not _REVIEWS.is_file(), reason="processed reviews are not built")
def test_rnn_smoke_runs_end_to_end(script, tmp_path: Path, monkeypatch) -> None:
    """Same check for the recurrent path, whose artifact format differs."""
    destination = tmp_path / "smoke-rnn"
    monkeypatch.setattr(script.tempfile, "mkdtemp", lambda **_: str(destination))
    assert script.main(["--model", "rnn", "--smoke"]) == 0

    assert (destination / "best" / "model.pt").is_file()
    assert (destination / "best" / "vocab.json").is_file()
    payload = json.loads(
        (destination / "reports" / "sentiment_rnn_val.json").read_text(encoding="utf-8")
    )
    assert payload["model"]["architecture"] == "rnn"
    assert payload["model"]["base_model"] is None


def test_test_split_is_only_read_with_final(script, tmp_path: Path, monkeypatch) -> None:
    """Without ``--final`` the loader is asked for ``val`` and nothing else."""
    train_frame = _tiny_frame("t")
    val_frame = _tiny_frame("v")
    test_frame = _tiny_frame("s")
    requested: list[str] = []

    def fake_build_train_sample(frame: pd.DataFrame, config: object) -> object:
        return _sample(train_frame)

    def fake_load_eval_split(frame: pd.DataFrame, split: str, config: object) -> pd.DataFrame:
        requested.append(split)
        return val_frame if split == "val" else test_frame

    monkeypatch.setattr(script, "load_settings", lambda path: _tiny_settings(tmp_path))
    monkeypatch.setattr(script, "read_reviews", lambda settings: pd.DataFrame())
    monkeypatch.setattr(script, "build_train_sample", fake_build_train_sample)
    monkeypatch.setattr(script, "load_eval_split", fake_load_eval_split)

    assert script.main(["--model", "rnn", "--output-dir", str(tmp_path / "run")]) == 0
    assert requested == ["val"]
    assert (tmp_path / "reports" / "sentiment_rnn_val.json").is_file()
    assert not (tmp_path / "reports" / "sentiment_rnn_final.json").exists()

    requested.clear()
    assert (
        script.main(
            ["--model", "rnn", "--output-dir", str(tmp_path / "run-final"), "--final"]
        )
        == 0
    )
    assert requested == ["val", "test"]
    assert (tmp_path / "reports" / "sentiment_rnn_final.json").is_file()


def _sample(frame: pd.DataFrame):
    """Wrap a frame in the shape ``run()`` expects from ``build_train_sample``.

    Args:
        frame: Training rows.

    Returns:
        Object with ``frame`` and a minimal ``report``.
    """
    report = type("Report", (), {"products": 2, "class_counts": {0: 2, 1: 2, 2: 2}})()
    return type("Sample", (), {"frame": frame, "report": report})()
