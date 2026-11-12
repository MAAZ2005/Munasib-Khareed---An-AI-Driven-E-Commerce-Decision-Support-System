"""Tests for the batch scorer CLI (``scripts/predict_reviews.py``).

The tests run the real entry point against the session's tiny transformer
checkpoint and a five-row review table, so chunking, resume, the sidecar and the
row-order guarantee are exercised end to end without a network or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from munasib.core.config import Settings, load_settings
from munasib.review_intel.sentiment.predictor import file_sha256, sidecar_path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "predict_reviews.py"
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

TEXTS: list[str | None] = [
    "good vacuum works great",
    None,
    "bad broke after a day",
    "great price and quiet",
    "the filter broke",
]

IDS = [f"review-{index}" for index in range(len(TEXTS))]


@pytest.fixture(scope="module")
def script():
    """Import ``scripts/predict_reviews.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_predict_reviews", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def settings() -> Settings:
    """Load the repository configuration the script would load."""
    return load_settings(str(_REPO_CONFIG))


def _write_reviews(path: Path, *, texts: list[str | None] | None = None) -> Path:
    """Write a two-column review parquet.

    Args:
        path: Destination file.
        texts: Texts to write (defaults to :data:`TEXTS`).

    Returns:
        The written path.
    """
    frame = pd.DataFrame(
        {
            "review_id": pd.array(IDS, dtype="string"),
            "review_text_clean": pd.array(
                TEXTS if texts is None else texts, dtype="string"
            ),
        }
    )
    frame.to_parquet(path, index=False)
    return path


def _args(
    script,
    checkpoint: Path,
    input_path: Path,
    output: Path,
    *extra: str,
):
    """Build parsed arguments for one scoring run.

    Args:
        script: Loaded script module.
        checkpoint: Tiny checkpoint directory.
        input_path: Review parquet.
        output: Prediction parquet.
        extra: Additional CLI flags.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    argv = [
        "--config",
        str(_REPO_CONFIG),
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--chunk-rows",
        "2",
        *extra,
    ]
    return script.parse_args(argv)


def test_run_writes_every_row_and_the_sidecar(
    script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """Chunks cover the table; the sidecar records what produced the file."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "preds.parquet"
    result = script.run(
        settings, _args(script, tiny_sentiment_checkpoint, reviews, output)
    )

    frame = pd.read_parquet(output)
    assert list(frame.columns) == list(script.PREDICTION_COLUMNS)
    assert len(frame) == len(TEXTS)
    assert frame["review_id"].tolist() == IDS
    assert frame["has_prediction"].tolist() == [True, False, True, True, True]
    assert frame["predicted_label"].isna().tolist() == [False, True, False, False, False]
    assert np.isnan(frame.loc[1, ["prob_negative", "prob_neutral", "prob_positive"]]).all()
    covered = frame.loc[frame["has_prediction"]]
    assert np.allclose(
        covered[["prob_negative", "prob_neutral", "prob_positive"]].sum(axis=1), 1.0, atol=1e-5
    )

    parts = sorted((tmp_path / "preds.parts").glob("part-*.parquet"))
    assert len(parts) == 3
    assert result.scored_rows == len(TEXTS)
    assert result.rows_with_prediction == 4
    assert result.rows_per_second is not None
    assert result.sidecar == sidecar_path(output)

    payload = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert payload["predictions"]["rows"] == len(TEXTS)
    assert payload["predictions"]["rows_without_text"] == 1
    assert payload["predictions"]["sha256"] == file_sha256(output)
    assert payload["input"]["sha256"] == file_sha256(reviews)
    assert payload["checkpoint"]["weights_sha256"] == file_sha256(
        tiny_sentiment_checkpoint / "model.safetensors"
    )
    assert payload["batching"]["chunk_rows"] == 2
    assert payload["batching"]["batch_size"] == 2
    assert len(payload["chunks"]) == 3
    assert payload["chunks"][0] == {"start": 0, "stop": 2, "rows": 2}


def test_limit_scores_only_the_first_rows(
    script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """``--limit`` is applied before chunking and recorded in the sidecar."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "limited.parquet"
    result = script.run(
        settings,
        _args(script, tiny_sentiment_checkpoint,
            reviews,
            output,
            "--limit",
            "3",
        ),
    )
    frame = pd.read_parquet(output)
    assert frame["review_id"].tolist() == IDS[:3]
    assert result.rows == 3
    payload = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert payload["input"]["limit"] == 3
    assert payload["predictions"]["rows"] == 3


def test_resume_reuses_completed_parts(
    script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """Deleting one part and resuming scores only that part's rows."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "resumed.parquet"
    args = _args(script, tiny_sentiment_checkpoint, reviews, output)
    first = script.run(settings, args)
    original = pd.read_parquet(output)

    middle = tmp_path / "resumed.parts" / "part-000000002-000000004.parquet"
    assert middle.is_file()
    middle.unlink()

    resume_args = _args(script, tiny_sentiment_checkpoint, reviews, output, "--resume"
    )
    second = script.run(settings, resume_args)
    assert second.scored_rows == 2
    assert second.rows_with_prediction == first.rows_with_prediction

    resumed = pd.read_parquet(output)
    pd.testing.assert_frame_equal(resumed, original)


def test_resume_refuses_a_changed_input(
    script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """A different review table must not borrow another run's parts."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "changed.parquet"
    script.run(settings, _args(script, tiny_sentiment_checkpoint, reviews, output))

    changed = list(TEXTS)
    changed[0] = "good vacuum works even better"
    _write_reviews(reviews, texts=changed)
    with pytest.raises(script.ResumeError, match="inputs differ"):
        script.run(
            settings,
            _args(script, tiny_sentiment_checkpoint, reviews, output, "--resume"
            ),
        )


def test_resume_without_a_marker_fails(script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path) -> None:
    """``--resume`` on a fresh directory cannot know what to reuse."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "fresh.parquet"
    with pytest.raises(script.ResumeError, match="no"):
        script.run(
            settings,
            _args(script, tiny_sentiment_checkpoint, reviews, output, "--resume"
            ),
        )


def test_main_returns_two_for_an_unmatched_resume(
    script, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """The CLI distinguishes a resume mismatch from a hard failure."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    output = tmp_path / "exit.parquet"
    argv = [
        "--config",
        str(_REPO_CONFIG),
        "--checkpoint",
        str(tiny_sentiment_checkpoint),
        "--input",
        str(reviews),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--chunk-rows",
        "2",
        "--resume",
    ]
    assert script.main(argv) == 2


def test_zero_limit_is_rejected(
    script, settings: Settings, tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """A zero limit would silently write an empty table."""
    reviews = _write_reviews(tmp_path / "reviews.parquet")
    with pytest.raises(ValueError, match="--limit"):
        script.run(
            settings,
            _args(script, tiny_sentiment_checkpoint,
                reviews,
                tmp_path / "zero.parquet",
                "--limit",
                "0",
            ),
        )
