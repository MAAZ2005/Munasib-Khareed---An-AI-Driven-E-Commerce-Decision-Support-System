"""Tests for the prediction ingest (``scripts/ingest_predictions.py``).

The database is an in-memory SQLite cache created from the real models, and the
prediction file is produced by the real batch scorer with the tiny checkpoint, so
the tests exercise the same parquet/sidecar contract the Colab run produces.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import DataError
from munasib.db.models import Product, Review, ReviewFeature
from munasib.db.repositories import ReviewFeatureRepository
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.review_intel.sentiment.predictor import file_sha256, sidecar_path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

ROWS: tuple[tuple[str, str | None], ...] = (
    ("r1", "good vacuum works great"),
    ("r2", "bad broke after a day"),
    ("r3", None),
)


def _load_script(filename: str, name: str) -> types.ModuleType:
    """Import one script module by path.

    Args:
        filename: Script filename under ``scripts/``.
        name: Module name to register.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest() -> types.ModuleType:
    """The ingest script module."""
    return _load_script("ingest_predictions.py", "munasib_ingest_predictions")


@pytest.fixture(scope="module")
def scorer() -> types.ModuleType:
    """The batch-scorer module used to produce a valid prediction file."""
    return _load_script("predict_reviews.py", "munasib_predict_reviews_for_ingest")


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Repository settings (columns, paths, report dir)."""
    return load_settings(str(_REPO_CONFIG))


@pytest.fixture()
def engine() -> Engine:
    """Empty serving cache with one product and three reviews."""
    database = create_engine_from_settings(db_url="sqlite://")
    init_db(database)
    factory = create_session_factory(database)
    with session_scope(factory) as session:
        session.add(
            Product(
                asin="A1",
                split="train",
                brand_missing=False,
                price_missing=False,
                rank_missing=False,
                n_reviews=len(ROWS),
            )
        )
        for review_id, text in ROWS:
            session.add(
                Review(
                    review_id=review_id,
                    asin="A1",
                    split="train",
                    reviewer_id="u1",
                    review_text_clean=text,
                    has_text=text is not None,
                    text_length=len(text or ""),
                    helpful_votes=0,
                    review_time=datetime(2015, 5, 1),
                )
            )
    yield database
    database.dispose()


@pytest.fixture()
def artifacts(
    tmp_path: Path,
    tiny_sentiment_checkpoint: Path,
    scorer: types.ModuleType,
    settings: Settings,
) -> types.SimpleNamespace:
    """Produce a real prediction parquet plus sidecar for the three reviews."""
    reviews = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "review_id": pd.array([review_id for review_id, _ in ROWS], dtype="string"),
            "review_text_clean": pd.array([text for _, text in ROWS], dtype="string"),
        }
    ).to_parquet(reviews, index=False)
    output = tmp_path / "preds.parquet"
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
    ]
    scorer.run(settings, scorer.parse_args(argv))
    return types.SimpleNamespace(
        predictions=output,
        sidecar=sidecar_path(output),
        reviews=reviews,
        frame=pd.read_parquet(output),
    )


def _ingest(ingest, engine: Engine, frame: pd.DataFrame, checkpoint: Path):
    """Run the transactional ingest against the cache.

    Args:
        ingest: Script module.
        engine: Target cache.
        frame: Prediction rows.
        checkpoint: Checkpoint directory recorded as the identity.

    Returns:
        ``(before, after, written)``.
    """
    factory = create_session_factory(engine)
    return ingest.ingest_frame(factory, frame.copy(), checkpoint=str(checkpoint))


def test_sidecar_hashes_and_checkpoint_identity_are_verified(
    ingest, artifacts, tiny_sentiment_checkpoint: Path
) -> None:
    """The happy path: both hashes recorded by the scorer are re-checked."""
    sidecar = ingest.load_sidecar(artifacts.sidecar)
    identity = ingest.verify_sidecar(sidecar, artifacts.predictions, tiny_sentiment_checkpoint)
    assert identity.weights_sha256 == file_sha256(
        tiny_sentiment_checkpoint / "model.safetensors"
    )
    assert sidecar["predictions"]["rows"] == len(ROWS)


def test_modified_parquet_is_refused(ingest, artifacts, tmp_path: Path) -> None:
    """A parquet replaced after scoring no longer matches its sidecar."""
    sidecar = ingest.load_sidecar(artifacts.sidecar)
    modified = tmp_path / "tampered.parquet"
    shutil.copyfile(artifacts.predictions, modified)
    with modified.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(DataError, match="hash does not match"):
        ingest.verify_sidecar(sidecar, modified, None)


def test_a_different_checkpoint_is_refused(
    ingest, artifacts, tiny_sentiment_checkpoint: Path, tmp_path: Path
) -> None:
    """Predictions must be traceable to the exact weights that produced them."""
    copied = tmp_path / "best"
    shutil.copytree(tiny_sentiment_checkpoint, copied)
    meta = copied / "model_meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    sidecar = ingest.load_sidecar(artifacts.sidecar)
    with pytest.raises(DataError, match="does not match the sidecar"):
        ingest.verify_sidecar(sidecar, artifacts.predictions, copied)


def test_missing_checkpoint_is_refused(ingest, artifacts, tmp_path: Path) -> None:
    """Without the checkpoint the identity cannot be verified, so ingest stops."""
    sidecar = ingest.load_sidecar(artifacts.sidecar)
    with pytest.raises(DataError, match="checkpoint directory not found"):
        ingest.verify_sidecar(sidecar, artifacts.predictions, tmp_path / "absent")


def test_missing_sidecar_is_refused(ingest, tmp_path: Path) -> None:
    """Silently trusting an unlabelled prediction file is not an option."""
    with pytest.raises(DataError, match="sidecar not found"):
        ingest.load_sidecar(tmp_path / "nope.meta.json")


def test_ingest_is_idempotent(ingest, engine: Engine, artifacts, tiny_sentiment_checkpoint: Path) -> None:
    """Re-running upserts the same rows instead of duplicating them."""
    first = _ingest(ingest, engine, artifacts.frame, tiny_sentiment_checkpoint)
    assert first == (0, len(ROWS), len(ROWS))

    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        repository = ReviewFeatureRepository(session)
        stored = repository.get_feature("r1")
        assert stored is not None
        assert stored.predicted_label == int(artifacts.frame.loc[0, "predicted_label"])
        assert stored.prob_negative == pytest.approx(
            float(artifacts.frame.loc[0, "prob_negative"])
        )
        assert repository.get_feature("r3").predicted_label is None
        assert repository.count_predictions() == 2
        assert repository.count_text_reviews_without_predictions() == 0
        assert repository.count_predictions_on_textless_reviews() == 0
        assert repository.count_probability_violations() == 0
        before_values = (
            stored.predicted_label,
            stored.prob_negative,
            stored.prob_neutral,
            stored.prob_positive,
        )

    second = _ingest(ingest, engine, artifacts.frame, tiny_sentiment_checkpoint)
    assert second == (len(ROWS), len(ROWS), len(ROWS))
    with session_scope(factory) as session:
        repository = ReviewFeatureRepository(session)
        assert repository.count_features() == len(ROWS)
        stored = repository.get_feature("r1")
        assert (
            stored.predicted_label,
            stored.prob_negative,
            stored.prob_neutral,
            stored.prob_positive,
        ) == before_values


def test_ingest_preserves_columns_owned_by_other_stages(
    ingest, engine: Engine, artifacts, tiny_sentiment_checkpoint: Path
) -> None:
    """The upsert writes prediction columns only; aspects survive a re-ingest."""
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.add(ReviewFeature(review_id="r1", aspects_json=[{"aspect": "noise"}]))
    _ingest(ingest, engine, artifacts.frame, tiny_sentiment_checkpoint)
    with session_scope(factory) as session:
        stored = ReviewFeatureRepository(session).get_feature("r1")
        assert stored.aspects_json == [{"aspect": "noise"}]
        assert stored.predicted_label is not None


def test_unknown_review_ids_roll_back_the_whole_ingest(
    ingest, engine: Engine, artifacts, tiny_sentiment_checkpoint: Path
) -> None:
    """A foreign key that does not exist must not leave a partial table."""
    frame = artifacts.frame.copy()
    frame.loc[0, "review_id"] = "not-in-reviews"
    with pytest.raises(DataError, match="do not exist in reviews"):
        _ingest(ingest, engine, frame, tiny_sentiment_checkpoint)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        assert ReviewFeatureRepository(session).count_features() == 0


def test_a_review_with_text_must_have_a_prediction(
    ingest, engine: Engine, artifacts, tiny_sentiment_checkpoint: Path
) -> None:
    """Dropping a covered review would silently bias the product aggregates."""
    frame = artifacts.frame.drop(index=0).reset_index(drop=True)
    with pytest.raises(DataError, match="no prediction"):
        _ingest(ingest, engine, frame, tiny_sentiment_checkpoint)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        assert ReviewFeatureRepository(session).count_features() == 0


def test_a_textless_review_must_not_get_a_prediction(
    ingest, engine: Engine, artifacts, tiny_sentiment_checkpoint: Path
) -> None:
    """A fabricated score on an empty review is rejected by the coverage check."""
    frame = artifacts.frame.copy()
    empty = frame.index[~frame["has_prediction"].astype(bool)][0]
    frame.loc[empty, ["prob_negative", "prob_neutral", "prob_positive"]] = [0.1, 0.2, 0.7]
    frame.loc[empty, "predicted_label"] = 2
    frame.loc[empty, "has_prediction"] = True
    with pytest.raises(DataError, match="without text"):
        _ingest(ingest, engine, frame, tiny_sentiment_checkpoint)


def test_probability_rows_must_sum_to_one(ingest, artifacts) -> None:
    """A malformed distribution is a broken file, not a rounding question."""
    frame = artifacts.frame.copy()
    frame.loc[0, "prob_negative"] = float(frame.loc[0, "prob_negative"]) + 0.5
    with pytest.raises(DataError, match="sum to 1"):
        ingest.validate_frame(frame)


def test_label_must_be_the_argmax(ingest, artifacts) -> None:
    """`predicted_label` and the probabilities must tell the same story."""
    frame = artifacts.frame.copy()
    frame.loc[0, "predicted_label"] = (int(frame.loc[0, "predicted_label"]) + 1) % 3
    with pytest.raises(DataError, match="argmax"):
        ingest.validate_frame(frame)


def test_column_schema_is_enforced(ingest, artifacts) -> None:
    """Reordered columns mean a different writer; the ingest refuses."""
    frame = artifacts.frame.loc[:, list(reversed(ingest.PREDICTION_COLUMNS))]
    with pytest.raises(DataError, match="columns must be"):
        ingest.validate_frame(frame)


def test_nan_probabilities_on_covered_rows_are_rejected(ingest, artifacts) -> None:
    """A covered row without a probability cannot be aggregated."""
    frame = artifacts.frame.copy()
    frame.loc[0, "prob_neutral"] = float("nan")
    with pytest.raises(DataError, match="missing probability"):
        ingest.validate_frame(frame)


def test_sidecar_json_is_readable(artifacts) -> None:
    """The sidecar is a JSON object with the blocks the ingest expects."""
    payload = json.loads(artifacts.sidecar.read_text(encoding="utf-8"))
    assert set(payload) >= {"predictions", "checkpoint", "input"}
    assert payload["checkpoint"]["weights_file"] == "model.safetensors"
