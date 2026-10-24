"""Tests for class weighting, subset selection, prediction order and the loop.

Everything trains the tiny recurrent model on a synthetic three-class sample, so
the file exercises the real ``train_model`` path (checkpoints, early stopping,
resume, reporting) in seconds and without a GPU or the network.

Environment note: pytest enables ``faulthandler``, and this torch build reports a
benign first-chance access violation from some CPU kernels before handling it
internally. It reproduces without any project code
(``python -X faulthandler -c "import torch, torch.nn as nn; nn.LayerNorm(8)(torch.randn(2, 8))"``),
so a dump in the output is not a test failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from munasib.core.config import RnnConfig, SentimentConfig
from munasib.core.constants import SENTIMENT_LABELS, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.data import SentimentDataset, TextEncoder, collate_batch
from munasib.review_intel.sentiment.models import (
    NUM_LABELS,
    Vocabulary,
    build_rnn,
    forward_logits,
    load_model,
)
from munasib.review_intel.sentiment.sampling import (
    assert_split_only,
    load_eval_split,
    text_mask,
)
from munasib.review_intel.sentiment.train import (
    TrainConfig,
    build_report_payload,
    class_weights,
    evaluate_predictions,
    frame_to_dataset,
    predict,
    report_stem,
    resolve_device,
    stratified_subset,
    train_model,
    write_reports,
)

TEXT_BY_LABEL = {
    2: "good great works well",
    0: "bad broke terrible",
    1: "okay average fine",
}


def _frame(per_class: int, *, split: str = SPLIT_TRAIN, prefix: str = "r") -> pd.DataFrame:
    """Build a synthetic balanced review frame.

    Args:
        per_class: Reviews per class.
        split: Value of the ``split`` column.
        prefix: Unique review-id prefix.

    Returns:
        Frame with the columns the trainer reads.
    """
    rows = []
    for label, text in TEXT_BY_LABEL.items():
        for index in range(per_class):
            rows.append(
                {
                    "review_id": f"{prefix}-{label}-{index}",
                    "asin": f"p{(label + index) % 4}",
                    "review_text_clean": text,
                    "sentiment_label_weak": label,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _encoder(frame: pd.DataFrame) -> TextEncoder:
    """Vocabulary encoder fitted on the frame's texts.

    Args:
        frame: Training frame.

    Returns:
        The encoder.
    """
    vocabulary = Vocabulary.from_texts(
        [str(text) for text in frame["review_text_clean"].tolist()], min_freq=1, max_size=500
    )
    return TextEncoder(max_length=10, vocabulary=vocabulary)


def _rnn_config() -> RnnConfig:
    """Tiny recurrent settings for fast tests.

    Returns:
        The validated config.
    """
    return RnnConfig(
        embedding_dim=8,
        hidden_size=8,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        vocab_size=500,
        min_freq=1,
        cell="gru",
    )


def _settings(tmp_path: Path, **overrides: object) -> TrainConfig:
    """Training settings pointed at a temporary run directory.

    Args:
        tmp_path: Pytest temporary directory.
        overrides: Fields to replace.

    Returns:
        The settings.
    """
    values: dict[str, object] = {
        "architecture": "rnn",
        "output_dir": tmp_path / "run",
        "batch_size": 8,
        "learning_rate": 1e-2,
        "epochs": 1,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 0,
        "class_weighting": "none",
        "seed": 7,
        "max_length": 10,
        "device": torch.device("cpu"),
    }
    values.update(overrides)
    return TrainConfig(**values)  # type: ignore[arg-type]


def test_class_weights_are_hand_computable() -> None:
    """``balanced`` is ``n/(k*count)`` and ``sqrt_inverse`` its square root."""
    labels = [0] * 60 + [1] * 30 + [2] * 10
    assert class_weights(labels, "none").tolist() == [1.0, 1.0, 1.0]
    balanced = class_weights(labels, "balanced")
    assert balanced == pytest.approx([100 / (3 * 60), 100 / (3 * 30), 100 / (3 * 10)])
    sqrt_inverse = class_weights(labels, "sqrt_inverse")
    assert sqrt_inverse == pytest.approx(np.sqrt(balanced))


def test_sqrt_inverse_is_geometric_between_ones_and_balanced() -> None:
    """The configured default must not be a disguised ``none`` or ``balanced``.

    Inverse-frequency weights below one are legitimate (the majority class is
    down-weighted), so the check is per class: the square root lies between the
    raw weight and the full balanced weight, and equals neither in general.
    """
    labels = [0] * 50 + [1] * 5 + [2] * 95
    ones = class_weights(labels, "none")
    middle = class_weights(labels, "sqrt_inverse")
    full = class_weights(labels, "balanced")
    assert middle == pytest.approx(np.sqrt(full))
    for value, full_value in zip(middle, full, strict=True):
        assert min(1.0, full_value) <= value <= max(1.0, full_value)
    assert not np.allclose(middle, ones)
    assert not np.allclose(middle, full)


def test_class_weights_reject_an_unknown_scheme() -> None:
    """A typo'd weighting would silently change the loss."""
    with pytest.raises(DataError, match="class_weighting"):
        class_weights([0, 1, 2], "inverse")


def test_class_weights_reject_a_missing_class() -> None:
    """A weight for an absent class cannot be estimated from the data."""
    with pytest.raises(DataError, match="absent"):
        class_weights([2, 2, 0], "balanced")


def test_stratified_subset_is_deterministic_and_keeps_every_class() -> None:
    """Selection uses a reproducible, class-aware draw, never the first N rows."""
    labels = [0] * 10 + [1] * 5 + [2] * 85
    first = stratified_subset(labels, 20, seed=3)
    second = stratified_subset(labels, 20, seed=3)
    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == first.tolist()
    assert len(set(first.tolist())) == 20
    assert len({labels[position] for position in first}) == 3


def test_stratified_subset_covers_a_tiny_class_even_when_rare() -> None:
    """One neutral review must not be crowded out by the positive mass."""
    labels = [0] + [1] + [2] * 98
    positions = stratified_subset(labels, 3, seed=0)
    assert {labels[position] for position in positions} == {0, 1, 2}


def test_stratified_subset_returns_everything_when_asked_for_it() -> None:
    """``None`` and an oversized request both mean the full split."""
    labels = [0, 1, 2, 2]
    assert stratified_subset(labels, None, seed=0).tolist() == [0, 1, 2, 3]
    assert stratified_subset(labels, 99, seed=0).tolist() == [0, 1, 2, 3]


def test_stratified_subset_rejects_a_zero_size() -> None:
    """A zero-row selection would report metrics from nothing."""
    with pytest.raises(ValueError, match="size"):
        stratified_subset([0, 1, 2], 0, seed=0)


def test_frame_to_dataset_rejects_a_missing_column() -> None:
    """A frame without the product column cannot be calibrated afterwards."""
    frame = _frame(2).drop(columns=["asin"])
    with pytest.raises(DataError, match="asin"):
        frame_to_dataset(frame, SentimentConfig(), _encoder(_frame(2)))


def test_predict_returns_rows_in_dataset_order() -> None:
    """Batches are length-sorted, so predictions must be written back by index."""
    frame = pd.DataFrame(
        {
            "review_id": [f"r{index}" for index in range(6)],
            "asin": [f"p{index}" for index in range(6)],
            "review_text_clean": [
                "good",
                "bad broke terrible",
                "okay",
                "good great works well",
                "bad",
                "okay average fine",
            ],
            "sentiment_label_weak": [2, 0, 1, 2, 0, 1],
        }
    )
    encoder = _encoder(frame)
    dataset = frame_to_dataset(frame, SentimentConfig(), encoder)
    model = build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary or {}))
    result = predict(model, dataset, batch_size=2, device=torch.device("cpu"))
    assert list(result.y_true) == dataset.labels
    assert list(result.asins) == dataset.asins
    assert result.proba.shape == (6, NUM_LABELS)
    assert np.allclose(result.proba.sum(axis=1), 1.0, atol=1e-5)


def test_one_optimisation_step_reduces_the_loss() -> None:
    """The gradient path is wired correctly: the loss falls on a fixed batch."""
    frame = _frame(3)
    encoder = _encoder(frame)
    dataset = frame_to_dataset(frame, SentimentConfig(), encoder)
    model = build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary or {}))
    items = [dataset[index] for index in range(len(dataset))]
    batch = collate_batch(items, pad_id=0)
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    labels = batch.labels

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.eval()
    with torch.no_grad():
        before = float(criterion(forward_logits(model, input_ids, attention_mask), labels))
    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(forward_logits(model, input_ids, attention_mask), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = float(criterion(forward_logits(model, input_ids, attention_mask), labels))
    assert after < before * 0.5


def test_train_model_writes_checkpoints_and_a_full_report(tmp_path: Path) -> None:
    """The loop saves best/last, a history and both calibration tables."""
    train_frame = _frame(6)
    val_frame = _frame(3, split=SPLIT_VAL, prefix="v")
    settings = _settings(tmp_path, epochs=2)
    encoder = _encoder(train_frame)
    model = build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary))

    outcome = train_model(
        model=model,
        train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
        selection_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        report_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        settings=settings,
        vocabulary=encoder.vocabulary,
        rnn_config=_rnn_config(),
    )

    assert outcome.epochs_run == 2
    assert [record.epoch for record in outcome.history] == [1, 2]
    assert 1 <= outcome.best_epoch <= 2
    assert len(outcome.validation.predictions.y_true) == len(val_frame)
    assert 0.0 <= outcome.validation.metrics.macro_f1 <= 1.0
    assert len(outcome.class_weights) == NUM_LABELS

    run_dir = settings.output_dir
    for name in ("best", "last"):
        assert (run_dir / name / "model_meta.json").is_file()
        assert (run_dir / name / "model.pt").is_file()
        assert (run_dir / name / "vocab.json").is_file()
    assert (run_dir / "last" / "trainer_state.pt").is_file()
    history = json.loads((run_dir / "training_log.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[-1]["epoch"] == 2

    payload = build_report_payload(
        outcome=outcome,
        settings=settings,
        evaluation=outcome.validation,
        train_rows=len(train_frame),
        selection_rows=len(val_frame),
    )
    assert payload["split"] == SPLIT_VAL
    assert payload["evaluated_rows"] == len(val_frame)
    assert payload["training"]["val_subset_requested"] is None
    assert payload["training"]["full_report_split"] is True
    assert set(payload["calibration"]) == {"hard_predictions", "mean_negative_probability"}

    paths = write_reports(
        outcome=outcome,
        settings=settings,
        evaluation=outcome.validation,
        report_dir=tmp_path / "reports",
        train_rows=len(train_frame),
        selection_rows=len(val_frame),
    )
    assert all(path.is_file() for path in paths)
    assert paths[0].name == "sentiment_rnn_val.json"
    assert "weak supervision" in paths[1].read_text(encoding="utf-8")


def test_report_covers_the_full_split_when_selection_used_a_subset(tmp_path: Path) -> None:
    """Per-epoch selection may be cheap, but the reported rows are the full split."""
    train_frame = _frame(6)
    val_frame = _frame(10, split=SPLIT_VAL, prefix="v")
    settings = _settings(tmp_path, val_subset=6)
    encoder = _encoder(train_frame)
    positions = stratified_subset(val_frame["sentiment_label_weak"].tolist(), 6, seed=7)
    selection_frame = val_frame.iloc[positions].reset_index(drop=True)

    outcome = train_model(
        model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
        train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
        selection_dataset=frame_to_dataset(selection_frame, SentimentConfig(), encoder),
        report_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        settings=settings,
        vocabulary=encoder.vocabulary,
        rnn_config=_rnn_config(),
    )
    payload = build_report_payload(
        outcome=outcome,
        settings=settings,
        evaluation=outcome.validation,
        train_rows=len(train_frame),
        selection_rows=len(selection_frame),
    )
    assert payload["training"]["selection_rows"] == 6
    assert payload["training"]["val_subset_requested"] == 6
    assert payload["evaluated_rows"] == len(val_frame) == 30
    markdown = (tmp_path / "run" / "training_log.json").is_file()
    assert markdown


def test_resume_continues_after_the_last_completed_epoch(tmp_path: Path) -> None:
    """``--resume`` picks up the epoch count instead of restarting the schedule."""
    train_frame = _frame(6)
    val_frame = _frame(3, split=SPLIT_VAL, prefix="v")
    encoder = _encoder(train_frame)
    first = _settings(tmp_path, epochs=1)
    train_model(
        model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
        train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
        selection_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        report_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        settings=first,
        vocabulary=encoder.vocabulary,
        rnn_config=_rnn_config(),
    )

    second = _settings(tmp_path, epochs=2, resume=True)
    outcome = train_model(
        model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
        train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
        selection_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        report_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        settings=second,
        vocabulary=encoder.vocabulary,
        rnn_config=_rnn_config(),
    )
    assert outcome.epochs_run == 2
    assert [record.epoch for record in outcome.history] == [1, 2]
    state = torch.load(
        second.output_dir / "last" / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    assert int(state["epoch"]) == 2


def test_resume_without_a_checkpoint_fails_loudly(tmp_path: Path) -> None:
    """A resume with nothing to resume from must not start a silent fresh run."""
    train_frame = _frame(2)
    encoder = _encoder(train_frame)
    with pytest.raises(DataError, match="resume"):
        train_model(
            model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
            train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
            selection_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
            report_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
            settings=_settings(tmp_path, resume=True),
            vocabulary=encoder.vocabulary,
            rnn_config=_rnn_config(),
        )


def test_train_model_requires_the_architecture_companions(tmp_path: Path) -> None:
    """A model without its tokenizer or vocabulary cannot be trained or saved."""
    frame = _frame(2)
    encoder = _encoder(frame)
    dataset = frame_to_dataset(frame, SentimentConfig(), encoder)
    with pytest.raises(DataError, match="tokenizer"):
        train_model(
            model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
            train_dataset=dataset,
            selection_dataset=dataset,
            report_dataset=dataset,
            settings=_settings(tmp_path, architecture="transformer"),
        )
    with pytest.raises(DataError, match="vocabulary"):
        train_model(
            model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
            train_dataset=dataset,
            selection_dataset=dataset,
            report_dataset=dataset,
            settings=_settings(tmp_path),
        )


def test_leakage_guard_still_rejects_mixed_splits() -> None:
    """Neither the sample builder nor the eval loader accepts a mixed frame."""
    frame = pd.concat(
        [_frame(2, split=SPLIT_TRAIN, prefix="t"), _frame(2, split=SPLIT_TEST, prefix="s")],
        ignore_index=True,
    )
    config = SentimentConfig()
    with pytest.raises(DataError):
        assert_split_only(frame, SPLIT_TRAIN)
    with pytest.raises(DataError):
        load_eval_split(frame, SPLIT_TRAIN, config)


def test_text_mask_is_shared_by_training_and_evaluation() -> None:
    """The same predicate decides what is usable on both paths."""
    frame = _frame(2)
    frame.loc[0, "review_text_clean"] = "  "
    assert text_mask(frame, SentimentConfig()).tolist() == [False, True, True, True, True, True]


def test_evaluate_predictions_reports_both_calibrations() -> None:
    """Hard and soft product calibration are both computed from one prediction set."""
    labels = np.array([0, 0, 2, 2], dtype=np.int64)
    predictions = predict(
        build_rnn(_rnn_config(), vocab_size=len(_encoder(_frame(2)).vocabulary)),
        frame_to_dataset(
            pd.DataFrame(
                {
                    "review_text_clean": ["bad", "bad broke", "good", "good great"],
                    "sentiment_label_weak": labels,
                    "asin": ["a", "a", "b", "b"],
                }
            ),
            SentimentConfig(),
            _encoder(_frame(2)),
        ),
        batch_size=2,
        device=torch.device("cpu"),
    )
    evaluation = evaluate_predictions(predictions, split=SPLIT_VAL)
    assert evaluation.rows == 4
    assert evaluation.calibration_hard.n_products == 2
    assert evaluation.calibration_soft.n_products == 2
    assert evaluation.metrics.n == 4


def test_report_stem_distinguishes_validation_from_the_one_shot_test() -> None:
    """The test report is named ``_final`` to match the flag that produced it."""
    assert report_stem("transformer", SPLIT_VAL) == "sentiment_transformer_val"
    assert report_stem("transformer", SPLIT_TEST) == "sentiment_transformer_final"
    assert report_stem("rnn", SPLIT_TEST) == "sentiment_rnn_final"


def test_resolve_device_rejects_an_impossible_request() -> None:
    """Silently falling back to the CPU would invalidate every timing claim."""
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(DataError, match="device"):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(DataError, match="CUDA is not available"):
            resolve_device("cuda")


def test_best_checkpoint_is_reloadable_after_training(tmp_path: Path) -> None:
    """Training leaves an artifact ``load_model`` can restore, not just memory."""
    train_frame = _frame(4)
    val_frame = _frame(2, split=SPLIT_VAL, prefix="v")
    encoder = _encoder(train_frame)
    settings = _settings(tmp_path)
    train_model(
        model=build_rnn(_rnn_config(), vocab_size=len(encoder.vocabulary)),
        train_dataset=frame_to_dataset(train_frame, SentimentConfig(), encoder),
        selection_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        report_dataset=frame_to_dataset(val_frame, SentimentConfig(), encoder),
        settings=settings,
        vocabulary=encoder.vocabulary,
        rnn_config=_rnn_config(),
    )
    saved = load_model(settings.output_dir / "best")
    assert saved.meta.architecture == "rnn"
    assert saved.vocabulary is not None
    assert saved.meta.max_length == settings.max_length
    assert saved.meta.num_labels == len(SENTIMENT_LABELS)


def test_dataset_can_be_built_from_a_frame_with_extra_columns() -> None:
    """Only the configured text/label/product columns are read."""
    frame = _frame(2).assign(extra=["x"] * 6)
    dataset = frame_to_dataset(frame, SentimentConfig(), _encoder(frame))
    assert isinstance(dataset, SentimentDataset)
    assert len(dataset) == 6
