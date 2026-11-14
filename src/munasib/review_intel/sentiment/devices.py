"""Compute-device choices shared by the training, scoring and evaluation CLIs.

The tuple lives here rather than in :mod:`munasib.review_intel.sentiment.train`
so a report-only command can validate ``--device`` without importing torch.
"""

from __future__ import annotations

#: Devices ``--device`` accepts. ``auto`` prefers CUDA when it is available.
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "cuda")

__all__ = ["DEVICE_CHOICES"]
