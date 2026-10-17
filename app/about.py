"""Typed, validated content for the About page.

The About page is data, not code: its text lives in ``configs/about.yaml`` and is
validated here by a small Pydantic model, so the page can be edited without a code
change and a malformed edit fails loudly at load time instead of rendering blanks.

The loader is pure (no Streamlit, no network) and unit-tested: the "reject a member
with no name" rule lives in the model, not in the page. The page catches
:class:`AboutConfigError` and shows a friendly message, so an invalid file can
never crash the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

#: Repository-relative location of the editable About content.
DEFAULT_ABOUT_PATH = Path(__file__).resolve().parents[1] / "configs" / "about.yaml"

#: Separator used to join the pipeline stages into the one-line overview.
PIPELINE_SEPARATOR = " -> "


class AboutConfigError(Exception):
    """Raised when ``configs/about.yaml`` is missing, unreadable or invalid."""


class AboutMember(BaseModel):
    """One project member shown on the About page.

    Attributes:
        name: Full name; must not be blank.
        registration: University registration number; must not be blank.
        role: Optional role, e.g. ``"Leader"``.
    """

    name: str = Field(min_length=1)
    registration: str = Field(min_length=1)
    role: str | None = None


class AboutConfig(BaseModel):
    """Validated, complete content of the About page.

    Attributes:
        title: Full project title.
        supervisor: Supervisor's name.
        members: Project members (at least one).
        department: Department / campus line.
        programme: Degree, project type and session line.
        description: Short description of what the system does.
        pipeline: Ordered pipeline stage names.
        technologies: Technology list.
        dataset_citation: Dataset citation text.
        notice: Academic-prototype / placeholder notice.
    """

    title: str = Field(min_length=1)
    supervisor: str = Field(min_length=1)
    members: list[AboutMember] = Field(min_length=1)
    department: str = Field(min_length=1)
    programme: str = Field(min_length=1)
    description: str = Field(min_length=1)
    pipeline: list[str] = Field(min_length=1)
    technologies: list[str] = Field(min_length=1)
    dataset_citation: str = Field(min_length=1)
    notice: str = Field(min_length=1)


def format_pipeline(stages: list[str]) -> str:
    """Join the pipeline stages into the one-line overview.

    Args:
        stages: Ordered stage names.

    Returns:
        The stages joined by :data:`PIPELINE_SEPARATOR`.
    """
    return PIPELINE_SEPARATOR.join(stage.strip() for stage in stages if stage.strip())


def load_about_config(path: Path | None = None) -> AboutConfig:
    """Load and validate the About page content from YAML.

    Args:
        path: YAML file to read, or None to use :data:`DEFAULT_ABOUT_PATH`.

    Returns:
        The validated :class:`AboutConfig`.

    Raises:
        AboutConfigError: When the file cannot be read, is not a mapping, or fails
            validation (for example a member with no name).
    """
    target = path if path is not None else DEFAULT_ABOUT_PATH
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AboutConfigError(f"Could not read the About configuration at {target}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise AboutConfigError(f"The About configuration at {target} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise AboutConfigError(
            f"The About configuration at {target} must be a mapping of fields to values."
        )

    try:
        return AboutConfig.model_validate(raw)
    except ValidationError as exc:
        raise AboutConfigError(
            f"The About configuration at {target} is incomplete or invalid:\n{exc}"
        ) from exc


__all__ = [
    "AboutConfig",
    "AboutConfigError",
    "AboutMember",
    "DEFAULT_ABOUT_PATH",
    "PIPELINE_SEPARATOR",
    "format_pipeline",
    "load_about_config",
]
