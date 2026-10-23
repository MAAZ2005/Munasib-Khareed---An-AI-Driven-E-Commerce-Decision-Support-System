"""Unit tests for the About-page configuration (``app.about``).

The About content is data validated by a Pydantic model, so the rules - every field
required, at least one member, a member name that may not be blank - are tested
here directly, without Streamlit. The real committed file is also checked so a bad
edit is caught by the test suite before it reaches the dashboard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app.about import (
    DEFAULT_ABOUT_PATH,
    AboutConfigError,
    format_pipeline,
    load_about_config,
)


def _base_payload() -> dict:
    """Return the committed About YAML as a mutable mapping.

    Returns:
        The parsed ``configs/about.yaml`` as a dictionary.
    """
    return yaml.safe_load(DEFAULT_ABOUT_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    """Write a payload to a temporary About YAML.

    Args:
        tmp_path: Temporary directory from the test fixture.
        payload: The mapping to serialise.

    Returns:
        The path of the written file.
    """
    target = tmp_path / "about.yaml"
    target.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return target


def test_real_config_loads_the_team_and_technologies() -> None:
    """The committed About file validates and carries the required content."""
    config = load_about_config()

    assert config.supervisor == "Mr. Ubaid Aftab Chawla"
    names = [member.name for member in config.members]
    assert names == [
        "Muhammad Maaz Nizami",
        "Muhammad Shehryar Choudhary",
        "Abuzar Ali",
    ]
    registrations = {member.registration for member in config.members}
    assert registrations == {"23K-2052", "23K-0503", "23K-0819"}
    assert [member.role for member in config.members if member.role] == ["Leader"]
    assert config.technologies[:3] == ["Python", "PyTorch", "spaCy"]
    assert "FAST National University" in config.department


def test_pipeline_renders_on_one_line() -> None:
    """The pipeline stages join into a single line in order."""
    config = load_about_config()

    line = format_pipeline(config.pipeline)

    assert line == (
        "Data -> Intelligence -> Evidence -> Risk -> Decision -> Explanation "
        "-> Alternatives -> Consumer Verdict"
    )
    assert "\n" not in line


def test_real_config_publishes_no_contact_details() -> None:
    """The About file contains no email address and no long digit run."""
    text = DEFAULT_ABOUT_PATH.read_text(encoding="utf-8")
    assert "@" not in text
    assert re.search(r"\d{8,}", text) is None


def test_blank_member_name_is_rejected(tmp_path: Path) -> None:
    """A member with an empty name fails validation instead of rendering blank."""
    payload = _base_payload()
    payload["members"][0]["name"] = ""

    with pytest.raises(AboutConfigError):
        load_about_config(_write(tmp_path, payload))


def test_empty_member_list_is_rejected(tmp_path: Path) -> None:
    """A configuration with no members is rejected."""
    payload = _base_payload()
    payload["members"] = []

    with pytest.raises(AboutConfigError):
        load_about_config(_write(tmp_path, payload))


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    """A configuration missing a required field is rejected."""
    payload = _base_payload()
    del payload["supervisor"]

    with pytest.raises(AboutConfigError):
        load_about_config(_write(tmp_path, payload))


def test_missing_file_raises_a_config_error(tmp_path: Path) -> None:
    """A missing file raises the friendly configuration error, not OSError."""
    with pytest.raises(AboutConfigError):
        load_about_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    """A YAML document that is not a mapping is rejected."""
    target = tmp_path / "about.yaml"
    target.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(AboutConfigError):
        load_about_config(target)


def test_blank_registration_is_rejected(tmp_path: Path) -> None:
    """A member with an empty registration number fails validation."""
    payload = _base_payload()
    payload["members"][0]["registration"] = ""

    with pytest.raises(AboutConfigError):
        load_about_config(_write(tmp_path, payload))


def test_member_missing_name_field_is_rejected(tmp_path: Path) -> None:
    """A member missing the name attribute entirely fails validation."""
    payload = _base_payload()
    del payload["members"][0]["name"]

    with pytest.raises(AboutConfigError):
        load_about_config(_write(tmp_path, payload))


def test_malformed_yaml_syntax_raises_config_error(tmp_path: Path) -> None:
    """Syntax errors in the YAML file raise an AboutConfigError."""
    target = tmp_path / "about.yaml"
    target.write_text("title: [unclosed list", encoding="utf-8")

    with pytest.raises(AboutConfigError):
        load_about_config(target)
