"""Tests for ``scripts/download_images.py`` (thumbnail cache).

Hermetic: every HTTP call goes through ``httpx.MockTransport`` and every image is
generated in memory with Pillow, so no test touches the network or the real
``data/`` directories.

Covers the pure pieces (URL selection, fetch + retry policy, resize/metadata
stripping, resume and ``--force``, the JSON report) and one end-to-end
``main()`` run with the HTTP client injected via ``build_client``.
"""

from __future__ import annotations

import importlib.util
import io
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image
from sqlalchemy import select

from munasib.db.models import Product
from munasib.db.session import create_engine_from_settings, create_session_factory, init_db

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_images.py"

ASIN_CACHED = "B00000000A"
ASIN_MISSING_URL = "B00000000B"
ASIN_BROKEN = "B00000000C"


def _load_module():
    """Import ``scripts/download_images.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_download_images_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the retry pause so the retry tests stay fast."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def _jpeg_bytes(size: tuple[int, int] = (600, 400)) -> bytes:
    """Build a real JPEG in memory.

    Args:
        size: Image size in pixels.

    Returns:
        Encoded JPEG bytes.
    """
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _png_with_alpha(size: tuple[int, int] = (64, 64)) -> bytes:
    """Build a PNG with a large fully transparent top-left block.

    Args:
        size: Image size in pixels.

    Returns:
        Encoded PNG bytes.
    """
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    transparent = Image.new("RGBA", (size[0] // 2, size[1] // 2), (0, 0, 0, 0))
    image.paste(transparent, (0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Wrap a mock handler in a client.

    Args:
        handler: Mock transport handler.

    Returns:
        A client that never touches the network.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------------------
# Argument parsing and the "no accidental full run" guard
# ------------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    """Defaults: master config, no selection, config-derived delay/size."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.asins is None and args.asin_file is None
    assert args.limit is None and args.all is False
    assert args.delay is None and args.size is None
    assert args.force is False


def test_parse_args_overrides() -> None:
    """Flags override every default."""
    module = _load_module()
    args = module.parse_args(
        ["--asins", ASIN_CACHED, ASIN_MISSING_URL, "--limit", "3", "--delay", "0",
         "--size", "64", "--force"]
    )
    assert args.asins == [ASIN_CACHED, ASIN_MISSING_URL]
    assert args.limit == 3 and args.delay == 0.0 and args.size == 64 and args.force is True


def test_asins_and_asin_file_are_mutually_exclusive() -> None:
    """Passing both selection flags is a usage error, not a silent choice."""
    module = _load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--asins", ASIN_CACHED, "--asin-file", "x.txt"])


def test_main_refuses_a_full_run_without_all(tmp_path: Path, monkeypatch) -> None:
    """A bare run must not download the whole subset; --all or --limit is required."""
    module = _load_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  db_path: "{(tmp_path / "cache.sqlite").as_posix()}"\n'
        f'  reports_dir: "{(tmp_path / "reports").as_posix()}"\n'
        f'images:\n  directory: "{(tmp_path / "images").as_posix()}"\n',
        encoding="utf-8",
    )
    assert module.main(["--config", str(config)]) == 1


def test_read_asin_file_and_deduplication(tmp_path: Path) -> None:
    """An ASIN file is trimmed, blank lines skipped and the selection de-duplicated."""
    module = _load_module()
    asin_file = tmp_path / "asins.txt"
    asin_file.write_text(f"{ASIN_CACHED}\n\n  {ASIN_MISSING_URL}  \n{ASIN_CACHED}\n", encoding="utf-8")
    args = module.parse_args(["--asin-file", str(asin_file)])
    assert module._requested_asins(args) == [ASIN_CACHED, ASIN_MISSING_URL]


def test_read_asin_file_missing(tmp_path: Path) -> None:
    """A missing ASIN file is a data error naming the path."""
    module = _load_module()
    with pytest.raises(Exception, match="not found"):
        module._read_asin_file(tmp_path / "nope.txt")


# ------------------------------------------------------------------------------
# Fetch + retry policy
# ------------------------------------------------------------------------------


def test_fetch_image_bytes_returns_the_body() -> None:
    """A 200 response body is returned unchanged."""
    module = _load_module()
    payload = _jpeg_bytes()
    client = _client(lambda request: httpx.Response(200, content=payload))
    assert module.fetch_image_bytes(client, "https://example.com/a.jpg") == payload


@pytest.mark.parametrize(
    ("status", "reason"),
    [(404, "not_found"), (403, "forbidden"), (410, "gone"), (418, "http_418")],
)
def test_fetch_image_bytes_permanent_failures_are_not_retried(
    status: int, reason: str
) -> None:
    """A permanent status maps to a reason and is attempted exactly once."""
    module = _load_module()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status)

    with pytest.raises(module.ImageDownloadError) as excinfo:
        module.fetch_image_bytes(_client(handler), "https://example.com/a.jpg")
    assert excinfo.value.reason == reason
    assert len(calls) == 1


def test_fetch_image_bytes_retries_a_server_error_once() -> None:
    """A 503 is retried once and the second success is used."""
    module = _load_module()
    payload = _jpeg_bytes()
    responses = [httpx.Response(503), httpx.Response(200, content=payload)]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    assert module.fetch_image_bytes(_client(handler), "https://example.com/a.jpg") == payload
    assert responses == []


def test_fetch_image_bytes_timeout_is_reported() -> None:
    """A timeout is retried once and then reported as ``timeout``."""
    module = _load_module()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(module.ImageDownloadError) as excinfo:
        module.fetch_image_bytes(_client(handler), "https://example.com/a.jpg")
    assert excinfo.value.reason == "timeout"
    assert len(calls) == 2


# ------------------------------------------------------------------------------
# Thumbnailing
# ------------------------------------------------------------------------------


def test_make_thumbnail_resizes_and_strips_metadata() -> None:
    """The longest side is capped, the aspect ratio kept and no EXIF written."""
    module = _load_module()
    encoded = module.make_thumbnail(_jpeg_bytes((600, 400)), 320)
    with Image.open(io.BytesIO(encoded)) as result:
        assert result.format == "JPEG"
        assert max(result.size) == 320
        assert result.size[0] / result.size[1] == pytest.approx(1.5, abs=0.01)
        assert not result.getexif()
        assert "icc_profile" not in result.info


def test_make_thumbnail_flattens_transparency() -> None:
    """A transparent PNG becomes an opaque RGB JPEG (white background)."""
    module = _load_module()
    encoded = module.make_thumbnail(_png_with_alpha((64, 64)), 320)
    with Image.open(io.BytesIO(encoded)) as result:
        assert result.format == "JPEG"
        assert result.mode == "RGB"
        # Inside the fully transparent block (far from the JPEG block edge) -> white.
        assert result.getpixel((16, 16)) == (255, 255, 255)
        # Inside the opaque block the red survives.
        assert result.getpixel((48, 48)) == pytest.approx((255, 0, 0), abs=6)


def test_make_thumbnail_rejects_a_non_image() -> None:
    """An HTML error page or truncated file never lands in the cache."""
    module = _load_module()
    with pytest.raises(module.ImageDownloadError) as excinfo:
        module.make_thumbnail(b"<html>not an image</html>", 320)
    assert excinfo.value.reason == "not_an_image"


# ------------------------------------------------------------------------------
# Store one image / resume
# ------------------------------------------------------------------------------


def test_download_one_writes_atomically_and_skips_next_time(tmp_path: Path) -> None:
    """The first call writes the file; the second skips it without a request."""
    module = _load_module()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=_jpeg_bytes())

    destination = tmp_path / "images" / f"{ASIN_CACHED}.jpg"
    client = _client(handler)
    first = module.download_one(
        client, ASIN_CACHED, "https://example.com/a.jpg", destination, max_side=320, force=False
    )
    assert first == module.OK
    assert destination.exists() and not destination.with_name(destination.name + ".tmp").exists()
    second = module.download_one(
        client, ASIN_CACHED, "https://example.com/a.jpg", destination, max_side=320, force=False
    )
    assert second == module.SKIPPED
    assert len(calls) == 1  # the skip never hits the network


def test_download_one_force_overwrites(tmp_path: Path) -> None:
    """--force re-downloads even when the file already exists."""
    module = _load_module()
    destination = tmp_path / "images" / f"{ASIN_CACHED}.jpg"
    destination.parent.mkdir()
    destination.write_bytes(b"stale")
    client = _client(lambda request: httpx.Response(200, content=_jpeg_bytes()))
    outcome = module.download_one(
        client, ASIN_CACHED, "https://example.com/a.jpg", destination, max_side=320, force=True
    )
    assert outcome == module.OK
    assert destination.read_bytes().startswith(b"\xff\xd8")


@pytest.mark.parametrize(
    ("asin", "image_url", "reason"),
    [
        (ASIN_MISSING_URL, None, "no_image_url"),
        ("../escape", "https://example.com/a.jpg", "invalid_asin"),
        ("", "https://example.com/a.jpg", "invalid_asin"),
    ],
)
def test_download_one_reports_bad_input(asin: str, image_url: str | None, reason: str, tmp_path: Path) -> None:
    """A missing URL or a malformed ASIN is a recorded reason, never a crash."""
    module = _load_module()
    client = _client(lambda request: httpx.Response(200, content=_jpeg_bytes()))
    with pytest.raises(module.ImageDownloadError) as excinfo:
        module.download_one(
            client,
            asin,
            image_url,
            tmp_path / "images" / "x.jpg",
            max_side=320,
            force=False,
        )
    assert excinfo.value.reason == reason


# ------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------


def test_build_report_counts_by_reason() -> None:
    """The report counts outcomes and groups failures by reason."""
    module = _load_module()
    report = module.build_report(
        {
            ASIN_CACHED: module.OK,
            ASIN_MISSING_URL: module.SKIPPED,
            ASIN_BROKEN: "not_found",
            "B00000000D": "not_found",
            "B00000000E": "timeout",
        }
    )
    assert report == {
        "attempted": 5,
        "ok": 1,
        "skipped": 1,
        "failed": {"not_found": 2, "timeout": 1},
    }


def test_write_report_writes_json(tmp_path: Path) -> None:
    """The report file is valid JSON with the computed counts."""
    module = _load_module()
    report = {"attempted": 2, "ok": 2, "skipped": 0, "failed": {}}
    path = module.write_report(tmp_path / "reports", report)
    assert path.name == module.REPORT_FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == report


# ------------------------------------------------------------------------------
# Selection against a temporary cache + one end-to-end run
# ------------------------------------------------------------------------------


def _cache(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a temporary serving cache with the given product rows.

    Args:
        tmp_path: Pytest temporary directory.
        rows: Product insert payloads (only the needed columns).

    Returns:
        The database path.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    if rows:
        with create_session_factory(engine)() as session:
            session.execute(Product.__table__.insert(), rows)
            session.commit()
    engine.dispose()
    return db_path


def _product_row(asin: str, image_url: str | None) -> dict:
    """Build a minimal products insert payload.

    Args:
        asin: Product ASIN.
        image_url: Stored image URL or None.

    Returns:
        Insert payload with the NOT NULL columns filled.
    """
    return {
        "asin": asin,
        "split": "train",
        "brand_missing": False,
        "price_missing": False,
        "rank_missing": True,
        "n_reviews": 1,
        "image_url": image_url,
        "image_missing": image_url is None,
        "image_cached": False,
    }


def _config(tmp_path: Path, db_path: Path, images_dir: Path) -> Path:
    """Write a minimal config pointing at the temp cache and image directory.

    Args:
        tmp_path: Pytest temporary directory.
        db_path: SQLite cache to use.
        images_dir: Thumbnail cache directory.

    Returns:
        The config path.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  db_path: "{db_path.as_posix()}"\n'
        f'  reports_dir: "{(tmp_path / "reports").as_posix()}"\n'
        f'images:\n  directory: "{images_dir.as_posix()}"\n  download_delay_seconds: 0.0\n',
        encoding="utf-8",
    )
    return config


def test_select_targets_default_and_limit(tmp_path: Path) -> None:
    """Without a selection only products with an image URL are candidates."""
    module = _load_module()
    db_path = _cache(
        tmp_path,
        [
            _product_row(ASIN_CACHED, "https://example.com/a.jpg"),
            _product_row(ASIN_MISSING_URL, None),
            _product_row(ASIN_BROKEN, "https://example.com/c.jpg"),
        ],
    )
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    args = module.parse_args(["--all", "--limit", "1"])
    targets, unknown = module.select_targets(engine, args)
    engine.dispose()
    assert [asin for asin, _ in targets] == [ASIN_CACHED]  # ASIN order, then limited
    assert unknown == []


def test_select_targets_explicit_and_unknown(tmp_path: Path) -> None:
    """Requested ASINs keep their order; unknown ones are reported separately."""
    module = _load_module()
    db_path = _cache(
        tmp_path,
        [
            _product_row(ASIN_CACHED, "https://example.com/a.jpg"),
            _product_row(ASIN_BROKEN, "https://example.com/c.jpg"),
            _product_row(ASIN_MISSING_URL, None),
        ],
    )
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    args = module.parse_args(
        ["--asins", ASIN_BROKEN, "NOTANASIN1", ASIN_MISSING_URL]
    )
    targets, unknown = module.select_targets(engine, args)
    engine.dispose()
    assert [asin for asin, _ in targets] == [ASIN_BROKEN, ASIN_MISSING_URL]
    assert unknown == ["NOTANASIN1"]


def test_select_targets_empty_cache_raises(tmp_path: Path) -> None:
    """An empty cache is a data error naming the build commands."""
    module = _load_module()
    db_path = _cache(tmp_path, [])
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    args = module.parse_args(["--all"])
    with pytest.raises(Exception, match="no products selected"):
        module.select_targets(engine, args)
    engine.dispose()


def test_main_downloads_marks_cached_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One full run: file written, ``image_cached`` set, counts reported."""
    module = _load_module()
    db_path = _cache(
        tmp_path,
        [
            _product_row(ASIN_CACHED, "https://example.com/a.jpg"),
            _product_row(ASIN_BROKEN, "https://example.com/gone.jpg"),
            _product_row(ASIN_MISSING_URL, None),
        ],
    )
    images_dir = tmp_path / "images"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("gone.jpg"):
            return httpx.Response(404)
        return httpx.Response(200, content=_jpeg_bytes())

    monkeypatch.setattr(module, "build_client", lambda settings: _client(handler))
    config = _config(tmp_path, db_path, images_dir)
    assert module.main(["--config", str(config), "--all"]) == 0

    assert (images_dir / f"{ASIN_CACHED}.jpg").exists()
    assert not (images_dir / f"{ASIN_BROKEN}.jpg").exists()
    report = json.loads((tmp_path / "reports" / module.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report == {
        "attempted": 2,  # only products with an image_url are candidates
        "ok": 1,
        "skipped": 0,
        "failed": {"not_found": 1},
    }
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    try:
        with engine.connect() as connection:
            cached = dict(
                connection.execute(select(Product.asin, Product.image_cached)).all()
            )
    finally:
        engine.dispose()
    assert cached == {ASIN_CACHED: True, ASIN_BROKEN: False, ASIN_MISSING_URL: False}


def test_main_resumes_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second run skips the existing thumbnail and reports it as skipped."""
    module = _load_module()
    db_path = _cache(tmp_path, [_product_row(ASIN_CACHED, "https://example.com/a.jpg")])
    images_dir = tmp_path / "images"
    monkeypatch.setattr(
        module, "build_client", lambda settings: _client(lambda request: httpx.Response(200, content=_jpeg_bytes()))
    )
    config = _config(tmp_path, db_path, images_dir)
    assert module.main(["--config", str(config), "--all"]) == 0
    assert module.main(["--config", str(config), "--all"]) == 0
    report = json.loads((tmp_path / "reports" / module.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["ok"] == 0 and report["skipped"] == 1


def test_main_reports_not_in_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested ASIN absent from the cache is a reported failure, not a crash."""
    module = _load_module()
    db_path = _cache(tmp_path, [_product_row(ASIN_CACHED, "https://example.com/a.jpg")])
    monkeypatch.setattr(
        module, "build_client", lambda settings: _client(lambda request: httpx.Response(200, content=_jpeg_bytes()))
    )
    config = _config(tmp_path, db_path, tmp_path / "images")
    assert module.main(["--config", str(config), "--asins", "NOTANASIN1"]) == 0
    report = json.loads((tmp_path / "reports" / module.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report == {"attempted": 1, "ok": 0, "skipped": 0, "failed": {"not_in_cache": 1}}
