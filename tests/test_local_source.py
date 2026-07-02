"""Tests for local_source — detecting local video files and synthesizing ids.
Pure helpers, no SDK/network."""

import pytest

from screenscribe.local_source import (
    is_local_source,
    local_video_id,
    resolve_local_path,
    source_video_id,
)


def _make_video(tmp_path, name="clip.mp4"):
    p = tmp_path / name
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # arbitrary bytes; suffix is what matters
    return p


def test_resolve_plain_path_to_existing_mp4(tmp_path):
    p = _make_video(tmp_path)
    assert resolve_local_path(str(p)) == str(p.resolve())


def test_resolve_file_uri(tmp_path):
    p = _make_video(tmp_path, "my clip.mp4")  # space → exercises URL unquoting
    uri = p.resolve().as_uri()
    assert resolve_local_path(uri) == str(p.resolve())


def test_resolve_accepts_common_video_suffixes(tmp_path):
    for name in ("a.mkv", "b.webm", "c.mov", "d.avi", "e.m4v", "f.MP4"):
        assert resolve_local_path(str(_make_video(tmp_path, name))) is not None


def test_youtube_url_is_not_local():
    assert resolve_local_path("https://youtu.be/dQw4w9WgXcQ") is None
    assert resolve_local_path("https://www.youtube.com/watch?v=abc123") is None


def test_nonexistent_path_is_not_local(tmp_path):
    assert resolve_local_path(str(tmp_path / "missing.mp4")) is None


def test_existing_nonvideo_file_is_not_local(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text("{}")
    assert resolve_local_path(str(p)) is None


def test_directory_is_not_local(tmp_path):
    assert resolve_local_path(str(tmp_path)) is None


def test_none_and_empty_are_not_local():
    assert resolve_local_path(None) is None
    assert resolve_local_path("") is None
    assert is_local_source("") is False


def test_local_video_id_is_stable_and_shaped(tmp_path):
    p = _make_video(tmp_path, "Aam Pora Chicken.mp4")
    vid = local_video_id(str(p))
    assert vid == local_video_id(str(p))            # deterministic
    assert vid.startswith("local-")
    assert "Aam-Pora-Chicken" in vid                # sanitized stem
    assert "/" not in vid and " " not in vid        # filesystem-safe id


def test_local_video_ids_differ_by_path(tmp_path):
    a = _make_video(tmp_path, "a.mp4")
    b = _make_video(tmp_path, "b.mp4")
    assert local_video_id(str(a)) != local_video_id(str(b))


def test_source_video_id_routes_local_and_youtube(tmp_path):
    p = _make_video(tmp_path)
    assert source_video_id(str(p)) == local_video_id(str(p))
    assert source_video_id("https://youtu.be/abc123") == "abc123"


def test_source_video_id_raises_on_bad_url():
    with pytest.raises(ValueError):
        source_video_id("not a url or a file")
