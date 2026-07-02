"""
Local video file support.

The rest of the pipeline is keyed on a YouTube URL that Gemini fetches itself
(``FileData(file_uri=url)``). A local ``.mp4`` has no URL, so a source can also be
a filesystem path (plain path or ``file://`` URI): we detect it here, and Gemini
uploads the bytes via the Files API instead (see ``gemini_selector._call_gemini``).

These helpers are pure (no SDK, no network) so they stay unit-testable. A local
source has no YouTube transcript — callers proceed with the video alone.
"""

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

# Suffixes we treat as a local video. Restricting by suffix keeps us from
# mistaking some unrelated existing file (e.g. a schema .json path) for a video.
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def resolve_local_path(source) -> str | None:
    """Return the absolute path if `source` is a local video file (a plain path or
    a ``file://`` URI), else None. YouTube URLs and video IDs return None."""
    if not isinstance(source, str) or not source:
        return None
    if source.startswith("file://"):
        source = unquote(urlparse(source).path)
    p = Path(source).expanduser()
    if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES:
        return str(p.resolve())
    return None


def is_local_source(source) -> bool:
    """True if `source` points at a local video file."""
    return resolve_local_path(source) is not None


def local_video_id(path: str) -> str:
    """A stable, human-readable session id for a local file: ``local-<stem>-<hash8>``.
    Deterministic per absolute path, so re-runs hit the same session/cache."""
    abs_path = str(Path(path).expanduser().resolve())
    digest = hashlib.sha1(abs_path.encode()).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(abs_path).stem).strip("-")[:32] or "video"
    return f"local-{stem}-{digest}"


def source_video_id(source: str) -> str:
    """The session id for any source: synthesized for a local file, else the
    canonical YouTube video id (raises ValueError if the URL isn't a single video)."""
    local = resolve_local_path(source)
    if local:
        return local_video_id(local)
    from screenscribe.resolver import parse_video_id
    return parse_video_id(source)
