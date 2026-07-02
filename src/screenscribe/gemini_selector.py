"""
Gemini-driven frame selection.

The transcript-based selector (transcript_selector.py) guesses key moments from
the words alone — blind to the pixels. This selector instead hands the YouTube
URL to Gemini, which actually watches the video (frames + audio) and returns the
timestamps where something visually important is on screen. ffmpeg then extracts
those exact frames.

Returns the same shape as transcript_selector — [{"timestamp": float, "reason":
str}, ...] — and reuses _validate_and_filter, so it's a drop-in replacement for
the selection step. Callers fall back to the transcript selector when no
GEMINI_API_KEY is set (see gemini_available()).
"""

import json
import os
import time

from screenscribe.local_source import resolve_local_path
from screenscribe.transcript_selector import (
    _parse_time_range,
    _parse_timestamp,
    _parse_timestamps_list,
    _validate_and_filter,
)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

# Files API: how long to wait for an uploaded local video to finish processing.
UPLOAD_POLL_INTERVAL = 2.0
UPLOAD_TIMEOUT = 300.0

# Uploaded local files, cached by absolute path for this process so retries and
# repeat calls in one run don't re-upload the same (possibly large) video.
_UPLOADED: dict[str, object] = {}


def _file_state(f) -> str:
    """The processing state name of a Files API file ('PROCESSING'|'ACTIVE'|'FAILED')."""
    state = getattr(f, "state", None)
    return getattr(state, "name", state)


def _upload_local_video(client, path):
    """Upload a local video via the Gemini Files API and block until it is ACTIVE.
    Returns the File. Raises RuntimeError if processing fails or times out."""
    f = client.files.upload(file=path)
    waited = 0.0
    while _file_state(f) == "PROCESSING":
        if waited >= UPLOAD_TIMEOUT:
            raise RuntimeError(
                f"Gemini did not finish processing '{path}' within {UPLOAD_TIMEOUT:.0f}s."
            )
        time.sleep(UPLOAD_POLL_INTERVAL)
        waited += UPLOAD_POLL_INTERVAL
        f = client.files.get(name=f.name)
    state = _file_state(f)
    if state != "ACTIVE":
        raise RuntimeError(f"Gemini failed to process '{path}' (state={state}).")
    return f


def gemini_available() -> bool:
    """True if a Gemini key is configured and the SDK is importable."""
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def _frame_schema():
    from google.genai import types
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "timestamp": types.Schema(
                    type=types.Type.STRING,
                    description="Timestamp in the video as MM:SS or H:MM:SS",
                ),
                "reason": types.Schema(
                    type=types.Type.STRING,
                    description="What is on screen at this moment and why it matters",
                ),
            },
            required=["timestamp", "reason"],
        ),
    )


def _build_prompt(purpose: str, max_items: int, focus: str) -> str:
    if purpose == "slides":
        body = (
            f"Watch this video and identify up to {max_items} moments that would make "
            f"strong standalone images — a complete, self-contained visual someone could "
            f"understand on its own.\n\n"
            f"Select a moment ONLY when the visual is COMPLETE and clear — a finished "
            f"diagram, chart, scene, result, or readable on-screen text/code/table (not "
            f"mid-transition, mid-draw, or mid-scroll). Prefer the instant when the visual "
            f"is most complete and readable. Avoid near-duplicates of the same shot. "
            f"Prioritise diversity across the video's major moments and topics.\n"
        )
    else:
        body = (
            f"Watch this video and identify up to {max_items} moments where a screenshot "
            f"best captures something visually important.\n\n"
            f"Choose moments where a key action, object, scene, demonstration, result, or "
            f"on-screen text/diagram/chart is clearly visible — especially right after "
            f"something important appears or is pointed out, or when a visual is complete. "
            f"Give the timestamp where the relevant thing is most fully and clearly "
            f"visible, not merely when it is first mentioned.\n"
        )
    if focus:
        body += (
            f'\nFOCUS: Prioritise moments related to "{focus}" above all else. '
            f"Only return moments genuinely relevant to it.\n"
        )
    body += (
        "\nReturn a JSON array ordered by importance (most critical first). Each item: "
        '{"timestamp": "MM:SS", "reason": "<what is on screen and why it matters>"}.'
    )
    return body


def _call_gemini(youtube_url, model, prompt, parsed_range, media_resolution_low,
                 response_schema=None, response_json_schema=None):
    """Run one Gemini call over a YouTube URL and return the JSON text. Shared by
    frame selection and whole-video analysis (each passes its own response_schema)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    video_metadata = None
    if parsed_range:
        video_metadata = types.VideoMetadata(
            start_offset=f"{int(parsed_range[0])}s",
            end_offset=f"{int(parsed_range[1])}s",
        )

    local_path = resolve_local_path(youtube_url)
    if local_path:
        uploaded = _UPLOADED.get(local_path)
        if uploaded is None:
            uploaded = _upload_local_video(client, local_path)
            _UPLOADED[local_path] = uploaded
        file_data = types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
    else:
        file_data = types.FileData(file_uri=youtube_url)

    part = types.Part(
        file_data=file_data,
        video_metadata=video_metadata,
    )
    media_resolution = (
        types.MediaResolution.MEDIA_RESOLUTION_LOW
        if media_resolution_low
        else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    )
    if response_json_schema is not None:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            media_resolution=media_resolution,
        )
    else:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema or _frame_schema(),
            media_resolution=media_resolution,
        )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=types.Content(parts=[part, types.Part(text=prompt)]),
                config=config,
            )
            return resp.text
        except Exception as e:  # noqa: BLE001 — surface after retries; caller falls back
            last_err = e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))
    raise last_err


def _call_gemini_text(model, prompt, response_json_schema) -> str:
    """Text-only structured Gemini call (no video upload) returning JSON text.
    Used by cross-video synthesis to classify titles and to aggregate per-video
    results. Sibling of _call_gemini; same retry/backoff."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=types.Content(parts=[types.Part(text=prompt)]),
                config=config,
            )
            return resp.text
        except Exception as e:  # noqa: BLE001 — surface after retries
            last_err = e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))
    raise last_err


def parse_gemini_selections(raw_text: str) -> list[dict]:
    """
    Turn Gemini's JSON response into raw [{"timestamp": float, "reason": str}].
    Timestamps may be "MM:SS", "H:MM:SS", or plain seconds; all are normalised to
    float seconds. Malformed items are skipped. Importance order is preserved.
    Separated from the network call so it can be unit-tested without the SDK.
    """
    data = json.loads(raw_text)
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict) or "timestamp" not in item:
            continue
        try:
            ts = _parse_timestamp(str(item["timestamp"]))
        except (ValueError, TypeError):
            continue
        out.append({"timestamp": ts, "reason": str(item.get("reason", ""))})
    return out


def select_with_gemini(
    youtube_url: str,
    model: str,
    purpose: str = "frames",
    max_items: int = 25,
    min_interval: float = 5.0,
    video_duration: float = 0.0,
    focus: str = "",
    time_range: str = "",
    media_resolution_low: bool = True,
) -> list[dict]:
    """
    Ask Gemini to watch the YouTube video and return key timestamps.

    purpose: "frames" (anything visually important) or "slides" (complete,
             standalone visuals). Returns [{"timestamp": float, "reason": str}],
             validated/deduped/capped by _validate_and_filter.
    """
    parsed_range = _parse_time_range(time_range) if time_range else None
    prompt = _build_prompt(purpose, max_items, focus)
    raw_text = _call_gemini(youtube_url, model, prompt, parsed_range, media_resolution_low)
    raw = parse_gemini_selections(raw_text)

    # Without a known duration the range filter in _validate_and_filter would drop
    # everything; derive a generous upper bound from the picks themselves.
    duration = video_duration
    if duration <= 0 and raw:
        duration = max(s["timestamp"] for s in raw) + min_interval

    return _validate_and_filter(raw, duration, max_items, min_interval, time_range=parsed_range)


def _explicit_timestamps(timestamps, video_duration, max_items, min_interval, time_range):
    """Honor user-specified timestamps without any model call (no key needed)."""
    # When the duration is unknown (no transcript), don't bound the upper range —
    # explicit timestamps should still pass.
    upper = video_duration if video_duration else float("inf")
    return _validate_and_filter(
        _parse_timestamps_list(timestamps), upper, max_items, min_interval,
        time_range=_parse_time_range(time_range) if time_range else None,
    )


def select_frames(
    youtube_url,
    *,
    gemini_model,
    max_frames,
    min_interval,
    focus="",
    time_range="",
    timestamps="",
    video_duration=0.0,
    media_resolution_low=True,
) -> list[dict]:
    """
    Pick frames to capture. Explicit timestamps bypass AI selection (no key
    needed); otherwise Gemini watches the video and picks the moments. Selection
    requires a GEMINI_API_KEY — returns [] if it's unavailable.
    """
    if timestamps:
        return _explicit_timestamps(timestamps, video_duration, max_frames, min_interval, time_range)

    if not gemini_available():
        print("  [selection] No GEMINI_API_KEY — frame selection unavailable (transcript-only).")
        return []

    sels = select_with_gemini(
        youtube_url, gemini_model, "frames", max_frames, min_interval,
        video_duration, focus, time_range, media_resolution_low,
    )
    if sels:
        print(f"  [selection] Gemini watched the video → {len(sels)} moments")
    return sels


def select_slides(
    youtube_url,
    *,
    gemini_model,
    max_slides,
    min_interval,
    focus="",
    time_range="",
    timestamps="",
    video_duration=0.0,
    media_resolution_low=True,
) -> list[dict]:
    """Slide variant of select_frames (complete, standalone visuals)."""
    if timestamps:
        return _explicit_timestamps(timestamps, video_duration, max_slides, min_interval, time_range)

    if not gemini_available():
        print("  [selection] No GEMINI_API_KEY — slide selection unavailable (transcript-only).")
        return []

    sels = select_with_gemini(
        youtube_url, gemini_model, "slides", max_slides, min_interval,
        video_duration, focus, time_range, media_resolution_low,
    )
    if sels:
        print(f"  [selection] Gemini watched the video → {len(sels)} slides")
    return sels
