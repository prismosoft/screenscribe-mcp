"""
Schema-driven (typed) extraction.

Hand it a JSON Schema (or a preset name) and it asks Gemini to watch the whole
video and return JSON conforming to that shape, validated with jsonschema. The
pure helpers here (resolve/validate/prompt/key) are SDK-free and unit-tested;
extract_structured() does the one network call + retry.
"""

import hashlib
import json
from pathlib import Path

import jsonschema

from screenscribe.config import GEMINI_MEDIA_RESOLUTION_LOW, GEMINI_MODEL
from screenscribe.local_source import source_video_id as _video_id
from screenscribe.session import session_dir
from screenscribe.transcript_selector import _parse_time_range

_SCHEMAS_DIR = Path(__file__).parent / "schemas"


def list_presets() -> list[str]:
    return sorted(p.stem for p in _SCHEMAS_DIR.glob("*.json"))


def load_preset(name: str) -> dict | None:
    path = _SCHEMAS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_schema(arg) -> dict:
    """Resolve a schema arg: dict (inline) | preset name | file path | inline JSON string."""
    if isinstance(arg, dict):
        return arg
    if not isinstance(arg, str):
        raise ValueError(f"schema must be a dict or string, got {type(arg).__name__}")

    preset = load_preset(arg)
    if preset is not None:
        return preset

    path = Path(arg)
    if path.exists():
        return json.loads(path.read_text())

    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError:
        raise ValueError(
            f"Schema '{arg[:40]}' is not a known preset, an existing file, or valid JSON. "
            f"Presets: {', '.join(list_presets())}"
        )
    if not isinstance(parsed, dict):
        raise ValueError("Inline schema JSON must be an object.")
    return parsed


def schema_key(arg) -> str:
    """Cache key: a preset name as-is, else sha256[:12] of the canonical schema JSON."""
    if isinstance(arg, str) and load_preset(arg) is not None:
        return arg
    schema = resolve_schema(arg)
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def validate_output(raw_text, schema) -> tuple[bool, dict | None, str]:
    """(ok, data, error_message). Pure — no SDK."""
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as e:
        return False, None, f"Response was not valid JSON: {e}"
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        return False, None, e.message
    return True, data, ""


def build_extraction_prompt(schema, focus) -> str:
    body = (
        "Watch this entire video and extract the requested information, using what is "
        "shown on screen as well as the narration. Return ONLY JSON that conforms to the "
        "provided schema. Use timestamps in seconds as numbers. Omit fields the video "
        "does not provide, unless the schema marks them required.\n"
        "The video may be in any language. Transcribe the spoken narration and read any "
        "on-screen text regardless of language; transliterate names into the Latin "
        "alphabet when helpful.\n"
    )
    description = schema.get("description")
    if description:
        body += f"\nWhat to extract: {description}\n"
    if focus:
        body += f'\nFOCUS: Pay special attention to "{focus}".\n'
    return body


def _cache_path(video_id: str, key: str) -> Path:
    return session_dir(video_id) / "structured" / f"{key}.json"


def extract_structured(
    url,
    schema_arg,
    *,
    focus="",
    time_range="",
    force=False,
    model=GEMINI_MODEL,
    media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
) -> dict:
    """
    Extract JSON conforming to `schema_arg` (dict | preset name | path | inline JSON)
    from the video at `url`. Returns a result dict:
      {"status": "success", "session_id", "key", "cached", "data"}
      {"status": "invalid", "session_id", "key", "error", "raw"}
      {"status": "error", "error"}
    """
    from screenscribe.gemini_selector import _call_gemini, gemini_available

    if not gemini_available():
        return {"status": "error",
                "error": "extract_structured needs GEMINI_API_KEY (Gemini watches the video)."}

    schema = resolve_schema(schema_arg)          # may raise ValueError (caller handles)
    key = schema_key(schema_arg)
    video_id = _video_id(url)
    cache = _cache_path(video_id, key)

    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        return {"status": "success", "session_id": video_id, "key": key,
                "cached": True, "data": cached["data"]}

    parsed_range = _parse_time_range(time_range) if time_range else None
    prompt = build_extraction_prompt(schema, focus)

    raw = _call_gemini(url, model, prompt, parsed_range, media_resolution_low,
                       response_json_schema=schema)
    ok, data, err = validate_output(raw, schema)
    if not ok:
        retry_prompt = (
            prompt
            + f"\nYour previous output failed validation: {err}\n"
            + "Return corrected JSON that matches the schema exactly."
        )
        raw = _call_gemini(url, model, retry_prompt, parsed_range, media_resolution_low,
                           response_json_schema=schema)
        ok, data, err = validate_output(raw, schema)

    if not ok:
        return {"status": "invalid", "session_id": video_id, "key": key,
                "error": err, "raw": raw}

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"schema": schema, "data": data}, indent=2))
    return {"status": "success", "session_id": video_id, "key": key,
            "cached": False, "data": data}


def extract_structured_batch(
    source,
    schema_arg,
    *,
    max_videos: int | None = None,
    min_duration: int = 0,
    focus="",
    time_range="",
    force=False,
    model=GEMINI_MODEL,
    media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
) -> dict:
    """
    Cross-video fan-out: resolve `source` (single URL / channel / playlist / list of
    URLs) to videos, then run extract_structured over each against `schema_arg`.

    The per-(video, schema) cache makes re-runs free. One video's failure never
    aborts the batch — it lands in `failed` and the rest proceed. This is the
    substrate cross-video synthesis aggregates over.

    Returns:
      {
        "kind", "source", "schema_key",
        "total_videos",                 # videos resolved (after resolver filtering/cap)
        "succeeded": [{video_id, data, cached}],
        "failed":    [{video_id, status, error}],
        "resolver_skipped": {too_short, unavailable},
        "resolver_total_found",         # qualifying videos before any max_videos cap
      }
    """
    from screenscribe.resolver import resolve_videos

    resolved = resolve_videos(source, max_videos=max_videos, min_duration=min_duration)
    key = schema_key(schema_arg)

    succeeded, failed = [], []
    for video_id in resolved["video_ids"]:
        url = f"https://www.youtube.com/watch?v={video_id}"
        result = extract_structured(
            url, schema_arg, focus=focus, time_range=time_range, force=force,
            model=model, media_resolution_low=media_resolution_low,
        )
        if result.get("status") == "success":
            succeeded.append({"video_id": video_id, "data": result["data"], "cached": result["cached"]})
        else:
            failed.append({"video_id": video_id, "status": result.get("status"), "error": result.get("error")})

    return {
        "kind": resolved["kind"],
        "source": source,
        "schema_key": key,
        "total_videos": len(resolved["video_ids"]),
        "succeeded": succeeded,
        "failed": failed,
        "resolver_skipped": resolved["skipped"],
        "resolver_total_found": resolved["total_found"],
    }
