"""Minimal Gemini-powered video analysis MCP server.

The MCP deliberately exposes one analysis primitive. The connected agent handles
reasoning, orchestration, and follow-up. No Anthropic/OpenAI model dependency is
embedded in this service.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import jsonschema
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_LEVEL = "medium"
DEFAULT_MEDIA_RESOLUTION = "low"
_ALLOWED_THINKING = {"low", "medium", "high"}
_ALLOWED_MEDIA_RESOLUTION = {"low", "medium", "high"}
_MAX_FPS = 24.0

mcp = FastMCP("video-analyzer", host="0.0.0.0", stateless_http=True, json_response=True)

_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start time as MM:SS or H:MM:SS"},
                    "end": {"type": "string", "description": "End time as MM:SS or H:MM:SS"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["start", "title", "summary"],
            },
        },
        "key_moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "MM:SS or H:MM:SS"},
                    "description": {"type": "string"},
                    "visual_importance": {"type": "string"},
                },
                "required": ["timestamp", "description"],
            },
        },
        "on_screen_text": {"type": "array", "items": {"type": "string"}},
        "visual_style": {"type": "string"},
        "people_objects_locations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "sections", "key_moments", "on_screen_text"],
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _model(override: str = "") -> str:
    """The per-call model, else GEMINI_MODEL. GEMINI_ALLOWED_MODELS (comma list) limits overrides."""
    if not override.strip():
        return _env("GEMINI_MODEL", DEFAULT_MODEL)
    model = override.strip()
    if not re.fullmatch(r"gemini-[a-z0-9.\-]+", model):
        raise ValueError(f"model must be a Gemini model id like gemini-3.8-pro; got {model!r}.")
    allowed = {m.strip() for m in os.getenv("GEMINI_ALLOWED_MODELS", "").split(",") if m.strip()}
    if allowed and model not in allowed:
        raise ValueError(f"model {model!r} is not in GEMINI_ALLOWED_MODELS ({', '.join(sorted(allowed))}).")
    return model


def _thinking_level(override: str = "") -> str:
    level = (override.strip() or _env("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL)).lower()
    if level not in _ALLOWED_THINKING:
        raise ValueError(f"thinking_level must be one of {sorted(_ALLOWED_THINKING)}; got {level!r}.")
    return level


def _media_resolution_name(override: str = "") -> str:
    level = (override.strip() or _env("GEMINI_MEDIA_RESOLUTION", DEFAULT_MEDIA_RESOLUTION)).lower()
    if level not in _ALLOWED_MEDIA_RESOLUTION:
        raise ValueError(f"media_resolution must be one of {sorted(_ALLOWED_MEDIA_RESOLUTION)}; got {level!r}.")
    return level


def _media_resolution(level: str) -> types.MediaResolution:
    return {
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }[level]


def _parse_timestamp(value: str) -> int:
    raw = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return int(float(raw))
    parts = raw.split(":")
    if len(parts) not in {2, 3} or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid timestamp {value!r}; use seconds, MM:SS, or H:MM:SS.")
    nums = [int(p) for p in parts]
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60 + seconds
    hours, minutes, seconds = nums
    return hours * 3600 + minutes * 60 + seconds


def _video_metadata(time_range: str, fps: float = 0) -> types.VideoMetadata | None:
    if fps < 0 or fps > _MAX_FPS:
        raise ValueError(f"fps must be between 0 (model default, about 1) and {_MAX_FPS:g}; got {fps:g}.")
    if not time_range.strip():
        return types.VideoMetadata(fps=fps) if fps else None
    if "-" not in time_range:
        raise ValueError("time_range must be START-END, e.g. 05:00-12:30.")
    start_raw, end_raw = (part.strip() for part in time_range.split("-", 1))
    start = _parse_timestamp(start_raw)
    end = _parse_timestamp(end_raw)
    if start < 0 or end <= start:
        raise ValueError("time_range end must be greater than start.")
    return types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s", fps=fps or None)


def _output_schema(output_schema: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if output_schema is None or output_schema == "" or output_schema == {}:
        return None
    schema = json.loads(output_schema) if isinstance(output_schema, str) else output_schema
    if not isinstance(schema, dict):
        raise ValueError("output_schema must be a JSON Schema object.")
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _schema_errors(data: Any, schema: dict[str, Any], limit: int = 20) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}" for e in errors[:limit]]


def _prompt(focus: str, prompt_override: str = "") -> str:
    if prompt_override.strip():
        prompt = prompt_override.strip()
        if focus.strip():
            prompt += f"\n\nFOCUS: {focus.strip()}"
        return prompt
    prompt = (
        "Analyze the supplied video using both the visible frames and the audio/narration. "
        "Do not infer details that are not supported by the video. Return a structured, "
        "timestamped analysis. Capture the main narrative, section boundaries, visually "
        "important moments, meaningful on-screen text, the overall visual style, and the "
        "important people/objects/locations visible. Timestamps should point to where the "
        "described content is actually visible or happening."
    )
    if focus.strip():
        prompt += f"\n\nFOCUS: {focus.strip()}"
    return prompt


def _analyze(
    url: str,
    focus: str = "",
    time_range: str = "",
    prompt: str = "",
    output_schema: dict[str, Any] | str | None = None,
    model: str = "",
    thinking_level: str = "",
    media_resolution: str = "",
    fps: float = 0,
    max_output_tokens: int = 0,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GEMINI_API_KEY is not configured on the server."}

    model_name = _model(model)
    thinking = _thinking_level(thinking_level)
    media_resolution_name = _media_resolution_name(media_resolution)
    schema = _output_schema(output_schema)
    if max_output_tokens < 0:
        raise ValueError("max_output_tokens must be 0 (model default) or more.")

    client = genai.Client(api_key=api_key)
    metadata = _video_metadata(time_range, fps)
    video_part = types.Part(file_data=types.FileData(file_uri=url), video_metadata=metadata)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=schema or _ANALYSIS_SCHEMA,
        media_resolution=_media_resolution(media_resolution_name),
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
        max_output_tokens=max_output_tokens or None,
    )
    contents = types.Content(parts=[video_part, types.Part(text=_prompt(focus, prompt))])
    meta = {
        "model": model_name,
        "thinking_level": thinking,
        "media_resolution": media_resolution_name,
        "fps": fps or None,
        "source": url,
        "time_range": time_range or None,
        "focus": focus or None,
        "custom_prompt": bool(prompt.strip()),
        "custom_schema": schema is not None,
    }

    # A custom schema is checked after the call; one retry, then a structured error (never malformed success).
    attempts = 2 if schema else 1
    errors: list[str] = []
    for _ in range(attempts):
        response = client.models.generate_content(model=model_name, contents=contents, config=config)
        candidate = (response.candidates or [None])[0]
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
        meta["usage"] = _usage(response)
        if finish == "MAX_TOKENS":
            return {"status": "truncated", "error": "Gemini hit the output token limit; raise max_output_tokens or narrow time_range.", **meta}
        if not response.text:
            return {"status": "error", "error": "Gemini returned an empty response.", "finish_reason": finish, **meta}
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as exc:
            errors = [f"invalid JSON: {exc}"]
            continue
        errors = _schema_errors(data, schema) if schema else []
        if not errors:
            return {"status": "success", **meta, "analysis": data}
    return {"status": "invalid", "error": "Gemini output did not match output_schema.", "schema_errors": errors, **meta}


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
    }


@mcp.tool()
def analyze_video(
    url: str,
    focus: str = "",
    time_range: str = "",
    prompt: str = "",
    output_schema: dict[str, Any] | str | None = None,
    model: str = "",
    thinking_level: str = "",
    media_resolution: str = "",
    fps: float = 0,
    max_output_tokens: int = 0,
) -> str:
    """Analyze a video with Gemini and return timestamped structured JSON.

    Intended for remote agent use. `url` should be a video source supported by the
    configured Gemini model (YouTube URLs are supported by the upstream workflow).

    Optional `focus` asks Gemini to pay extra attention to a subject. Optional
    `time_range` restricts analysis to START-END in seconds, MM:SS, or H:MM:SS.

    Custom extraction (all optional):
    - `prompt` replaces the built-in instructions (`focus` is still appended).
    - `output_schema` is a JSON Schema (object or JSON string) Gemini must fill instead of
      the built-in one. The reply is validated against it, retried once, and returned with
      status "invalid" and the schema errors if it still doesn't match.
    - `model` picks any Gemini model for this call (e.g. a Pro model); the server may limit
      choices with GEMINI_ALLOWED_MODELS.
    - `thinking_level` (low|medium|high) and `media_resolution` (low|medium|high) override
      the server defaults for this call.
    - `fps` samples the video at this many frames per second (default about 1, max 24);
      raise it for fast cuts. Cost grows with fps and resolution.
    - `max_output_tokens` raises the output cap for long structured replies; a reply cut
      at the cap comes back with status "truncated".

    Every reply includes `usage` (input, output and thinking tokens).

    Server defaults come from environment variables:
    - GEMINI_MODEL (default: gemini-3.8-flash)
    - GEMINI_THINKING_LEVEL (low|medium|high, default: medium)
    - GEMINI_MEDIA_RESOLUTION (low|medium|high, default: low)
    """
    try:
        result = _analyze(
            url,
            focus=focus,
            time_range=time_range,
            prompt=prompt,
            output_schema=output_schema,
            model=model,
            thinking_level=thinking_level,
            media_resolution=media_resolution,
            fps=fps,
            max_output_tokens=max_output_tokens,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {"status": "error", "error": f"{exc.__class__.__name__}: {exc}", "model": model or _env("GEMINI_MODEL", DEFAULT_MODEL)},
            ensure_ascii=False,
        )
