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

from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_LEVEL = "medium"
DEFAULT_MEDIA_RESOLUTION = "low"
_ALLOWED_THINKING = {"low", "medium", "high"}
_ALLOWED_MEDIA_RESOLUTION = {"low", "medium", "high"}

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


def _thinking_level() -> str:
    level = _env("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL).lower()
    if level not in _ALLOWED_THINKING:
        raise ValueError(f"GEMINI_THINKING_LEVEL must be one of {sorted(_ALLOWED_THINKING)}; got {level!r}.")
    return level


def _media_resolution() -> types.MediaResolution:
    level = _env("GEMINI_MEDIA_RESOLUTION", DEFAULT_MEDIA_RESOLUTION).lower()
    if level not in _ALLOWED_MEDIA_RESOLUTION:
        raise ValueError(f"GEMINI_MEDIA_RESOLUTION must be one of {sorted(_ALLOWED_MEDIA_RESOLUTION)}; got {level!r}.")
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


def _video_metadata(time_range: str) -> types.VideoMetadata | None:
    if not time_range.strip():
        return None
    if "-" not in time_range:
        raise ValueError("time_range must be START-END, e.g. 05:00-12:30.")
    start_raw, end_raw = (part.strip() for part in time_range.split("-", 1))
    start = _parse_timestamp(start_raw)
    end = _parse_timestamp(end_raw)
    if start < 0 or end <= start:
        raise ValueError("time_range end must be greater than start.")
    return types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s")


def _prompt(focus: str) -> str:
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


def _analyze(url: str, focus: str = "", time_range: str = "") -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GEMINI_API_KEY is not configured on the server."}

    model = _env("GEMINI_MODEL", DEFAULT_MODEL)
    thinking = _thinking_level()
    media_resolution_name = _env("GEMINI_MEDIA_RESOLUTION", DEFAULT_MEDIA_RESOLUTION).lower()

    client = genai.Client(api_key=api_key)
    metadata = _video_metadata(time_range)
    video_part = types.Part(file_data=types.FileData(file_uri=url), video_metadata=metadata)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_ANALYSIS_SCHEMA,
        media_resolution=_media_resolution(),
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
    )

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[video_part, types.Part(text=_prompt(focus))]),
        config=config,
    )
    if not response.text:
        return {"status": "error", "error": "Gemini returned an empty response."}

    data = json.loads(response.text)
    return {
        "status": "success",
        "model": model,
        "thinking_level": thinking,
        "media_resolution": media_resolution_name,
        "source": url,
        "time_range": time_range or None,
        "focus": focus or None,
        "analysis": data,
    }


@mcp.tool()
def analyze_video(url: str, focus: str = "", time_range: str = "") -> str:
    """Analyze a video with Gemini and return timestamped structured JSON.

    Intended for remote agent use. `url` should be a video source supported by the
    configured Gemini model (YouTube URLs are supported by the upstream workflow).

    Optional `focus` asks Gemini to pay extra attention to a subject. Optional
    `time_range` restricts analysis to START-END in seconds, MM:SS, or H:MM:SS.

    Server configuration is controlled by environment variables:
    - GEMINI_MODEL (default: gemini-3.8-flash)
    - GEMINI_THINKING_LEVEL (low|medium|high, default: medium)
    - GEMINI_MEDIA_RESOLUTION (low|medium|high, default: low)
    """
    try:
        return json.dumps(_analyze(url, focus=focus, time_range=time_range), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {"status": "error", "error": f"{exc.__class__.__name__}: {exc}", "model": _env("GEMINI_MODEL", DEFAULT_MODEL)},
            ensure_ascii=False,
        )
