"""
screenscribe MCP server.

Built on Gemini, which watches the video to pick frames and produce whole-video
analysis. The agent (the MCP client) does the reasoning over what's returned —
including viewing the extracted frame images. A free transcript layer runs
alongside. Needs only a GEMINI_API_KEY (frames/analysis); transcript is free.

Exposes tools to any MCP client (Claude Code, Claude Desktop, etc.):
  extract_transcript(url)    — fast: fetch transcript only (free, no key)
  analyze_video(url)         — cheap: Gemini watches the whole video → structured analysis
  extract_frames(url, style) — Gemini picks moments, ffmpeg extracts PNGs the agent views
                               (style="keyframes" | "slides")
  extract_structured(url, schema) — extract typed JSON against a schema/preset
  synthesize_categorize(url) — discover categories from a channel/playlist's titles (cheap)
  synthesize_pass(url, ...)  — fold top-N of a category into a compounding aggregate
  get_video_analysis(id)     — read Gemini's whole-video analysis
  get_session(session_id)    — return session data (transcript, analysis, frame paths)
  list_sessions()            — list all processed videos

Claude Code usage:
  claude mcp add screenscribe -- uvx screenscribe-mcp

  Or add to ~/.claude.json directly:
    {
      "mcpServers": {
        "screenscribe": {
          "command": "uvx",
          "args": ["screenscribe-mcp"],
          "env": { "GEMINI_API_KEY": "..." }
        }
      }
    }
  (The key is also read from the shell environment / a .env in the working dir.)

Then in your client, just mention a YouTube URL — the agent will call
extract_frames / analyze_video as needed, then use get_session to answer
questions with full repo context.
"""

import json

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from screenscribe.config import (
    FRAME_SELECTION_MAX,
    FRAME_SELECTION_MIN_INTERVAL,
    GEMINI_MEDIA_RESOLUTION_LOW,
    GEMINI_MODEL,
    IMAGE_MAX_WIDTH,
    MAX_INLINE_TRANSCRIPT_CHARS,
    SLIDE_SELECTION_MAX,
    SLIDE_SELECTION_MIN_INTERVAL,
)
from screenscribe.downloader import download_video, fetch_transcript, fetch_transcript_safe
from screenscribe.frame_extractor import extract_frames_at_timestamps
from screenscribe.session import (
    frames_dir as session_frames_dir,
    list_sessions as _list_sessions,
    load_analysis,
    load_session,
    save_analysis,
    save_session,
    session_dir,
    session_exists,
    slides_dir as session_slides_dir,
)

mcp = FastMCP("screenscribe")


from screenscribe.resolver import parse_video_id as _extract_video_id


def _get_title(url: str) -> str:
    """Fetch video title without downloading."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown")
    except Exception:
        return "Unknown"


@mcp.tool()
def extract_transcript(url: str) -> str:
    """
    Fetch the transcript of a YouTube video. Fast and free — no video
    download, no frame analysis, no API credits used.

    Use this by default when a user shares a YouTube URL and wants to
    discuss, summarize, or ask questions about its content. For most
    videos the transcript alone is sufficient.

    Only use extract_frames instead if the user specifically needs
    visual/frame analysis (e.g. "what's shown on screen", charts,
    diagrams, code on screen).

    Returns: session_id to use with get_session.
    """
    video_id = _extract_video_id(url)

    if session_exists(video_id):
        session = load_session(video_id)
        return json.dumps({
            "status": "already_extracted",
            "session_id": video_id,
            "title": session.get("title", "Unknown"),
            "frame_count": session.get("frame_count", 0),
            "message": f"Session already exists. Use get_session('{video_id}') to access it.",
        })

    s_dir = session_dir(video_id)
    title = _get_title(url)

    s_dir.mkdir(parents=True, exist_ok=True)
    transcript = fetch_transcript(video_id, s_dir)

    duration = 0.0
    if transcript:
        last = transcript[-1]
        duration = last["start"] + last["duration"]

    save_session(
        video_id=video_id,
        url=url,
        title=title,
        duration=duration,
        transcript=transcript,
        frames=[],
    )

    return json.dumps({
        "status": "success",
        "session_id": video_id,
        "title": title,
        "frame_count": 0,
        "duration_seconds": duration,
        "mode": "transcript_only",
        "transcript_segments": len(transcript),
        "message": f"Transcript-only session ready. Call get_session('{video_id}') to access the content.",
    })


@mcp.tool()
def extract_frames(
    url: str,
    style: str = "keyframes",
    focus: str = "",
    time_range: str = "",
    timestamps: str = "",
) -> str:
    """
    Extract frames from a YouTube video as PNG images you (the agent) can open and
    read directly. Gemini watches the video to pick the moments; ffmpeg extracts
    them. There is no server-side description step — you view the images yourself.

    Use this when the user needs the actual visuals (what's on screen: code,
    diagrams, UI, demonstrations, scenes). For text-only questions,
    extract_transcript or analyze_video is usually enough.

    style:
    - "keyframes" (default): a denser set of visually important moments.
    - "slides": fewer, complete standalone visuals (finished diagrams, charts,
      code, summaries) that work as a slide deck.

    Requires GEMINI_API_KEY for selection — UNLESS you pass explicit `timestamps`.

    Optional:
    - focus: narrow what to capture, e.g. "only code examples".
    - time_range: "START-END" in seconds or MM:SS, e.g. "5:00-15:00".
    - timestamps: comma-separated exact timestamps (seconds or MM:SS), bypassing
      AI selection, e.g. "5:30,10:00".

    Returns: the saved PNG paths + timestamps + reasons. Open the images to read
    what is on screen.
    """
    from screenscribe.gemini_selector import gemini_available, select_frames, select_slides

    if style not in ("keyframes", "slides"):
        return json.dumps({"status": "error", "message": f"Unknown style '{style}'. Use 'keyframes' or 'slides'."})

    if not timestamps and not gemini_available():
        return json.dumps({
            "status": "error",
            "message": "extract_frames needs GEMINI_API_KEY (Gemini watches the video to pick frames). "
                       "Pass explicit `timestamps` to bypass selection, or use extract_transcript for text only.",
        })

    try:
        video_id = _extract_video_id(url)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})

    s_dir = session_dir(video_id)
    out_dir = session_slides_dir(video_id) if style == "slides" else session_frames_dir(video_id)
    has_custom_params = bool(focus or time_range or timestamps)

    # Cache: reuse a prior extraction of this style when no custom params are set.
    meta = out_dir / "frames.json"
    if meta.exists() and not has_custom_params:
        cached = json.loads(meta.read_text())
        if cached:
            return json.dumps({
                "status": "cached",
                "session_id": video_id,
                "style": style,
                "frame_count": len(cached),
                "frames": [
                    {"index": i + 1, "timestamp": f["timestamp"], "path": f["path"], "reason": f.get("reason", "")}
                    for i, f in enumerate(cached)
                ],
                "message": f"{len(cached)} {style} frames already extracted.",
            })

    try:
        # Reuse a previously downloaded video if present, else download.
        video_path = None
        if s_dir.exists():
            candidates = [f for f in s_dir.iterdir()
                          if f.suffix in ('.mp4', '.mkv', '.webm') and f.stem != 'thumbnail']
            if candidates:
                video_path = candidates[0]
        if video_path is None:
            video_path, _, _ = download_video(url, s_dir)

        transcript_file = s_dir / "transcript.json"
        if transcript_file.exists():
            transcript = json.loads(transcript_file.read_text())
        else:
            transcript = fetch_transcript_safe(video_id, s_dir)

        video_duration = (
            transcript[-1]["start"] + transcript[-1].get("duration", 0) if transcript else 0.0
        )

        if style == "slides":
            selections = select_slides(
                url, gemini_model=GEMINI_MODEL, max_slides=SLIDE_SELECTION_MAX,
                min_interval=SLIDE_SELECTION_MIN_INTERVAL, focus=focus, time_range=time_range,
                timestamps=timestamps, video_duration=video_duration,
                media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
            )
        else:
            selections = select_frames(
                url, gemini_model=GEMINI_MODEL, max_frames=FRAME_SELECTION_MAX,
                min_interval=FRAME_SELECTION_MIN_INTERVAL, focus=focus, time_range=time_range,
                timestamps=timestamps, video_duration=video_duration,
                media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
            )

        if not selections:
            return json.dumps({"status": "error", "message": "No moments were selected."})

        frames = extract_frames_at_timestamps(
            video_path=video_path,
            frames_dir=out_dir,
            selections=selections,
            max_width=IMAGE_MAX_WIDTH,
            save_metadata=not has_custom_params,
        )
        if not frames:
            return json.dumps({"status": "error", "message": "No frames extracted."})

        # Persist: keyframes go into the session (so get_session can surface them);
        # slides live in slides/frames.json. Always ensure a session exists.
        if style == "keyframes" and not has_custom_params:
            save_session(
                video_id=video_id, url=url, title=_get_title(url),
                duration=video_duration, transcript=transcript, frames=frames,
            )
        elif not session_exists(video_id):
            save_session(
                video_id=video_id, url=url, title=_get_title(url),
                duration=video_duration, transcript=transcript, frames=[],
            )

        return json.dumps({
            "status": "success",
            "session_id": video_id,
            "style": style,
            "frame_count": len(frames),
            "frames": [
                {"index": i + 1, "timestamp": f["timestamp"], "path": f["path"], "reason": f.get("reason", "")}
                for i, f in enumerate(frames)
            ],
            "message": f"Extracted {len(frames)} {style} frames. Open the PNG paths to view them.",
        })
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def analyze_video(url: str, focus: str = "", time_range: str = "") -> str:
    """
    Whole-video visual analysis with Gemini — cheap and fast. Gemini watches the
    entire video and returns a structured understanding: a summary, a section/
    topic breakdown with timestamps, the key moments (with what is on screen),
    and notable on-screen text/data. No download, no frame extraction.

    This is the sweet spot between extract_transcript (free, words only) and
    extract_frames (the actual frame images): whole-video visual coverage for a
    fraction of the cost. Use it to summarise, outline, or answer "what is shown /
    what happens" questions. Use extract_frames only when you need the frame
    images saved on disk for the agent to view.

    Optional:
    - focus: pay special attention to a subject (e.g. "the demo", "pricing").
    - time_range: restrict to a portion, "START-END" in seconds or MM:SS.

    Returns: session_id; read the full analysis with get_video_analysis.
    """
    video_id = _extract_video_id(url)
    has_custom = bool(focus or time_range)

    if load_analysis(video_id) is not None and not has_custom:
        return json.dumps({
            "status": "already_analyzed",
            "session_id": video_id,
            "message": f"Analysis already exists. Use get_video_analysis('{video_id}').",
        })

    try:
        from screenscribe.gemini_analyzer import analyze_video_with_gemini, gemini_available
        if not gemini_available():
            return json.dumps({
                "status": "error",
                "message": "analyze_video requires GEMINI_API_KEY (Gemini watches the video).",
            })

        analysis = analyze_video_with_gemini(
            url, GEMINI_MODEL, focus=focus, time_range=time_range,
            media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
        )
        if not has_custom:
            save_analysis(video_id, analysis)

        # Ensure a session exists (with transcript) so get_session/ask can use it.
        if not session_exists(video_id):
            s_dir = session_dir(video_id)
            s_dir.mkdir(parents=True, exist_ok=True)
            try:
                transcript = fetch_transcript(video_id, s_dir)
            except Exception:
                transcript = []
            duration = 0.0
            if transcript:
                last = transcript[-1]
                duration = last["start"] + last.get("duration", 0)
            save_session(
                video_id=video_id, url=url, title=_get_title(url),
                duration=duration, transcript=transcript, frames=[],
            )

        resp = {
            "status": "success",
            "session_id": video_id,
            "summary": analysis.get("summary", ""),
            "section_count": len(analysis.get("sections", [])),
            "key_moment_count": len(analysis.get("key_moments", [])),
            "message": f"Analyzed. Read the full analysis with get_video_analysis('{video_id}').",
        }
        if has_custom:
            resp["analysis"] = analysis  # one-off (not cached) — return inline
        return json.dumps(resp)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_video_analysis(session_id: str) -> str:
    """
    Return Gemini's structured whole-video analysis for a session — summary,
    sections (with timestamps), key moments, and on-screen text — if
    analyze_video has been run for it.
    """
    analysis = load_analysis(session_id)
    if analysis is None:
        return json.dumps({
            "error": f"No Gemini analysis for '{session_id}'.",
            "hint": "Run analyze_video(url) first.",
        })
    return json.dumps({"session_id": session_id, "analysis": analysis})


@mcp.tool()
def get_session(session_id: str) -> str:
    """
    Return the full processed content of a video session: the transcript, any
    Gemini whole-video analysis, and the paths of any extracted key frames.

    The response includes an 'analysis_source' field telling you exactly what data
    is available. ALWAYS mention the source when answering — e.g. "Based on the
    transcript..." or "Based on transcript + N extracted frames...". When frames
    are present, open the PNG paths to read what is on screen.

    You (the agent) provide any codebase/project context from the current
    conversation — no need to pass it here.

    Args:
        session_id: The video ID returned by extract_transcript, extract_frames, or list_sessions.
    """
    try:
        session = load_session(session_id)
    except FileNotFoundError:
        return json.dumps({
            "error": f"No session found for '{session_id}'.",
            "hint": "Run extract_transcript(url) or extract_frames(url) first.",
        })

    full_transcript = " ".join(seg["text"] for seg in session["transcript"])
    frames = session.get("frames", [])
    gemini_analysis = load_analysis(session_id)

    # Build analysis source metadata
    if frames:
        analysis_source = {
            "type": "transcript + extracted frames",
            "frames": len(frames),
            "note": "Open the frame `path`s to view what is on screen.",
        }
    elif gemini_analysis:
        analysis_source = {
            "type": "transcript + Gemini whole-video analysis",
            "note": "Gemini watched the whole video; see gemini_analysis for the visual content.",
        }
    else:
        analysis_source = {
            "type": "transcript only",
            "note": "No frames extracted. Answers are based solely on the transcript.",
        }

    # Return the full transcript inline. Only when it exceeds a generous cap do
    # we return a preview and point to the on-disk file — never a silent cut.
    cap = MAX_INLINE_TRANSCRIPT_CHARS
    truncated = cap is not None and len(full_transcript) > cap
    transcript_inline = full_transcript[:cap] if truncated else full_transcript

    return json.dumps({
        "video_id": session["video_id"],
        "title": session.get("title", "Unknown"),
        "url": session.get("url", ""),
        "duration_seconds": session.get("duration", 0),
        "frame_count": session.get("frame_count", 0),
        "extracted_at": session.get("extracted_at", ""),
        "analysis_source": analysis_source,
        "gemini_analysis": gemini_analysis,
        "frames": [
            {"timestamp": f.get("timestamp"), "path": f.get("path"), "reason": f.get("reason", "")}
            for f in frames
        ],
        "transcript": transcript_inline,
        "transcript_chars": len(full_transcript),
        "transcript_truncated": truncated,
        "transcript_path": str(session_dir(session["video_id"]) / "transcript.json"),
    })


@mcp.tool()
def extract_structured(url: str, schema: str, focus: str = "", time_range: str = "") -> str:
    """
    Extract typed JSON from a video against a schema you provide. Gemini watches
    the whole video and returns JSON validated against the schema.

    schema: a preset name, a path to a .json schema file, or an inline JSON Schema
    string. Built-in presets: cli_commands, final_config, step_sequence,
    code_blocks, resources_mentioned, chapters, recipe.

    Works regardless of the video's language — Gemini transcribes the narration and
    reads on-screen text directly; no captions/transcript are required.

    Requires GEMINI_API_KEY. Returns the validated `data` on success, or a
    structured error ({"status": "invalid", ...}) if the model could not produce
    schema-conforming output after one retry.

    Optional:
    - focus: narrow what to extract, e.g. "only the auth setup".
    - time_range: "START-END" in seconds or MM:SS.
    """
    from screenscribe.structured_extractor import extract_structured as _extract, list_presets
    try:
        result = _extract(url, schema, focus=focus, time_range=time_range)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e), "presets": list_presets()})
    return json.dumps(result)


@mcp.tool()
def synthesize_categorize(url: str) -> str:
    """
    Discover dish/topic categories from a channel or playlist's video titles — one
    cheap classification pass, cached per source. Read-only: no extraction, no cost
    beyond a single text call. Use this FIRST to show the user the category buckets
    (with counts) before running any paid synthesis pass.

    Returns: {categories:[{name,count,video_ids}], by_id, ranked, total}.
    Needs GEMINI_API_KEY. Best for channel/playlist URLs (needs titles to classify).
    """
    from screenscribe.synthesis import categorize
    return json.dumps(categorize(url))


@mcp.tool()
def synthesize_pass(
    url: str,
    item_schema: str,
    aggregate_schema: str,
    category: str = "",
    top_n: int = 20,
    focus: str = "",
) -> str:
    """
    Fold the top-N videos of `category` (or the whole resolved set if `category` is
    empty) into a compounding, persisted aggregate that conforms to `aggregate_schema`.

    Resolves the source, extracts each video with `item_schema` (cached per video),
    and merges the new results into the running aggregate for (source, aggregate_schema).
    Call repeatedly — once per category — to grow the artifact; each pass is bounded
    by `top_n`, so it scales to a whole channel. One video's failure is isolated.

    item_schema / aggregate_schema: a preset name or an inline JSON Schema string.
    Requires GEMINI_API_KEY. Run synthesize_categorize first to pick a category.
    """
    from screenscribe.synthesis import synthesize_pass as _synth_pass
    try:
        return json.dumps(_synth_pass(
            url, category or None, item_schema=item_schema,
            aggregate_schema=aggregate_schema, top_n=top_n, focus=focus,
        ))
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def list_sessions() -> str:
    """
    List all videos that have been processed and are available to query.
    Returns session IDs, titles, durations, and extraction timestamps.
    """
    sessions = _list_sessions()
    if not sessions:
        return json.dumps({
            "sessions": [],
            "message": "No sessions yet. Run extract_frames(url) to process a video.",
        })
    return json.dumps({"sessions": sessions})


def run():
    """Console-script entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
