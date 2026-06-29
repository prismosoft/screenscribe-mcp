"""
screenscribe — CLI

Commands:
  extract <url>              Download + extract Gemini-selected key frames as PNGs.
  extract <url> --transcript-only
                             Fetch transcript only — no key, no download.
  slides <url>               Extract standalone "slide" frames as PNGs.
  analyze <url>              Gemini whole-video structured analysis (no frames).
  sessions                   List all processed videos.

Frames are saved as PNGs under ~/.video-analyzer/<id>/. Open them to see what is
on screen — answering questions about a video is your agent/LLM's job; point it at
the extracted frames and the transcript. Everything except transcript needs a
GEMINI_API_KEY.

Examples:
  screenscribe extract "https://youtu.be/RnP08K2SAZs" --transcript-only
  screenscribe extract "https://youtu.be/RnP08K2SAZs" --focus "architecture diagrams"
  screenscribe slides  "https://youtu.be/RnP08K2SAZs"
  screenscribe analyze "https://youtu.be/RnP08K2SAZs"
  screenscribe sessions
"""

import argparse
import json
import sys

from dotenv import load_dotenv
load_dotenv()

import yt_dlp

from screenscribe.config import (
    FRAME_SELECTION_MAX,
    FRAME_SELECTION_MIN_INTERVAL,
    GEMINI_MEDIA_RESOLUTION_LOW,
    GEMINI_MODEL,
    IMAGE_MAX_WIDTH,
    SLIDE_SELECTION_MAX,
    SLIDE_SELECTION_MIN_INTERVAL,
)
from screenscribe.downloader import download_video, fetch_transcript, fetch_transcript_safe
from screenscribe.frame_extractor import extract_frames_at_timestamps
from screenscribe.gemini_selector import gemini_available, select_frames, select_slides
from screenscribe.session import (
    frames_dir as session_frames_dir,
    list_sessions,
    load_analysis,
    save_analysis,
    save_session,
    session_dir,
    session_exists,
    slides_dir as session_slides_dir,
)


from screenscribe.resolver import parse_video_id as extract_video_id


def get_video_title(url: str) -> str:
    """Fetch video title without downloading."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown")
    except Exception:
        return "Unknown"


def _require_gemini_for_frames(timestamps: str):
    """Frame selection needs a Gemini key unless explicit timestamps are given."""
    if not timestamps and not gemini_available():
        print("ERROR: frame extraction needs GEMINI_API_KEY (Gemini watches the video to "
              "pick frames).\n       Pass --timestamps to bypass selection, or use "
              "--transcript-only for text only.")
        sys.exit(1)


def _video_duration(transcript: list[dict]) -> float:
    if not transcript:
        return 0.0
    last = transcript[-1]
    return last["start"] + last.get("duration", 0)


# ── extract ───────────────────────────────────────────────────────────────────

def cmd_extract(args):
    video_id = extract_video_id(args.url)
    s_dir = session_dir(video_id)
    f_dir = session_frames_dir(video_id)
    transcript_only = args.transcript_only

    print(f"\n{'=' * 44}")
    print(f"  screenscribe extract")
    print(f"  Video ID : {video_id}")
    print(f"  Session  : {s_dir}")
    print(f"  Mode     : {'transcript-only (no frames)' if transcript_only else 'frames'}")
    if args.focus:
        print(f"  Focus    : {args.focus}")
    if args.time_range:
        print(f"  Range    : {args.time_range}")
    if args.timestamps:
        print(f"  Stamps   : {args.timestamps}")
    print(f"{'=' * 44}\n")

    if session_exists(video_id) and not args.force:
        print("Session already exists. Use --force to re-extract.")
        return

    # ── Transcript-only mode (free, no key) ───────────────────────────────
    if transcript_only:
        print("[1/1] Fetching transcript...")
        s_dir.mkdir(parents=True, exist_ok=True)
        transcript = fetch_transcript(video_id, s_dir)
        session_path = save_session(
            video_id=video_id, url=args.url, title=get_video_title(args.url),
            duration=_video_duration(transcript), transcript=transcript, frames=[],
        )
        print(f"\n{'=' * 44}")
        print(f"  DONE — transcript-only session saved")
        print(f"  {session_path}")
        print(f"  Transcript: {len(transcript)} segments")
        print(f"{'=' * 44}\n")
        return

    # ── Frame extraction (Gemini-selected) ────────────────────────────────
    _require_gemini_for_frames(args.timestamps)

    print("[1/3] Downloading video and transcript...")
    title = get_video_title(args.url)
    video_path, _, _ = download_video(args.url, s_dir)
    transcript = fetch_transcript_safe(video_id, s_dir)
    video_duration = _video_duration(transcript)

    print("\n[2/3] Identifying key visual moments...")
    selections = select_frames(
        args.url, gemini_model=GEMINI_MODEL, max_frames=args.max_frames,
        min_interval=FRAME_SELECTION_MIN_INTERVAL, focus=args.focus,
        time_range=args.time_range, timestamps=args.timestamps,
        video_duration=video_duration, media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
    )
    print(f"  Identified {len(selections)} key moments")
    for i, sel in enumerate(selections, 1):
        print(f"    {i:2d}. {sel['timestamp']:.1f}s — {sel['reason']}")

    if not selections:
        print("ERROR: No moments selected.")
        sys.exit(1)

    print(f"\n[3/3] Extracting {len(selections)} frames...")
    frames = extract_frames_at_timestamps(
        video_path=video_path, frames_dir=f_dir, selections=selections, max_width=IMAGE_MAX_WIDTH,
    )
    if not frames:
        print("ERROR: No frames extracted.")
        sys.exit(1)

    duration = frames[-1]["timestamp"] if frames else video_duration
    session_path = save_session(
        video_id=video_id, url=args.url, title=title,
        duration=duration, transcript=transcript, frames=frames,
    )

    print(f"\n{'=' * 44}")
    print(f"  DONE — {len(frames)} frames saved")
    print(f"  {f_dir}/")
    for i, f in enumerate(frames, 1):
        print(f"    {i:2d}. {f['timestamp']:.1f}s — {f.get('reason', '')}")
    print(f"  Session: {session_path}")
    print(f"{'=' * 44}\n")


# ── slides ────────────────────────────────────────────────────────────────────

def cmd_slides(args):
    video_id = extract_video_id(args.url)
    s_dir = session_dir(video_id)
    sl_dir = session_slides_dir(video_id)
    has_custom_params = bool(args.focus or args.time_range or args.timestamps)

    print(f"\n{'=' * 44}")
    print(f"  screenscribe slides")
    print(f"  Video ID : {video_id}")
    print(f"  Session  : {s_dir}")
    print(f"  Max      : {args.max_slides} slides")
    if args.focus:
        print(f"  Focus    : {args.focus}")
    if args.time_range:
        print(f"  Range    : {args.time_range}")
    if args.timestamps:
        print(f"  Stamps   : {args.timestamps}")
    print(f"{'=' * 44}\n")

    # Cache check — skip cache when custom params are set
    slides_meta = sl_dir / "frames.json"
    if slides_meta.exists() and not args.force and not has_custom_params:
        slides = json.loads(slides_meta.read_text())
        print(f"Slides already extracted ({len(slides)} slides). Use --force to re-extract.\n")
        for i, s in enumerate(slides, 1):
            print(f"  {i:2d}. {s['timestamp']:.1f}s — {s.get('reason', '')}")
            print(f"      {s['path']}")
        return

    _require_gemini_for_frames(args.timestamps)

    print("[1/3] Ensuring video and transcript...")
    video_path = None
    if s_dir.exists():
        candidates = [f for f in s_dir.iterdir()
                      if f.suffix in ('.mp4', '.mkv', '.webm') and f.stem != 'thumbnail']
        if candidates:
            video_path = candidates[0]
            print(f"  Video found: {video_path.name}")
    title = get_video_title(args.url)
    if video_path is None:
        video_path, _, _ = download_video(args.url, s_dir)

    transcript_file = s_dir / "transcript.json"
    if transcript_file.exists():
        transcript = json.loads(transcript_file.read_text())
        print(f"  Transcript found: {len(transcript)} segments")
    else:
        transcript = fetch_transcript_safe(video_id, s_dir)

    print("\n[2/3] Identifying slide-worthy moments...")
    selections = select_slides(
        args.url, gemini_model=GEMINI_MODEL, max_slides=args.max_slides,
        min_interval=SLIDE_SELECTION_MIN_INTERVAL, focus=args.focus,
        time_range=args.time_range, timestamps=args.timestamps,
        video_duration=_video_duration(transcript), media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
    )
    print(f"  Identified {len(selections)} slide moments:")
    for i, sel in enumerate(selections, 1):
        print(f"    {i:2d}. {sel['timestamp']:.1f}s — {sel['reason']}")

    if not selections:
        print("ERROR: No slide moments selected.")
        sys.exit(1)

    print(f"\n[3/3] Extracting {len(selections)} slides...")
    slides = extract_frames_at_timestamps(
        video_path=video_path, frames_dir=sl_dir, selections=selections, max_width=IMAGE_MAX_WIDTH,
    )

    if not session_exists(video_id):
        save_session(
            video_id=video_id, url=args.url, title=title,
            duration=_video_duration(transcript), transcript=transcript, frames=[],
        )

    print(f"\n{'=' * 44}")
    print(f"  DONE — {len(slides)} slides extracted")
    print(f"  {sl_dir}/")
    for i, s in enumerate(slides, 1):
        print(f"    {i:2d}. {s['timestamp']:.1f}s — {s.get('reason', '')}")
    print(f"{'=' * 44}\n")


# ── analyze ───────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    from screenscribe.gemini_analyzer import analyze_video_with_gemini, gemini_available

    if not gemini_available():
        print("ERROR: `analyze` needs GEMINI_API_KEY — Gemini watches the whole video.")
        sys.exit(1)

    video_id = extract_video_id(args.url)
    s_dir = session_dir(video_id)

    if load_analysis(video_id) is not None and not args.force:
        print(f"Analysis already exists for {video_id}. Use --force to regenerate.")
        return

    print(f"Analyzing the whole video with Gemini ({GEMINI_MODEL})...")
    analysis = analyze_video_with_gemini(
        args.url, GEMINI_MODEL, focus=args.focus, time_range=args.time_range,
        media_resolution_low=GEMINI_MEDIA_RESOLUTION_LOW,
    )
    save_analysis(video_id, analysis)

    # Fetch the transcript (free) so the session is queryable with text too;
    # create a session only if one doesn't already exist (don't clobber frames).
    s_dir.mkdir(parents=True, exist_ok=True)
    if not session_exists(video_id):
        try:
            transcript = fetch_transcript(video_id, s_dir)
        except Exception:
            transcript = []
        save_session(
            video_id=video_id, url=args.url, title=get_video_title(args.url),
            duration=_video_duration(transcript), transcript=transcript, frames=[],
        )

    print(f"\n  Session  : {video_id}")
    print(f"  Summary  : {analysis.get('summary', '')[:300]}")
    print(f"  {len(analysis.get('sections', []))} sections, "
          f"{len(analysis.get('key_moments', []))} key moments")
    print(f"  Analysis : {s_dir / 'gemini_analysis.json'}")


# ── extract-structured ──────────────────────────────────────────────────────────

def cmd_extract_structured(args):
    from screenscribe.structured_extractor import extract_structured, list_presets

    try:
        result = extract_structured(
            args.url, args.schema, focus=args.focus,
            time_range=args.time_range, force=args.force,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"Presets: {', '.join(list_presets())}", file=sys.stderr)
        sys.exit(1)

    status = result.get("status")
    if status == "success":
        # Data → stdout (pipeable); metadata → stderr.
        print(json.dumps(result["data"], indent=2))
        print(f"[{result['key']}] cached={result['cached']} session={result['session_id']}",
              file=sys.stderr)
    elif status == "invalid":
        print(f"ERROR: model output failed schema validation: {result['error']}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR: {result.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)


# ── sessions ──────────────────────────────────────────────────────────────────

def cmd_sessions(_args):
    sessions = list_sessions()
    if not sessions:
        print("No sessions found. Run: screenscribe extract <url>")
        return

    print(f"\n{'─' * 60}")
    print(f"  {'SESSION ID':<20} {'FRAMES':>6}  {'DUR':>5}  TITLE")
    print(f"{'─' * 60}")
    for s in sessions:
        dur = f"{int(s['duration'] // 60)}m{int(s['duration'] % 60):02d}s"
        title = s['title'][:30] + ("…" if len(s['title']) > 30 else "")
        print(f"  {s['session_id']:<20} {s['frames']:>6}  {dur:>5}  {title}")
    print(f"{'─' * 60}\n")


# ── synthesize ──────────────────────────────────────────────────────────────────

def cmd_synthesize_categorize(args):
    from screenscribe.synthesis import categorize

    out = categorize(args.source, max_videos=args.max_videos,
                     min_duration=args.min_duration, force=args.force)
    if out.get("status") != "success":
        print(f"ERROR: {out.get('error', out)}", file=sys.stderr)
        sys.exit(1)

    ranked = " (ranked by views)" if out.get("ranked") else ""
    print(f"\n{out['kind']} — {out['total']} videos, {len(out['categories'])} categories{ranked}:")
    for c in out["categories"]:
        print(f"  {c['count']:4d}  {c['name']}")
    print(f"\nNext: screenscribe synthesize pass \"{args.source}\" --category <name> "
          f"--item-schema <preset|path> --aggregate-schema <preset|path>")


def cmd_synthesize_pass(args):
    from screenscribe.synthesis import synthesize_pass

    out = synthesize_pass(
        args.source, args.category or None,
        item_schema=args.item_schema, aggregate_schema=args.aggregate_schema,
        top_n=args.top_n, focus=args.focus, force=args.force,
        media_resolution_low=(args.media_res == "low"),
    )
    status = out.get("status")
    if status == "success":
        print(json.dumps(out["aggregate"], indent=2))   # the artifact → stdout (pipeable)
        print(f"[{out['category']}] added={len(out['added'])} "
              f"failed={len(out['extraction_failed'])} passes={out['passes_so_far']} "
              f"truncated={out['truncated']}", file=sys.stderr)
    elif status == "invalid":
        print(f"ERROR: aggregate failed schema validation: {out.get('error')}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR: {out.get('error', out)}", file=sys.stderr)
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="screenscribe",
        description="Turn instructional videos into queryable knowledge sessions.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # extract
    p_extract = sub.add_parser("extract", help="Download a video and extract key frames (run once)")
    p_extract.add_argument("url", help="YouTube URL")
    p_extract.add_argument("--max-frames", type=int, default=FRAME_SELECTION_MAX,
                           help=f"Max frames Gemini selects (default {FRAME_SELECTION_MAX})")
    p_extract.add_argument("--transcript-only", action="store_true",
                           help="Fetch transcript only — no key, no video download or frames")
    p_extract.add_argument("--force", action="store_true",
                           help="Re-extract even if session already exists")
    p_extract.add_argument("--focus", type=str, default="",
                           help="Focus on specific content (e.g. 'architecture diagrams')")
    p_extract.add_argument("--time-range", type=str, default="",
                           help="Restrict to time range: START-END in seconds or MM:SS (e.g. '5:00-15:00')")
    p_extract.add_argument("--timestamps", type=str, default="",
                           help="Extract at exact timestamps, bypass AI selection (e.g. '5:30,10:00')")

    # slides
    p_slides = sub.add_parser("slides", help="Extract presentation slides from a video")
    p_slides.add_argument("url", help="YouTube URL")
    p_slides.add_argument("--max-slides", type=int, default=SLIDE_SELECTION_MAX,
                          help=f"Max slides to extract (default {SLIDE_SELECTION_MAX})")
    p_slides.add_argument("--force", action="store_true",
                          help="Re-extract even if slides already exist")
    p_slides.add_argument("--focus", type=str, default="",
                          help="Focus on specific content (e.g. 'only code examples')")
    p_slides.add_argument("--time-range", type=str, default="",
                          help="Restrict to time range: START-END in seconds or MM:SS (e.g. '5:00-15:00')")
    p_slides.add_argument("--timestamps", type=str, default="",
                          help="Extract at exact timestamps, bypass AI selection (e.g. '5:30,10:00')")

    # analyze
    p_analyze = sub.add_parser("analyze",
                               help="Gemini watches the whole video → structured analysis (cheap, no frames)")
    p_analyze.add_argument("url", help="YouTube URL")
    p_analyze.add_argument("--focus", type=str, default="",
                           help="Focus the analysis on a specific subject")
    p_analyze.add_argument("--time-range", type=str, default="",
                           help="Restrict to time range: START-END in seconds or MM:SS")
    p_analyze.add_argument("--force", action="store_true",
                           help="Regenerate even if an analysis already exists")

    # extract-structured
    p_struct = sub.add_parser("extract-structured",
                              help="Extract typed JSON from a video against a schema/preset")
    p_struct.add_argument("url", help="YouTube URL")
    p_struct.add_argument("--schema", required=True,
                          help="Preset name, path to a .json schema, or inline JSON schema. "
                               "Presets: cli_commands, final_config, step_sequence, "
                               "code_blocks, resources_mentioned, chapters, recipe")
    p_struct.add_argument("--focus", type=str, default="",
                          help="Narrow what to extract (e.g. 'only the auth setup')")
    p_struct.add_argument("--time-range", type=str, default="",
                          help="Restrict to time range: START-END in seconds or MM:SS")
    p_struct.add_argument("--force", action="store_true",
                          help="Re-run even if a cached result exists")

    # synthesize (nested: categorize | pass)
    p_syn = sub.add_parser("synthesize",
                           help="Cross-video synthesis: categorize a channel, then compound passes")
    syn_sub = p_syn.add_subparsers(dest="syn_action", metavar="ACTION")

    p_cat = syn_sub.add_parser("categorize",
                               help="Discover categories from a channel/playlist's titles (cheap, no extraction)")
    p_cat.add_argument("source", help="Channel / playlist URL")
    p_cat.add_argument("--max-videos", type=int, default=None, help="Cap videos resolved")
    p_cat.add_argument("--min-duration", type=int, default=0, help="Drop videos shorter than N seconds")
    p_cat.add_argument("--force", action="store_true", help="Re-classify even if cached")

    p_pass = syn_sub.add_parser("pass",
                                help="Fold top-N of a category (or the whole set) into a compounding aggregate")
    p_pass.add_argument("source", help="Channel / playlist / video URL")
    p_pass.add_argument("--item-schema", required=True,
                        help="Per-video schema: preset name, path, or inline JSON Schema")
    p_pass.add_argument("--aggregate-schema", required=True,
                        help="Aggregate schema: preset (e.g. 'cookbook'), path, or inline JSON Schema")
    p_pass.add_argument("--category", default="",
                        help="Category from `categorize` to pace this pass; omit for the whole set")
    p_pass.add_argument("--top", type=int, default=20, dest="top_n", help="Max videos this pass (default 20)")
    p_pass.add_argument("--focus", type=str, default="", help="Narrow what each extraction captures")
    p_pass.add_argument("--media-res", choices=["low", "medium"], default="low",
                        help="medium reads tiny on-screen detail (e.g. code) better; default low")
    p_pass.add_argument("--force", action="store_true", help="Re-extract + re-fold even if cached/included")

    # sessions
    sub.add_parser("sessions", help="List all processed videos")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "slides":
        cmd_slides(args)
    elif args.command == "extract-structured":
        cmd_extract_structured(args)
    elif args.command == "synthesize":
        if args.syn_action == "categorize":
            cmd_synthesize_categorize(args)
        elif args.syn_action == "pass":
            cmd_synthesize_pass(args)
        else:
            p_syn.print_help()
    elif args.command == "sessions":
        cmd_sessions(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
