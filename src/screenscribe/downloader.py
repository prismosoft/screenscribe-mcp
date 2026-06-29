import json
import os
from pathlib import Path

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from screenscribe.ffmpeg_paths import ffmpeg_dir

_ytt = YouTubeTranscriptApi()


def _cookie_opts() -> dict:
    """yt-dlp cookie options from the environment, to authenticate downloads when
    YouTube challenges with 'Sign in to confirm you're not a bot'. Set ONE of:
      YTDLP_COOKIES_FROM_BROWSER=safari|firefox|chrome|...   (read browser cookies)
      YTDLP_COOKIES_FILE=/path/cookies.txt                   (Netscape cookies file)
    Returns {} when neither is set."""
    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
    if browser:
        return {"cookiesfrombrowser": (browser,)}
    cookie_file = os.environ.get("YTDLP_COOKIES_FILE")
    if cookie_file:
        return {"cookiefile": cookie_file}
    return {}


def download_video(url: str, output_dir: Path) -> tuple[Path, str, list[dict]]:
    """Download video to output_dir. Returns (video_path, video_id)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
    }

    # bestvideo+bestaudio makes yt-dlp merge streams with ffmpeg; point it at the
    # bundled binary so a download works with no system ffmpeg installed.
    ff_dir = ffmpeg_dir()
    if ff_dir:
        ydl_opts["ffmpeg_location"] = ff_dir

    ydl_opts.update(_cookie_opts())   # authenticate if cookies are configured

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    # yt-dlp may merge into .mkv if mp4 isn't available — find whatever was saved
    candidates = list(output_dir.glob(f"{video_id}.*"))
    video_path = candidates[0] if candidates else output_dir / f"{video_id}.mp4"

    # Save chapter metadata if available
    chapters = info.get("chapters") or []
    if chapters:
        chapters_path = output_dir / "chapters.json"
        chapters_path.write_text(json.dumps(chapters, indent=2))
        print(f"  Chapters saved: {len(chapters)} chapters → {chapters_path}")

    print(f"  Video saved: {video_path}")
    return video_path, video_id, chapters


def fetch_transcript(video_id: str, output_dir: Path) -> list[dict]:
    """
    Fetch the YouTube transcript for video_id.
    Each entry: {'text': str, 'start': float, 'duration': float}
    """
    fetched = _ytt.fetch(video_id)
    transcript = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]

    transcript_path = output_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2))

    print(f"  Transcript saved: {len(transcript)} segments → {transcript_path}")
    return transcript


def fetch_transcript_safe(video_id: str, output_dir: Path) -> list[dict]:
    """fetch_transcript, but non-fatal: returns [] (with a warning) when no
    transcript is available — e.g. a non-English video with no English captions.
    Use this where the transcript is optional context (frame extraction watches
    the video directly), so a missing transcript must not abort the command."""
    try:
        return fetch_transcript(video_id, output_dir)
    except Exception as e:
        print(f"  Warning: no transcript fetched ({e.__class__.__name__}); continuing without it.")
        return []
