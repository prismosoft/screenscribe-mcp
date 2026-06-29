"""
Tests for downloader helpers. fetch_transcript (network) is monkeypatched, so
these run with no network and no API key.
"""

from screenscribe import downloader


def test_cookie_opts_empty_by_default(monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES_FROM_BROWSER", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_FILE", raising=False)
    assert downloader._cookie_opts() == {}


def test_cookie_opts_from_browser(monkeypatch):
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "safari")
    monkeypatch.delenv("YTDLP_COOKIES_FILE", raising=False)
    assert downloader._cookie_opts() == {"cookiesfrombrowser": ("safari",)}


def test_cookie_opts_file(monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES_FROM_BROWSER", raising=False)
    monkeypatch.setenv("YTDLP_COOKIES_FILE", "/tmp/cookies.txt")
    assert downloader._cookie_opts() == {"cookiefile": "/tmp/cookies.txt"}


def test_fetch_transcript_safe_returns_transcript_on_success(monkeypatch, tmp_path):
    segments = [{"text": "hi", "start": 0.0, "duration": 1.0}]
    monkeypatch.setattr(downloader, "fetch_transcript", lambda vid, out: segments)
    assert downloader.fetch_transcript_safe("abc", tmp_path) == segments


def test_fetch_transcript_safe_warns_and_returns_empty_when_fetch_raises(monkeypatch, tmp_path, capsys):
    def boom(vid, out):
        raise RuntimeError("No transcript found in English")

    monkeypatch.setattr(downloader, "fetch_transcript", boom)
    result = downloader.fetch_transcript_safe("abc", tmp_path)
    assert result == []                                    # non-fatal: empty, not a crash
    assert "transcript" in capsys.readouterr().out.lower()  # never a silent cut: it warns
