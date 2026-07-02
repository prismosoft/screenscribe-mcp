"""Tests for local-file upload to Gemini (Files API) and _call_gemini routing.

No live API: a fake genai client stands in for the SDK. Exercises the poll-until-
ACTIVE loop and that a local path is uploaded (rather than passed as a file_uri)."""

import pytest

import screenscribe.gemini_selector as gs


class _State:
    def __init__(self, name):
        self.name = name


class _File:
    def __init__(self, state, name="files/abc", uri="https://g/files/abc", mime="video/mp4"):
        self.state = _State(state)
        self.name = name
        self.uri = uri
        self.mime_type = mime


class _Files:
    def __init__(self, upload_state, get_states):
        self.upload_state = upload_state
        self.get_states = list(get_states)
        self.upload_arg = None
        self.get_calls = 0

    def upload(self, file=None):
        self.upload_arg = file
        return _File(self.upload_state)

    def get(self, name=None):
        self.get_calls += 1
        return _File(self.get_states.pop(0))


class _Models:
    def __init__(self):
        self.contents = None

    def generate_content(self, model=None, contents=None, config=None):
        self.contents = contents
        return type("R", (), {"text": "[]"})()


class _Client:
    def __init__(self, upload_state="ACTIVE", get_states=()):
        self.files = _Files(upload_state, get_states)
        self.models = _Models()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(gs.time, "sleep", lambda *_: None)


def _mp4(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return str(p.resolve())


def test_upload_returns_active_file_immediately(tmp_path):
    client = _Client(upload_state="ACTIVE")
    path = _mp4(tmp_path)
    f = gs._upload_local_video(client, path)
    assert f.state.name == "ACTIVE"
    assert client.files.upload_arg == path
    assert client.files.get_calls == 0


def test_upload_polls_until_active(tmp_path):
    client = _Client(upload_state="PROCESSING", get_states=["PROCESSING", "ACTIVE"])
    f = gs._upload_local_video(client, _mp4(tmp_path))
    assert f.state.name == "ACTIVE"
    assert client.files.get_calls == 2


def test_upload_raises_on_failed_state(tmp_path):
    client = _Client(upload_state="PROCESSING", get_states=["FAILED"])
    with pytest.raises(RuntimeError):
        gs._upload_local_video(client, _mp4(tmp_path))


def test_upload_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "UPLOAD_TIMEOUT", 0.0)
    client = _Client(upload_state="PROCESSING", get_states=["PROCESSING"])
    with pytest.raises(RuntimeError):
        gs._upload_local_video(client, _mp4(tmp_path))


def test_call_gemini_uploads_local_file(tmp_path, monkeypatch):
    from google import genai

    gs._UPLOADED.clear()
    client = _Client(upload_state="ACTIVE")
    monkeypatch.setattr(genai, "Client", lambda **k: client)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    path = _mp4(tmp_path)
    gs._call_gemini(path, "model", "prompt", None, True)

    # Uploaded the bytes, and referenced the uploaded file's uri (not the raw path).
    assert client.files.upload_arg == path
    part = client.models.contents.parts[0]
    assert part.file_data.file_uri == "https://g/files/abc"


def test_call_gemini_youtube_url_is_not_uploaded(tmp_path, monkeypatch):
    from google import genai

    gs._UPLOADED.clear()
    client = _Client(upload_state="ACTIVE")
    monkeypatch.setattr(genai, "Client", lambda **k: client)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    url = "https://youtu.be/dQw4w9WgXcQ"
    gs._call_gemini(url, "model", "prompt", None, True)

    assert client.files.upload_arg is None            # no upload for a URL
    part = client.models.contents.parts[0]
    assert part.file_data.file_uri == url
