"""
Tests for the MCP server's analyze_video. genai.Client is replaced with a fake,
so these run with no API calls.
"""

import json
from types import SimpleNamespace

import pytest

from screenscribe import server

SCHEMA = {
    "type": "object",
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_s", "size"],
                "properties": {
                    "start_s": {"type": "number"},
                    "size": {"type": "string", "enum": ["CU", "WS"]},
                },
            },
        }
    },
}


class FakeClient:
    def __init__(self, replies, finish="STOP"):
        self.replies = list(replies)
        self.finish = finish
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        text = self.replies.pop(0)
        return SimpleNamespace(
            text=text,
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=self.finish))],
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50, thoughts_token_count=10),
        )


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.delenv("GEMINI_ALLOWED_MODELS", raising=False)
    holder = {}

    def install(replies, finish="STOP"):
        client = FakeClient(replies, finish)
        monkeypatch.setattr(server.genai, "Client", lambda api_key: client)
        holder["client"] = client
        return client

    return install


def _call(**kwargs):
    return json.loads(server.analyze_video("https://www.youtube.com/watch?v=x", **kwargs))


def test_default_call_uses_builtin_schema_and_env_model(fake):
    client = fake([json.dumps({"summary": "s", "sections": [], "key_moments": [], "on_screen_text": []})])
    out = _call()
    assert out["status"] == "success"
    assert out["model"] == server.DEFAULT_MODEL
    assert client.calls[0].config.response_json_schema == server._ANALYSIS_SCHEMA
    assert out["usage"] == {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 10}


def test_custom_schema_prompt_and_overrides(fake):
    client = fake([json.dumps({"shots": [{"start_s": 0, "size": "CU"}]})])
    out = _call(
        prompt="List every shot.",
        output_schema=SCHEMA,
        model="gemini-3.8-pro",
        thinking_level="high",
        media_resolution="medium",
        fps=5,
        max_output_tokens=60000,
    )
    assert out["status"] == "success"
    assert out["analysis"]["shots"][0]["size"] == "CU"
    call = client.calls[0]
    assert call.model == "gemini-3.8-pro"
    assert call.config.response_json_schema == SCHEMA
    assert call.config.max_output_tokens == 60000
    assert call.config.thinking_config.thinking_level.value.lower() == "high"
    assert call.contents.parts[0].video_metadata.fps == 5
    assert call.contents.parts[1].text == "List every shot."


def test_schema_as_json_string(fake):
    fake([json.dumps({"shots": []})])
    assert _call(output_schema=json.dumps(SCHEMA))["status"] == "success"


def test_invalid_output_retries_once_then_reports_errors(fake):
    bad = json.dumps({"shots": [{"start_s": 0, "size": "XL"}]})
    client = fake([bad, bad])
    out = _call(output_schema=SCHEMA)
    assert out["status"] == "invalid"
    assert len(client.calls) == 2
    assert any("shots/0/size" in e for e in out["schema_errors"])


def test_retry_can_recover(fake):
    client = fake(["not json", json.dumps({"shots": []})])
    assert _call(output_schema=SCHEMA)["status"] == "success"
    assert len(client.calls) == 2


def test_truncated_reply_is_reported(fake):
    fake(['{"shots": ['], finish="MAX_TOKENS")
    out = _call(output_schema=SCHEMA)
    assert out["status"] == "truncated"


def test_bad_schema_is_rejected_before_any_call(fake):
    client = fake([])
    out = _call(output_schema={"type": "not-a-type"})
    assert out["status"] == "error"
    assert client.calls == []


@pytest.mark.parametrize("kwargs", [{"model": "gpt-5"}, {"fps": 30}, {"thinking_level": "max"}, {"media_resolution": "ultra"}])
def test_bad_overrides_are_rejected(fake, kwargs):
    client = fake([])
    assert _call(**kwargs)["status"] == "error"
    assert client.calls == []


def test_allowed_models_limits_overrides(fake, monkeypatch):
    monkeypatch.setenv("GEMINI_ALLOWED_MODELS", "gemini-3.8-flash, gemini-3.8-pro")
    client = fake([json.dumps({"summary": "s", "sections": [], "key_moments": [], "on_screen_text": []})])
    assert _call(model="gemini-2.0-flash")["status"] == "error"
    assert _call(model="gemini-3.8-pro")["status"] == "success"
    assert client.calls[0].model == "gemini-3.8-pro"


def test_fps_without_time_range():
    meta = server._video_metadata("", 5)
    assert meta.fps == 5 and meta.start_offset is None
