import json
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "src" / "screenscribe" / "schemas"

EXPECTED_PRESETS = {
    "cli_commands", "final_config", "step_sequence",
    "code_blocks", "resources_mentioned", "chapters", "recipe",
}


def test_all_expected_presets_exist():
    found = {p.stem for p in SCHEMAS_DIR.glob("*.json")}
    assert EXPECTED_PRESETS <= found, f"missing presets: {EXPECTED_PRESETS - found}"


@pytest.mark.parametrize("name", sorted(EXPECTED_PRESETS))
def test_preset_is_valid_json_schema(name):
    schema = json.loads((SCHEMAS_DIR / f"{name}.json").read_text())
    # Raises SchemaError if the schema itself is malformed.
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema.get("type") == "object"


from screenscribe.structured_extractor import (
    build_extraction_prompt,
    list_presets,
    load_preset,
    resolve_schema,
    schema_key,
    validate_output,
)


def test_list_presets_returns_all_seven():
    assert set(list_presets()) == EXPECTED_PRESETS


def test_load_preset_known_and_unknown():
    assert load_preset("cli_commands")["type"] == "object"
    assert load_preset("nope") is None


def test_recipe_preset_exposes_hero_shot_seconds():
    # The recipe preset asks Gemini for the best frame to screenshot for a recipe
    # card (a hero image for the website), as a numeric timestamp.
    hero = load_preset("recipe")["properties"]["hero_shot"]
    assert hero["properties"]["seconds"]["type"] == "number"


def test_recipe_preset_captures_native_title():
    # The dish name in its original script (e.g. Bengali) for the website cards.
    assert load_preset("recipe")["properties"]["title_bn"]["type"] == "string"


def test_recipe_preset_tips_carry_a_timestamp():
    # Each tip records the second it is given, so a downstream exporter can attach
    # it to the recipe step in progress at that moment.
    items = load_preset("recipe")["properties"]["tips"]["items"]
    assert items["type"] == "object"
    assert "text" in items["properties"]
    assert items["properties"]["seconds"]["type"] == "number"


def test_resolve_schema_dict_passthrough():
    s = {"type": "object"}
    assert resolve_schema(s) is s


def test_resolve_schema_preset_name():
    assert resolve_schema("cli_commands")["type"] == "object"


def test_resolve_schema_inline_json():
    assert resolve_schema('{"type": "array"}') == {"type": "array"}


def test_resolve_schema_file_path(tmp_path):
    p = tmp_path / "shape.json"
    p.write_text('{"type": "object"}')
    assert resolve_schema(str(p)) == {"type": "object"}


def test_resolve_schema_unknown_raises():
    with pytest.raises(ValueError):
        resolve_schema("this is not json, a file, or a preset")


def test_schema_key_preset_is_name():
    assert schema_key("cli_commands") == "cli_commands"


def test_schema_key_freeform_is_stable_12_hex():
    a = schema_key({"b": 1, "a": 2})
    b = schema_key({"a": 2, "b": 1})
    assert a == b
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


def test_validate_output_accepts_valid():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    ok, data, err = validate_output('{"n": 5}', schema)
    assert ok and data == {"n": 5} and err == ""


def test_validate_output_rejects_schema_violation():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    ok, data, err = validate_output('{"n": "five"}', schema)
    assert not ok and data is None and err


def test_validate_output_rejects_malformed_json():
    ok, data, err = validate_output("not json", {"type": "object"})
    assert not ok and "JSON" in err


def test_build_extraction_prompt_includes_focus_and_description():
    schema = {"type": "object", "description": "the CLI commands"}
    prompt = build_extraction_prompt(schema, "auth setup")
    assert "the CLI commands" in prompt and "auth setup" in prompt


import screenscribe.gemini_selector as gs
import screenscribe.session as sess
import screenscribe.structured_extractor as se

_OBJ_SCHEMA = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(sess, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(gs, "gemini_available", lambda: True)


def test_extract_structured_success(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(gs, "_call_gemini", lambda *a, **k: '{"n": 7}')
    out = se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    assert out["status"] == "success"
    assert out["data"] == {"n": 7}
    assert out["cached"] is False
    assert out["session_id"] == "abc123"


def test_extract_structured_retries_then_succeeds(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    calls = []
    def fake(*a, **k):
        calls.append(1)
        return '{"n": "bad"}' if len(calls) == 1 else '{"n": 7}'
    monkeypatch.setattr(gs, "_call_gemini", fake)
    out = se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    assert len(calls) == 2
    assert out["status"] == "success" and out["data"] == {"n": 7}


def test_extract_structured_invalid_after_retry(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(gs, "_call_gemini", lambda *a, **k: '{"n": "bad"}')
    out = se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    assert out["status"] == "invalid"
    assert out["raw"] == '{"n": "bad"}'
    assert out["error"]


def test_extract_structured_uses_cache(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    calls = []
    def fake(*a, **k):
        calls.append(1)
        return '{"n": 7}'
    monkeypatch.setattr(gs, "_call_gemini", fake)
    se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    out = se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    assert len(calls) == 1          # second served from cache
    assert out["cached"] is True
    assert out["data"] == {"n": 7}


def test_extract_structured_requires_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(sess, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(gs, "gemini_available", lambda: False)
    out = se.extract_structured("https://youtu.be/abc123", _OBJ_SCHEMA)
    assert out["status"] == "error"


def test_extract_structured_local_file(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(gs, "_call_gemini", lambda *a, **k: '{"n": 7}')
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    out = se.extract_structured(str(clip), _OBJ_SCHEMA)
    assert out["status"] == "success"
    assert out["data"] == {"n": 7}
    assert out["session_id"].startswith("local-")
    # Cache is keyed on the synthesized local id — a re-run is free.
    assert se.extract_structured(str(clip), _OBJ_SCHEMA)["cached"] is True


# ── batch fan-out (extract over resolved videos) ─────────────────────────────

import screenscribe.resolver as _rv


def test_batch_fans_out_and_isolates_failures(monkeypatch):
    monkeypatch.setattr(_rv, "resolve_videos", lambda source, **k: {
        "kind": "list", "source": source, "video_ids": ["a", "b", "c"],
        "title": None, "skipped": {"too_short": 1, "unavailable": 0}, "total_found": 3,
    })
    calls = []

    def fake_extract(url, schema, **k):
        calls.append(url)
        vid = url.split("v=")[-1]
        if vid == "b":
            return {"status": "invalid", "key": "k", "error": "bad", "raw": "{}"}
        return {"status": "success", "session_id": vid, "key": "k", "cached": False, "data": {"n": 1}}

    monkeypatch.setattr(se, "extract_structured", fake_extract)
    out = se.extract_structured_batch("any-source", {"type": "object"})

    assert out["kind"] == "list"
    assert out["total_videos"] == 3
    assert len(calls) == 3                                   # one extraction per video
    assert [s["video_id"] for s in out["succeeded"]] == ["a", "c"]
    assert [f["video_id"] for f in out["failed"]] == ["b"]   # failure isolated, run continues
    assert out["resolver_skipped"]["too_short"] == 1
    assert out["resolver_total_found"] == 3


def test_batch_passes_cached_flag_through(monkeypatch):
    monkeypatch.setattr(_rv, "resolve_videos", lambda source, **k: {
        "kind": "video", "source": source, "video_ids": ["x"],
        "title": None, "skipped": {"too_short": 0, "unavailable": 0}, "total_found": 1,
    })
    monkeypatch.setattr(se, "extract_structured", lambda url, schema, **k: {
        "status": "success", "session_id": "x", "key": "k", "cached": True, "data": {"ok": 1},
    })
    out = se.extract_structured_batch("https://youtu.be/x", "cli_commands")
    assert out["succeeded"][0]["cached"] is True
    assert out["succeeded"][0]["data"] == {"ok": 1}
