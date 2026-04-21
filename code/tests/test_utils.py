from src.utils import clamp, sanitize_filename, utc_timestamp
from src.utils import extract_json_text


def test_sanitize_filename():
    assert sanitize_filename("openai/gpt-4o-mini") == "openai_gpt-4o-mini"
    assert sanitize_filename("  weird name  ") == "weird_name"


def test_clamp():
    assert clamp(-1.0) == 0.0
    assert clamp(0.4) == 0.4
    assert clamp(2.0) == 1.0


def test_timestamp_shape():
    ts = utc_timestamp()
    assert len(ts) == 15
    assert ts[8] == "-"


def test_extract_json_text_from_fenced_block():
    raw = """```json
    {
      "a": 1,
      "b": ["x", "y"]
    }
    ```"""
    assert extract_json_text(raw) == {"a": 1, "b": ["x", "y"]}


def test_extract_json_text_from_prefixed_text():
    raw = "Here you go:\n{\"score\": 1, \"rationale\": \"ok\"}\nThanks."
    assert extract_json_text(raw) == {"score": 1, "rationale": "ok"}
