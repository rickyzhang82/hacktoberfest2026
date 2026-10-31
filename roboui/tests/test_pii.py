import pytest

from roboui.pii import redact


@pytest.mark.parametrize("text,kind", [
    ("mail me at jane.doe@example.com", "EMAIL"),
    ("key sk-abcdefghijklmnop1234", "API_KEY"),
    ("AKIAIOSFODNN7EXAMPLE", "AWS_KEY"),
    ("ghp_" + "a" * 36, "GITHUB_TOKEN"),
    ("api_key = abcd1234efgh5678", "KEY_ASSIGNMENT"),
    ("SSN 123-45-6789", "SSN"),
    ("call 415-555-0134", "PHONE"),
])
def test_masks_pii(text, kind):
    r = redact(text)
    assert f"[REDACTED:{kind}]" in r.text
    assert kind in r.counts


def test_clean_text_untouched():
    r = redact("A header with a Sign Up button and 3 cards")
    assert r.text == "A header with a Sign Up button and 3 cards" and r.total == 0


def test_secret_value_removed():
    assert "jane.doe" not in redact("jane.doe@example.com").text
