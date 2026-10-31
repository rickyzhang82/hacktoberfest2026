import pytest

from roboui.errors import CodeExtractionError
from roboui.extract import extract_html

DOC = "<!DOCTYPE html><html><body><h1>Hi</h1></body></html>"


def test_strips_html_fence():
    assert extract_html(f"```html\n{DOC}\n```") == DOC


def test_strips_fence_with_surrounding_prose():
    assert extract_html(f"Sure! Here you go:\n```html\n{DOC}\n```\nEnjoy.") == DOC


def test_uppercase_and_no_language_fence():
    assert extract_html(f"```HTML\n{DOC}\n```") == DOC
    assert extract_html(f"```\n{DOC}\n```") == DOC


def test_unclosed_fence_from_truncated_output():
    out = extract_html(f"```html\n{DOC}")
    assert out == DOC and "```" not in out


def test_bare_document_with_chatter():
    assert extract_html(f"Here:\n{DOC}\nDone!") == DOC


def test_fragment_is_returned():
    assert extract_html("```html\n<div class='a'>x</div>\n```") == "<div class='a'>x</div>"


def test_picks_full_document_among_multiple_blocks():
    raw = f"```html\n<div>snippet</div>\n```\n```html\n{DOC}\n```"
    assert extract_html(raw) == DOC


def test_no_backticks_remain():
    assert "```" not in extract_html(f"```html\n{DOC}\n```")


@pytest.mark.parametrize("raw", ["", "   \n", "I cannot help with that."])
def test_raises_when_no_html(raw):
    with pytest.raises(CodeExtractionError):
        extract_html(raw)
