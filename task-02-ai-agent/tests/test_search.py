"""Unit tests for Wikipedia knowledge search tool."""

import pytest
from tools.search import search_wikipedia, TOOL_DEFINITION


def test_search_tool_definition():
    """Verify tool definition dictionary conforms to standard."""
    assert TOOL_DEFINITION["name"] == "wikipedia_search"
    assert "query" in TOOL_DEFINITION["parameters"]["properties"]
    assert "query" in TOOL_DEFINITION["parameters"]["required"]
    assert callable(TOOL_DEFINITION["function"])


def test_empty_query():
    """Verify empty query string raises descriptive ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        search_wikipedia("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        search_wikipedia("")


def test_live_wikipedia_summary():
    """Verify live search retrieval returns title, extract, and url."""
    res = search_wikipedia("Alan Turing")

    assert "title" in res
    assert "extract" in res
    assert "url" in res
    assert "Turing" in res["title"]
    assert len(res["extract"]) > 30
    assert "wikipedia.org" in res["url"]


def test_nonexistent_query():
    """Verify nonexistent topic raises graceful ValueError."""
    with pytest.raises(ValueError, match="No Wikipedia article found"):
        search_wikipedia("asldkjfhasldkjfhasdlkjfh98723948723498xyz")
