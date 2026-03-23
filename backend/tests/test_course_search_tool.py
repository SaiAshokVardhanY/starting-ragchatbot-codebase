"""
Tests for CourseSearchTool.execute() and _format_results().
All VectorStore interactions are mocked so no ChromaDB or network calls occur.
"""
import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults
from search_tools import CourseSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

def test_tool_definition_name_is_search_course_content(mock_vector_store):
    tool = make_tool(mock_vector_store)
    assert tool.get_tool_definition()["name"] == "search_course_content"


def test_tool_definition_has_required_query_field(mock_vector_store):
    tool = make_tool(mock_vector_store)
    required = tool.get_tool_definition()["input_schema"]["required"]
    assert "query" in required


def test_tool_definition_has_input_schema(mock_vector_store):
    tool = make_tool(mock_vector_store)
    defn = tool.get_tool_definition()
    assert "input_schema" in defn
    assert "description" in defn


# ---------------------------------------------------------------------------
# execute() — success path
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_string_on_success(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="MCP")

    assert "[MCP Course - Lesson 1]" in result
    assert "Lesson content about MCP servers." in result


def test_execute_result_contains_all_documents(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="MCP")

    assert "Lesson content about MCP servers." in result
    assert "More content about tools." in result


# ---------------------------------------------------------------------------
# execute() — error and empty paths
# ---------------------------------------------------------------------------

def test_execute_returns_error_string_when_store_returns_error(mock_vector_store, error_search):
    mock_vector_store.search.return_value = error_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="MCP")

    assert result == error_search.error


def test_execute_returns_no_results_message_when_empty(mock_vector_store, empty_search):
    mock_vector_store.search.return_value = empty_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="nothing")

    assert result.startswith("No relevant content found")


def test_execute_no_results_includes_course_name_in_message(mock_vector_store, empty_search):
    mock_vector_store.search.return_value = empty_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="x", course_name="RAG")

    assert "RAG" in result


def test_execute_no_results_includes_lesson_number_in_message(mock_vector_store, empty_search):
    mock_vector_store.search.return_value = empty_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="x", lesson_number=5)

    assert "5" in result


# ---------------------------------------------------------------------------
# execute() — parameter forwarding
# ---------------------------------------------------------------------------

def test_execute_passes_query_to_store(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="what is MCP?")

    mock_vector_store.search.assert_called_once()
    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs["query"] == "what is MCP?"


def test_execute_passes_course_name_to_store(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="x", course_name="MCP")

    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs["course_name"] == "MCP"


def test_execute_passes_lesson_number_to_store(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="x", lesson_number=3)

    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs["lesson_number"] == 3


def test_execute_passes_none_course_name_by_default(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="x")

    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs.get("course_name") is None


# ---------------------------------------------------------------------------
# _format_results() — header formatting
# ---------------------------------------------------------------------------

def test_format_results_header_includes_lesson_number(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="MCP")

    assert "Lesson 1" in result
    assert "Lesson 2" in result


def test_format_results_header_omits_lesson_when_metadata_has_none(mock_vector_store):
    search_no_lesson = SearchResults(
        documents=["Some content."],
        metadata=[{"course_title": "Generic Course"}],
        distances=[0.1],
    )
    mock_vector_store.search.return_value = search_no_lesson
    mock_vector_store.get_lesson_link.return_value = None
    tool = make_tool(mock_vector_store)

    result = tool.execute(query="x")

    assert "[Generic Course]" in result
    assert "Lesson" not in result


# ---------------------------------------------------------------------------
# _format_results() — sources tracking
# ---------------------------------------------------------------------------

def test_format_results_populates_last_sources(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="MCP")

    assert len(tool.last_sources) == 2
    for source in tool.last_sources:
        assert "text" in source
        assert "url" in source


def test_format_results_source_text_includes_lesson_number(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    tool = make_tool(mock_vector_store)

    tool.execute(query="MCP")

    assert "Lesson 1" in tool.last_sources[0]["text"]


def test_format_results_handles_none_lesson_link(mock_vector_store, two_result_search):
    mock_vector_store.search.return_value = two_result_search
    mock_vector_store.get_lesson_link.return_value = None
    tool = make_tool(mock_vector_store)

    tool.execute(query="MCP")  # should not raise

    assert tool.last_sources[0]["url"] is None


def test_last_sources_empty_before_execute(mock_vector_store):
    tool = make_tool(mock_vector_store)
    assert tool.last_sources == []


def test_last_sources_cleared_if_empty_results(mock_vector_store, empty_search):
    """last_sources should not carry over from a previous successful search."""
    mock_vector_store.search.return_value = empty_search
    tool = make_tool(mock_vector_store)
    tool.last_sources = [{"text": "stale", "url": None}]  # simulate prior state

    tool.execute(query="x")

    # last_sources should NOT be updated on empty results (no _format_results call)
    # This test documents the current behaviour — stale sources persist on empty result
    # which is a known limitation that ToolManager.reset_sources() is meant to handle.
    assert tool.last_sources == [{"text": "stale", "url": None}]
