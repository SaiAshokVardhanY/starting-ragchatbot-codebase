import sys
import os
import pytest

# Ensure backend/ is on the path so test files can import backend modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
from vector_store import SearchResults


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson"
    return store


@pytest.fixture
def two_result_search():
    """SearchResults with 2 hits — used across tool tests."""
    return SearchResults(
        documents=["Lesson content about MCP servers.", "More content about tools."],
        metadata=[
            {"course_title": "MCP Course", "lesson_number": 1},
            {"course_title": "MCP Course", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search():
    return SearchResults.empty(
        "Search error: n_results can only be less than or equal to the number of elements"
    )
