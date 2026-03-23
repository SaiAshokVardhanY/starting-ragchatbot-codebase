"""
Integration tests for RAGSystem.query().

All external dependencies (VectorStore, AIGenerator, Anthropic client,
SentenceTransformer) are mocked so tests run offline with no ChromaDB.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY


# ---------------------------------------------------------------------------
# Mock config
# ---------------------------------------------------------------------------

class MockConfig:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-test"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "/tmp/test_chroma_rag"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag(tmp_path):
    """
    Creates a RAGSystem with all heavy dependencies mocked out.
    Patches: VectorStore, AIGenerator, DocumentProcessor, and the
    chromadb/sentence_transformers imports that VectorStore uses at module level.
    """
    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.DocumentProcessor") as MockDP,
    ):
        mock_vs = MagicMock()
        mock_vs.get_existing_course_titles.return_value = []
        mock_vs.get_course_count.return_value = 3
        MockVS.return_value = mock_vs

        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "Mock AI answer"
        MockAI.return_value = mock_ai

        MockDP.return_value = MagicMock()

        from rag_system import RAGSystem
        config = MockConfig()
        config.CHROMA_PATH = str(tmp_path / "chroma")

        system = RAGSystem(config)
        # Expose mocks for assertions in tests
        system._mock_vs = mock_vs
        system._mock_ai = mock_ai
        yield system


# ---------------------------------------------------------------------------
# Basic query return shape
# ---------------------------------------------------------------------------

def test_query_returns_tuple_of_response_and_sources(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[{"text": "Source", "url": None}])
    rag.tool_manager.reset_sources = MagicMock()

    result = rag.query("What is RAG?")

    assert isinstance(result, tuple)
    assert len(result) == 2
    response, sources = result
    assert response == "Mock AI answer"
    assert sources == [{"text": "Source", "url": None}]


def test_query_returns_empty_sources_when_no_tool_used(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    _, sources = rag.query("What is Python?")

    assert sources == []


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_query_prompt_wraps_user_question(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    rag.query("What is RAG?")

    call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
    assert call_kwargs["query"] == "Answer this question about course materials: What is RAG?"


# ---------------------------------------------------------------------------
# Session / history
# ---------------------------------------------------------------------------

def test_query_passes_session_history_to_generator(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    session_id = rag.session_manager.create_session()
    rag.session_manager.add_exchange(session_id, "hi", "hello")

    rag.query("Follow up question", session_id=session_id)

    call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
    assert call_kwargs["conversation_history"] is not None
    assert "hi" in call_kwargs["conversation_history"]


def test_query_passes_none_history_for_new_session(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    session_id = rag.session_manager.create_session()
    # No exchanges added — history should be None
    rag.query("First question", session_id=session_id)

    call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
    assert call_kwargs["conversation_history"] is None


def test_query_updates_session_with_exchange(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    session_id = rag.session_manager.create_session()
    rag.query("What is MCP?", session_id=session_id)

    history = rag.session_manager.get_conversation_history(session_id)
    assert history is not None
    assert "What is MCP?" in history
    assert "Mock AI answer" in history


def test_query_with_no_session_id_still_returns_response(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    response, sources = rag.query("What is Python?", session_id=None)

    assert response == "Mock AI answer"


# ---------------------------------------------------------------------------
# Sources lifecycle
# ---------------------------------------------------------------------------

def test_query_sources_reset_after_retrieval(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    rag.query("What is RAG?")

    rag.tool_manager.reset_sources.assert_called_once()


def test_query_sources_reset_called_after_get_last_sources(rag):
    call_order = []
    rag.tool_manager.get_last_sources = MagicMock(
        side_effect=lambda: call_order.append("get") or []
    )
    rag.tool_manager.reset_sources = MagicMock(
        side_effect=lambda: call_order.append("reset")
    )

    rag.query("test")

    assert call_order.index("get") < call_order.index("reset")


# ---------------------------------------------------------------------------
# Tools registration
# ---------------------------------------------------------------------------

def test_both_tools_registered_at_init(rag):
    registered_names = list(rag.tool_manager.tools.keys())
    assert "search_course_content" in registered_names
    assert "get_course_outline" in registered_names


def test_tool_definitions_passed_to_ai_generator(rag):
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()

    rag.query("What is MCP?")

    call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
    # tools should be the list of definitions from the tool_manager
    assert "tools" in call_kwargs
    tool_names = [t["name"] for t in call_kwargs["tools"]]
    assert "search_course_content" in tool_names
    assert "get_course_outline" in tool_names


# ---------------------------------------------------------------------------
# BUG #1 (end-to-end) — Exception propagation
# ---------------------------------------------------------------------------

def test_query_exception_propagates_from_ai_generator(rag):
    """
    FIX: ai_generator.generate_response() now wraps raw API errors in
    RuntimeError("AI service unavailable: ..."). The exception still propagates
    through RAGSystem.query() → app.py → HTTP 500, but with a clear message
    instead of a raw SDK traceback.
    """
    rag.tool_manager.get_last_sources = MagicMock(return_value=[])
    rag.tool_manager.reset_sources = MagicMock()
    rag._mock_ai.generate_response.side_effect = RuntimeError("AI service unavailable: API auth failed")

    with pytest.raises(RuntimeError, match="AI service unavailable"):
        rag.query("What is covered in lesson 1 of the MCP course?")
