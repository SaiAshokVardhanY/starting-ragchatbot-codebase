"""
Tests for AIGenerator.generate_response() — sequential tool calling.

All Anthropic API calls are mocked. Tests verify external behaviour:
  - tools are forwarded correctly to Claude on every intermediate round
  - tool execution is triggered on stop_reason == "tool_use"
  - natural termination (Claude stops mid-loop) returns without extra API call
  - forced synthesis call (MAX_ROUNDS hit) excludes tools
  - up to 2 sequential tool rounds are supported
  - exceptions from the API are wrapped as RuntimeError
  - _extract_text() safely finds the first text block
"""
import pytest
from unittest.mock import MagicMock, patch, call
import anthropic as anthropic_sdk

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_text_block(text="Hello"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name="search_course_content", input_data=None, block_id="tu_abc"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data or {"query": "MCP"}
    block.id = block_id
    return block


def make_response(stop_reason="end_turn", content=None):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content if content is not None else [make_text_block()]
    return resp


@pytest.fixture
def mock_client():
    """Patches anthropic.Anthropic so no real HTTP calls are made."""
    with patch("ai_generator.anthropic.Anthropic") as MockClass:
        instance = MagicMock()
        MockClass.return_value = instance
        yield instance


@pytest.fixture
def generator(mock_client):
    return AIGenerator(api_key="test-key", model="claude-test")


@pytest.fixture
def mock_tool_manager():
    tm = MagicMock()
    tm.execute_tool.return_value = "Tool result content"
    return tm


# ---------------------------------------------------------------------------
# Tool choice forwarding
# ---------------------------------------------------------------------------

def test_tool_choice_auto_set_when_tools_provided(mock_client, generator, mock_tool_manager):
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")

    generator.generate_response(
        query="What is MCP?",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "auto"}


def test_no_tool_choice_when_no_tools_provided(mock_client, generator):
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")

    generator.generate_response(query="What is Python?")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "tool_choice" not in call_kwargs


def test_tools_included_in_api_call_when_provided(mock_client, generator, mock_tool_manager):
    tools = [{"name": "search_course_content"}]
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")

    generator.generate_response(query="x", tools=tools, tool_manager=mock_tool_manager)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == tools


# ---------------------------------------------------------------------------
# Direct (non-tool) response path
# ---------------------------------------------------------------------------

def test_direct_text_response_returned_on_end_turn(mock_client, generator):
    mock_client.messages.create.return_value = make_response(
        stop_reason="end_turn",
        content=[make_text_block("Direct answer")],
    )

    result = generator.generate_response(query="What is Python?")

    assert result == "Direct answer"


def test_only_one_api_call_on_end_turn(mock_client, generator):
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")

    generator.generate_response(query="What is Python?")

    assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Single tool round — natural termination
# ---------------------------------------------------------------------------

def test_tool_executed_on_tool_use_stop(mock_client, generator, mock_tool_manager):
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Final answer")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="Tell me about MCP",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_tool_manager.execute_tool.called
    assert result == "Final answer"


def test_execute_tool_called_with_correct_name_and_kwargs(mock_client, generator, mock_tool_manager):
    tool_block = make_tool_use_block(name="search_course_content", input_data={"query": "MCP servers"})
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="MCP servers")


def test_single_round_natural_stop_makes_exactly_two_api_calls(mock_client, generator, mock_tool_manager):
    """Round 1 tool_use → round 2 end_turn (natural stop): exactly 2 calls total."""
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_client.messages.create.call_count == 2


def test_single_round_natural_stop_does_not_make_extra_synthesis_call(mock_client, generator, mock_tool_manager):
    """
    When Claude returns end_turn naturally after round 1, no extra synthesis call is made.
    Both calls include tools because natural termination bypasses the else/synthesis branch.
    """
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Answer")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_client.messages.create.call_count == 2
    assert result == "Answer"


def test_round_two_api_call_includes_tools(mock_client, generator, mock_tool_manager):
    """
    In the natural-stop 2-call path, call2 (round 2) still includes tools because
    it is an intermediate loop iteration, not the forced synthesis call.
    """
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    assert "tools" in second_call_kwargs
    assert "tool_choice" in second_call_kwargs


def test_single_round_messages_include_assistant_and_tool_result(mock_client, generator, mock_tool_manager):
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in second_call_messages]
    # [user (original), assistant (tool call r1), user (tool result r1)]
    assert roles == ["user", "assistant", "user"]


def test_tool_result_content_includes_execute_output(mock_client, generator, mock_tool_manager):
    mock_tool_manager.execute_tool.return_value = "Search returned: lesson content"
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [first_response, second_response]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_message = second_call_messages[2]
    assert tool_result_message["role"] == "user"
    assert any(
        tr.get("content") == "Search returned: lesson content"
        for tr in tool_result_message["content"]
    )


# ---------------------------------------------------------------------------
# Two sequential tool rounds
# ---------------------------------------------------------------------------

def test_two_sequential_tool_rounds_make_three_api_calls(mock_client, generator, mock_tool_manager):
    """Round 1 tool_use → round 2 tool_use → round 3 end_turn (natural stop): 3 calls total."""
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    r3 = make_response(stop_reason="end_turn", content=[make_text_block("Final")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    result = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_client.messages.create.call_count == 3
    assert result == "Final"


def test_intermediate_round_two_api_call_includes_tools(mock_client, generator, mock_tool_manager):
    """In the 3-call two-round scenario, call2 (round 2) must still include tools."""
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    r3 = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    assert "tools" in second_call_kwargs
    assert "tool_choice" in second_call_kwargs


def test_execute_tool_called_twice_across_two_rounds(mock_client, generator, mock_tool_manager):
    r1 = make_response(stop_reason="tool_use", content=[
        make_tool_use_block(name="get_course_outline", input_data={"course_name": "MCP"}, block_id="r1")
    ])
    r2 = make_response(stop_reason="tool_use", content=[
        make_tool_use_block(name="search_course_content", input_data={"query": "context window"}, block_id="r2")
    ])
    r3 = make_response(stop_reason="end_turn", content=[make_text_block("Answer")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    generator.generate_response(
        query="x",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_tool_manager.execute_tool.call_count == 2
    calls = mock_tool_manager.execute_tool.call_args_list
    assert calls[0] == call("get_course_outline", course_name="MCP")
    assert calls[1] == call("search_course_content", query="context window")


def test_messages_after_two_rounds_have_correct_roles(mock_client, generator, mock_tool_manager):
    """After 2 tool rounds, the third call's messages list has 5 alternating entries."""
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    r3 = make_response(stop_reason="end_turn", content=[make_text_block("Done")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    third_call_messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
    roles = [m["role"] for m in third_call_messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# MAX_ROUNDS cap — forced synthesis call
# ---------------------------------------------------------------------------

def test_max_rounds_forces_synthesis_call_without_tools(mock_client, generator, mock_tool_manager):
    """
    When 2 tool rounds complete without natural stop, a forced synthesis call
    (3rd call, no tools) is made.
    """
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    synthesis = make_response(stop_reason="end_turn", content=[make_text_block("Synthesis")])
    mock_client.messages.create.side_effect = [r1, r2, synthesis]

    result = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert mock_client.messages.create.call_count == 3
    synthesis_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
    assert "tools" not in synthesis_call_kwargs
    assert "tool_choice" not in synthesis_call_kwargs
    assert result == "Synthesis"


def test_synthesis_call_after_max_rounds_raises_value_error_on_empty_content(mock_client, generator, mock_tool_manager):
    """Synthesis call (after max rounds) uses _extract_text — raises ValueError on empty content."""
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    synthesis = make_response(stop_reason="end_turn", content=[])
    mock_client.messages.create.side_effect = [r1, r2, synthesis]

    with pytest.raises(ValueError, match="No text block"):
        generator.generate_response(
            query="x",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )


# ---------------------------------------------------------------------------
# Tool execution error handling
# ---------------------------------------------------------------------------

def test_tool_execution_exception_serialized_as_tool_result(mock_client, generator, mock_tool_manager):
    """
    When execute_tool() raises, the exception is caught and serialized as an error
    string in the tool_result content. The loop continues and Claude receives context
    about the failure.
    """
    mock_tool_manager.execute_tool.side_effect = Exception("DB down")
    tool_block = make_tool_use_block()
    r1 = make_response(stop_reason="tool_use", content=[tool_block])
    r2 = make_response(stop_reason="end_turn", content=[make_text_block("Handled")])
    mock_client.messages.create.side_effect = [r1, r2]

    result = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    # Two API calls made — loop did not abort
    assert mock_client.messages.create.call_count == 2

    # Tool result in round-2 call's messages contains the error string
    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_message = second_call_messages[2]
    assert any(
        "Tool execution error" in tr.get("content", "")
        for tr in tool_result_message["content"]
    )
    assert result == "Handled"


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def test_conversation_history_appended_to_system_prompt(mock_client, generator):
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")
    history = "User: hi\nAssistant: hello"

    generator.generate_response(query="What is RAG?", conversation_history=history)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert history in call_kwargs["system"]


def test_no_history_uses_base_system_prompt(mock_client, generator):
    mock_client.messages.create.return_value = make_response(stop_reason="end_turn")

    generator.generate_response(query="What is RAG?")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "Previous conversation" not in call_kwargs["system"]


# ---------------------------------------------------------------------------
# API exception handling
# ---------------------------------------------------------------------------

def test_api_exception_wrapped_as_runtime_error(mock_client, generator):
    mock_client.messages.create.side_effect = Exception("Authentication failed: invalid API key")

    with pytest.raises(RuntimeError, match="AI service unavailable"):
        generator.generate_response(query="What is MCP?")


def test_api_exception_in_intermediate_round_wrapped_as_runtime_error(mock_client, generator, mock_tool_manager):
    """API failure during an intermediate tool round is wrapped as RuntimeError."""
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    mock_client.messages.create.side_effect = [
        first_response,
        Exception("Rate limit exceeded"),
    ]

    with pytest.raises(RuntimeError, match="AI service unavailable"):
        generator.generate_response(
            query="x",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )


def test_api_exception_in_synthesis_call_wrapped_as_runtime_error(mock_client, generator, mock_tool_manager):
    """API failure during the forced synthesis call (after MAX_ROUNDS) is also wrapped."""
    r1 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r1")])
    r2 = make_response(stop_reason="tool_use", content=[make_tool_use_block(block_id="r2")])
    mock_client.messages.create.side_effect = [r1, r2, Exception("Timeout")]

    with pytest.raises(RuntimeError, match="AI service unavailable"):
        generator.generate_response(
            query="x",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )


# ---------------------------------------------------------------------------
# Safe content extraction via _extract_text()
# ---------------------------------------------------------------------------

def test_direct_response_raises_value_error_on_empty_content(mock_client, generator):
    mock_client.messages.create.return_value = make_response(
        stop_reason="end_turn", content=[]
    )

    with pytest.raises(ValueError, match="No text block"):
        generator.generate_response(query="What is Python?")


def test_natural_stop_response_raises_value_error_on_empty_content(mock_client, generator, mock_tool_manager):
    """Natural-stop response after a tool round uses _extract_text — raises ValueError on empty content."""
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])
    second_response = make_response(stop_reason="end_turn", content=[])
    mock_client.messages.create.side_effect = [first_response, second_response]

    with pytest.raises(ValueError, match="No text block"):
        generator.generate_response(
            query="x",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )


def test_extract_text_skips_non_text_blocks_to_find_text(mock_client, generator, mock_tool_manager):
    """_extract_text() iterates all blocks — finds a text block even if not at index 0."""
    tool_block = make_tool_use_block()
    first_response = make_response(stop_reason="tool_use", content=[tool_block])

    non_text_block = MagicMock()
    non_text_block.type = "tool_use"
    text_block = make_text_block("Found it")
    second_response = make_response(stop_reason="end_turn", content=[non_text_block, text_block])

    mock_client.messages.create.side_effect = [first_response, second_response]

    result = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Found it"
