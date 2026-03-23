import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use `search_course_content` for questions about specific course content or detailed educational materials
- Use `get_course_outline` for questions asking about a course's outline, structure, lesson list, or what lessons exist
- **Up to 2 sequential tool calls per query**
- Use a second call only when the first result reveals you need additional information (e.g., retrieve a course outline first, then search specific lesson content)
- Do not make a second call if the first result is already sufficient to answer the question
- When returning a course outline, include: course title, course link, and each lesson's number and title
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls per query.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Phase 1: Build system content and initial API params
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Phase 2: Tool-use loop — each iteration is one API round with tools available
        round_count = 0
        while round_count < self.MAX_TOOL_ROUNDS:
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                raise RuntimeError(f"AI service unavailable: {e}") from e

            # Natural termination: Claude is done calling tools
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Execute all tool_use blocks in this round
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Safety exit: stop_reason was tool_use but no tool blocks found
            if not tool_results:
                return self._extract_text(response)

            # Grow the messages list in place for the next iteration
            api_params["messages"].append({"role": "assistant", "content": response.content})
            api_params["messages"].append({"role": "user", "content": tool_results})
            round_count += 1

        else:
            # Phase 3: MAX_TOOL_ROUNDS exhausted — force a synthesis call without tools
            try:
                response = self.client.messages.create(
                    **self.base_params,
                    messages=api_params["messages"],
                    system=api_params["system"],
                )
            except Exception as e:
                raise RuntimeError(f"AI service unavailable: {e}") from e

        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Safely extract the first text block from a Claude response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        raise ValueError(
            f"No text block found in Claude response (stop_reason={response.stop_reason})"
        )
