"""Unit tests for ToolCallingAgent and multi-turn execution loop."""

import json
from unittest.mock import MagicMock, patch
import pytest
from agent import ToolCallingAgent, AGENT_SYSTEM_PROMPT


def test_agent_initialization():
    """Verify ToolCallingAgent initializes with defaults and schemas."""
    agent = ToolCallingAgent(api_key="gsk_test123", model="llama-3.3-70b-versatile")
    assert agent.model == "llama-3.3-70b-versatile"
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert "calculator" in agent.tools_map
    assert "get_weather" in agent.tools_map
    assert "wikipedia_search" in agent.tools_map
    assert len(agent.tool_schemas) == 3


def test_agent_clear_memory():
    """Verify clear_memory restores original system prompt."""
    agent = ToolCallingAgent(api_key="gsk_test123")
    agent.messages.append({"role": "user", "content": "Hello"})
    agent.messages.append({"role": "assistant", "content": "Hi"})
    assert len(agent.messages) == 3

    agent.clear_memory()
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == AGENT_SYSTEM_PROMPT


def test_agent_missing_key_raises():
    """Verify agent without key raises ValueError on chat execution."""
    agent = ToolCallingAgent(api_key="")
    with pytest.raises(ValueError, match="Groq API Key missing"):
        agent.chat("Hello")


def test_agent_direct_response_loop():
    """Verify agent returns direct response when no tools are requested by model."""
    agent = ToolCallingAgent(api_key="gsk_test123")

    mock_choice = MagicMock()
    mock_choice.message.tool_calls = None
    mock_choice.message.content = "Hello there! How can I assist you today?"

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch.object(agent._client.chat.completions, "create", return_value=mock_completion):
        response_text, traces = agent.chat("Hi")

        assert response_text == "Hello there! How can I assist you today?"
        assert traces == []
        assert len(agent.messages) == 3
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[2]["role"] == "assistant"


def test_agent_tool_calling_loop():
    """Verify agent executes requested tool and completes final answer synthesis."""
    agent = ToolCallingAgent(api_key="gsk_test123")

    # Step 1: Model requests calculator tool
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "calculator"
    mock_tool_call.function.arguments = json.dumps({"expression": "100 + 50"})

    mock_choice_step1 = MagicMock()
    mock_choice_step1.message.content = ""
    mock_choice_step1.message.tool_calls = [mock_tool_call]

    mock_resp_step1 = MagicMock()
    mock_resp_step1.choices = [mock_choice_step1]

    # Step 2: Model synthesizes answer from tool output
    mock_choice_step2 = MagicMock()
    mock_choice_step2.message.tool_calls = None
    mock_choice_step2.message.content = "100 + 50 equals 150."

    mock_resp_step2 = MagicMock()
    mock_resp_step2.choices = [mock_choice_step2]

    with patch.object(
        agent._client.chat.completions,
        "create",
        side_effect=[mock_resp_step1, mock_resp_step2],
    ):
        response_text, traces = agent.chat("What is 100 + 50?")

        assert response_text == "100 + 50 equals 150."
        assert len(traces) == 1
        assert traces[0]["tool_name"] == "calculator"
        assert traces[0]["arguments"] == {"expression": "100 + 50"}
        assert traces[0]["output"]["result"] == 150
        assert traces[0]["success"] is True

        # Messages should have: system, user, assistant(with tool_calls), tool, assistant
        assert len(agent.messages) == 5
        assert agent.messages[3]["role"] == "tool"
        assert agent.messages[3]["tool_call_id"] == "call_abc123"


def test_agent_tool_error_graceful_handling():
    """Verify agent handles tool execution errors without unhandled exceptions."""
    agent = ToolCallingAgent(api_key="gsk_test123")

    # Step 1: Tool call that triggers an error (e.g. division by zero)
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_err123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "calculator"
    mock_tool_call.function.arguments = json.dumps({"expression": "10 / 0"})

    mock_choice_step1 = MagicMock()
    mock_choice_step1.message.content = ""
    mock_choice_step1.message.tool_calls = [mock_tool_call]

    mock_resp_step1 = MagicMock()
    mock_resp_step1.choices = [mock_choice_step1]

    # Step 2: Final response explaining error
    mock_choice_step2 = MagicMock()
    mock_choice_step2.message.tool_calls = None
    mock_choice_step2.message.content = "Cannot divide by zero."

    mock_resp_step2 = MagicMock()
    mock_resp_step2.choices = [mock_choice_step2]

    with patch.object(
        agent._client.chat.completions,
        "create",
        side_effect=[mock_resp_step1, mock_resp_step2],
    ):
        response_text, traces = agent.chat("Calculate 10 / 0")

        assert response_text == "Cannot divide by zero."
        assert len(traces) == 1
        assert traces[0]["success"] is False
        assert "Division by zero" in traces[0]["error"]
