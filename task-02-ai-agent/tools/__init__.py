"""Tools package exporting standardized tool definitions and mapping."""

from typing import Any, Callable, Dict, List
from tools.weather import TOOL_DEFINITION as WEATHER_TOOL, get_weather
from tools.calculator import TOOL_DEFINITION as CALCULATOR_TOOL, calculate
from tools.search import TOOL_DEFINITION as SEARCH_TOOL, search_wikipedia

# List of all available tool definitions for LLM registration
AVAILABLE_TOOLS: List[Dict[str, Any]] = [
    WEATHER_TOOL,
    CALCULATOR_TOOL,
    SEARCH_TOOL,
]

# Mapping tool names to executable functions
TOOLS_MAP: Dict[str, Callable] = {
    WEATHER_TOOL["name"]: WEATHER_TOOL["function"],
    CALCULATOR_TOOL["name"]: CALCULATOR_TOOL["function"],
    SEARCH_TOOL["name"]: SEARCH_TOOL["function"],
}


def get_openai_tool_schemas() -> List[Dict[str, Any]]:
    """Convert registered tool definitions into OpenAI/Groq function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in AVAILABLE_TOOLS
    ]


__all__ = [
    "WEATHER_TOOL",
    "CALCULATOR_TOOL",
    "SEARCH_TOOL",
    "AVAILABLE_TOOLS",
    "TOOLS_MAP",
    "get_openai_tool_schemas",
    "get_weather",
    "calculate",
    "search_wikipedia",
]
