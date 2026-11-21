"""Unit tests for weather tool."""

import pytest
from tools.weather import get_weather, TOOL_DEFINITION


def test_weather_tool_definition():
    assert TOOL_DEFINITION["name"] == "get_weather"
    assert "city" in TOOL_DEFINITION["parameters"]["properties"]
    assert "city" in TOOL_DEFINITION["parameters"]["required"]
    assert callable(TOOL_DEFINITION["function"])


def test_get_weather_valid_city():
    result = get_weather("Tokyo")
    assert result["city"] == "Tokyo"
    assert "country" in result
    assert "temperature_c" in result
    assert "windspeed" in result
    assert "weather_condition" in result
    assert isinstance(result["temperature_c"], (int, float))


def test_get_weather_empty_city():
    with pytest.raises(ValueError, match="cannot be empty"):
        get_weather("")


def test_get_weather_unknown_city():
    with pytest.raises(ValueError, match="Could not find coordinates"):
        get_weather("NonExistentCityNameXYZ987654321")
