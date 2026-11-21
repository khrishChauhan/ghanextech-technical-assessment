"""Unit tests for AST-based safe calculator tool."""

import pytest
from tools.calculator import calculate, TOOL_DEFINITION


def test_calculator_tool_definition():
    """Verify standard tool definition schema."""
    assert TOOL_DEFINITION["name"] == "calculator"
    assert "expression" in TOOL_DEFINITION["parameters"]["properties"]
    assert "expression" in TOOL_DEFINITION["parameters"]["required"]
    assert callable(TOOL_DEFINITION["function"])


def test_basic_arithmetic():
    """Verify arithmetic operations (+, -, *, /, //, %, **, ^)."""
    assert calculate("2 + 2")["result"] == 4
    assert calculate("100 - 45")["result"] == 55
    assert calculate("12 * 12")["result"] == 144
    assert calculate("840 / 20")["result"] == 42
    assert calculate("17 // 3")["result"] == 5
    assert calculate("17 % 5")["result"] == 2
    assert calculate("2 ** 8")["result"] == 256
    assert calculate("2 ^ 8")["result"] == 256


def test_complex_precedence_and_parentheses():
    """Verify operator precedence and grouping parentheses."""
    res = calculate("(10 + 5) * (4 - 2) / 3")
    assert res["result"] == 10

    res2 = calculate("(450 * 12) / 3.5")
    assert round(res2["result"], 2) == 1542.86


def test_division_by_zero():
    """Verify division and modulo by zero are safely intercepted."""
    with pytest.raises(ValueError, match="Division by zero"):
        calculate("50 / 0")

    with pytest.raises(ValueError, match="Division by zero"):
        calculate("10 // 0")

    with pytest.raises(ValueError, match="Division by zero"):
        calculate("15 % 0")


def test_security_prohibits_arbitrary_code():
    """Verify strict prohibition of code execution, imports, and variables."""
    # Reject imports
    with pytest.raises(ValueError):
        calculate("__import__('os').system('ls')")

    # Reject builtins
    with pytest.raises(ValueError):
        calculate("eval('2 + 2')")

    # Reject function calls
    with pytest.raises(ValueError):
        calculate("print('hello')")

    # Reject assignments
    with pytest.raises(ValueError):
        calculate("x = 5")

    # Reject variable names
    with pytest.raises(ValueError):
        calculate("secret_variable * 2")


def test_empty_expression():
    """Verify empty or blank expressions raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        calculate("")
