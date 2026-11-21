"""Safe Mathematical Calculator using strict Python Abstract Syntax Tree (AST) node visitation."""

import ast
import operator
from typing import Any, Dict

# Allowed binary arithmetic operations
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operations
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_ast_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node strictly restricted to safe arithmetic."""
    # Numbers / Constants (int, float)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Prohibited constant type: {type(node.value).__name__}")

    # Binary Operators (e.g. x + y, a * b)
    elif isinstance(node, ast.BinOp):
        left = evaluate_ast_node(node.left)
        right = evaluate_ast_node(node.right)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Prohibited operator: {op_type.__name__}")

        # Guard against division by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero is undefined.")

        # Guard against exponential resource exhaustion
        if op_type is ast.Pow:
            if abs(left) > 1000 and right > 1000:
                raise ValueError("Potential numerical overflow in exponentiation.")
            if right > 10000:
                raise ValueError("Exponent exceeds maximum safe threshold (10000).")

        return ALLOWED_OPERATORS[op_type](left, right)

    # Unary Operators (e.g. -x, +x)
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_ast_node(node.operand)
        op_type = type(node.op)

        if op_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError(f"Prohibited unary operator: {op_type.__name__}")

        return ALLOWED_UNARY_OPERATORS[op_type](operand)

    # Any other AST node (Call, Import, Attribute, Name, Assign, etc.) is strictly forbidden
    else:
        raise ValueError(f"Prohibited or unsafe syntax detected: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """Safely compute a mathematical expression using AST parsing.

    Args:
        expression: String mathematical expression, e.g. '144 * 12 + 25' or '100 / 4'.

    Returns:
        Structured dict with keys: {expression, result}.

    Raises:
        ValueError: For syntax errors, unsafe syntax, or division by zero.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    # Support ^ as exponentiation alias
    cleaned = expression.strip().replace("^", "**")

    try:
        parsed_tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid mathematical expression syntax: {e}")

    result = evaluate_ast_node(parsed_tree.body)

    # Format integers cleanly
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return {
        "expression": expression,
        "result": result,
    }


# Standardized tool definition dictionary
TOOL_DEFINITION: Dict[str, Any] = {
    "name": "calculator",
    "description": (
        "Perform exact mathematical computations safely. "
        "Supports addition (+), subtraction (-), multiplication (*), division (/), "
        "integer division (//), modulo (%), and powers (** or ^)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '(450 * 12) / 3.5' or '2 ** 10'",
            }
        },
        "required": ["expression"],
    },
    "function": calculate,
}
