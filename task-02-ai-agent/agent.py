"""ToolCallingAgent module implementing autonomous multi-turn reasoning and dynamic tool execution with Groq."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from groq import Groq

from tools import TOOLS_MAP, get_openai_tool_schemas

AGENT_SYSTEM_PROMPT = """You are an intelligent, capable, and helpful Multi-Turn AI Assistant equipped with dynamic external tools.

### YOUR ACTIVE TOOLS:
1. `get_weather`: Query real-time weather and forecast data for any city or global location.
2. `calculator`: Safely compute mathematical operations, arithmetic, powers, and roots.
3. `wikipedia_search`: Query Wikipedia for factual overviews, scientific concepts, historical context, or encyclopedic knowledge.

### OPERATIONAL DIRECTIVES & REASONING RULES:
1. SMART TOOL ROUTING:
   - DO NOT invoke tools for simple conversational greetings, gratitude, or general questions (e.g., "Hello", "How are you?", "Tell me a joke"). Reply directly and warmly.
   - ONLY invoke a tool when real-time, external, exact, or calculated data is needed.
   - If a user asks for live weather, exact arithmetic, or encyclopedia summaries, invoke the appropriate tool.
2. SEQUENTIAL & MULTI-TOOL EXECUTION:
   - If a user query requires multiple tools (e.g., "What is the weather in Paris, and calculate 45 * 12?"), invoke all necessary tools to fulfill the user's intent.
3. SYNTHESIS & PRESENTATION:
   - After receiving tool outputs, synthesize a coherent, engaging, and well-structured response.
   - Never output raw unformatted JSON to the user. Present the findings clearly with markdown formatting, bullet points, or bold highlights.
4. GRACEFUL ERROR RECOVERY:
   - If a tool reports an error (e.g. unknown location or division by zero), explain the issue politely to the user and suggest a clarification rather than crashing.
"""


class ToolCallingAgent:
    """Multi-Turn conversational agent with dynamic tool invocation via Groq Cloud API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        max_iterations: int = 5,
    ):
        """
        Args:
            api_key: Groq Cloud API Key (starts with gsk_...).
            model: Target Groq LLM model identifier.
            max_iterations: Maximum tool-execution iterations per turn to prevent infinite loops.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = AGENT_SYSTEM_PROMPT

        # Initialize conversation memory
        self.messages: List[Dict[str, Any]] = []
        self.clear_memory()

        # Tools configuration
        self.tools_map = dict(TOOLS_MAP)
        self.tool_schemas = get_openai_tool_schemas()

        self._client: Optional[Groq] = None
        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str):
        """Update API key and reinitialize client."""
        self.api_key = api_key.strip()
        self._init_client()

    def _init_client(self):
        """Instantiate Groq client."""
        if not self.api_key:
            raise ValueError("Groq API key is required to initialize ToolCallingAgent.")
        self._client = Groq(api_key=self.api_key)

    def _ensure_client(self):
        """Ensure Groq client is active before API calls."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Groq API Key missing. Please provide it in the sidebar or set GROQ_API_KEY in .env")
            self._init_client()

    def clear_memory(self):
        """Reset conversation memory back to initial system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute one complete multi-turn conversational turn with agent reasoning loop.

        Args:
            user_prompt: Natural language user prompt.

        Returns:
            Tuple of:
                - Synthesized assistant response string
                - List of tool execution trace records
        """
        self._ensure_client()
        self.messages.append({"role": "user", "content": user_prompt})

        tool_executions_trace: List[Dict[str, Any]] = []
        iteration = 0
        final_answer = ""

        while iteration < self.max_iterations:
            iteration += 1

            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as e:
                # If specific model is not found on current API key, attempt graceful fallback
                err_str = str(e)
                if "model_not_found" in err_str or "404" in err_str:
                    fallback_model = "qwen/qwen3.8-27b"
                    if self.model != fallback_model:
                        self.model = fallback_model
                        return self.chat(user_prompt)

                err_msg = f"I encountered an error communicating with the Groq API: {e}"
                self.messages.append({"role": "assistant", "content": err_msg})
                return err_msg, tool_executions_trace

            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls

            # Case 1: Model replies directly without calling tools
            if not tool_calls:
                final_answer = message.content or ""
                self.messages.append({"role": "assistant", "content": final_answer})
                break

            # Case 2: Model requested tool calls
            # Append assistant message with tool calls to conversation history
            assistant_msg_dict = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            self.messages.append(assistant_msg_dict)

            # Execute requested tools
            for tc in tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments

                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {"raw_input": raw_args}

                t0 = time.perf_counter()
                tool_fn = self.tools_map.get(tool_name)

                if tool_fn:
                    try:
                        tool_output = tool_fn(**parsed_args)
                        success = True
                        error_msg = None
                    except Exception as e:
                        tool_output = {"error": str(e)}
                        success = False
                        error_msg = str(e)
                else:
                    tool_output = {"error": f"Tool '{tool_name}' not found."}
                    success = False
                    error_msg = f"Tool '{tool_name}' not found."

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Record execution trace
                tool_executions_trace.append(
                    {
                        "tool_name": tool_name,
                        "arguments": parsed_args,
                        "output": tool_output,
                        "success": success,
                        "error": error_msg,
                        "execution_time_ms": round(elapsed_ms, 2),
                    }
                )

                # Append tool response message according to OpenAI/Groq specification
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_output),
                    }
                )

        # Fallback if max iterations reached without final text
        if not final_answer:
            try:
                synth = self._client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.2,
                )
                final_answer = synth.choices[0].message.content or "Completed tool actions."
            except Exception:
                final_answer = f"Completed actions across {len(tool_executions_trace)} tool execution(s)."
            self.messages.append({"role": "assistant", "content": final_answer})

        return final_answer, tool_executions_trace
