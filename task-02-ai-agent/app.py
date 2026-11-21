"""Streamlit Web Application for Ghanex Multi-Turn AI Agent with Dynamic Tool Calling."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Add task directory to python path for modular imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import ToolCallingAgent
from tools import AVAILABLE_TOOLS

# Page configuration
st.set_page_config(
    page_title="Ghanex AI Agent with Dynamic Tool Calling",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .agent-header {
        background: linear-gradient(135deg, #111827 0%, #1E3A8A 50%, #4338CA 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .agent-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .agent-header p {
        margin-top: 6px;
        margin-bottom: 0;
        color: #E2E8F0;
        font-size: 1.05rem;
    }
    .status-badge-active {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-inactive {
        background-color: #FEECDC;
        color: #9C4221;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tool-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .tool-trace-card {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .tool-trace-title {
        font-weight: 700;
        color: #312E81;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_api_key() -> str:
    """Retrieve Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            if hasattr(st, "secrets"):
                key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    return key or ""


# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = get_default_api_key()

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

if "agent" not in st.session_state:
    st.session_state.agent = ToolCallingAgent(
        api_key=st.session_state.api_key,
        model=st.session_state.selected_model,
    )

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🤖 Agent Control Center")

    # API Key Configuration
    has_key = bool(st.session_state.api_key)
    if has_key:
        st.markdown(
            '<span class="status-badge-active">● Groq API Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge-inactive">○ Groq API Key Missing</span>',
            unsafe_allow_html=True,
        )

    user_key = st.text_input(
        "Groq Cloud API Key (`gsk_...`)",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter gsk_...",
        help="Reads automatically from .env or Streamlit secrets.",
    )
    if user_key != st.session_state.api_key:
        st.session_state.api_key = user_key.strip()
        st.session_state.agent.set_api_key(user_key.strip())
        st.rerun()

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 Inference Model")
    model_choices = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
    ]
    model_idx = (
        model_choices.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_choices
        else 0
    )
    chosen_model = st.selectbox(
        "Select Model",
        options=model_choices,
        index=model_idx,
        help="Target Groq LLM model for tool calling and reasoning.",
    )
    if chosen_model != st.session_state.selected_model:
        st.session_state.selected_model = chosen_model
        st.session_state.agent.model = chosen_model

    st.markdown("---")

    # Tool Catalog
    st.markdown("### 🛠️ Active Tools Catalog")
    for t in AVAILABLE_TOOLS:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**`{t['name']}`**")
        with col2:
            st.markdown('<span class="status-badge-active">Active</span>', unsafe_allow_html=True)
        st.caption(t["description"])
        with st.expander(f"Parameters Schema: `{t['name']}`", expanded=False):
            st.json(t["parameters"], expanded=False)
        st.markdown("")

    st.markdown("---")

    # Reset Chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.agent.clear_memory()
        st.session_state.chat_messages = []
        st.success("Conversation history cleared.")
        st.rerun()


# ---------------- MAIN CONTENT ----------------
st.markdown(
    """
    <div class="agent-header">
        <h1>🤖 Ghanex Multi-Turn AI Agent</h1>
        <p>Autonomous Reasoning & Dynamic Tool Orchestration powered by Groq Function Calling</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested Prompts (Pills)
if len(st.session_state.chat_messages) == 0:
    st.markdown("##### 💡 Quick-Prompt Suggestions:")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🌤️ What is the weather in Tokyo?", use_container_width=True):
            st.session_state.pending_prompt = "What is the weather in Tokyo?"
            st.rerun()
        if st.button("🧮 Calculate (450 * 12) / 3.5", use_container_width=True):
            st.session_state.pending_prompt = "Calculate (450 * 12) / 3.5"
            st.rerun()
    with colB:
        if st.button("📚 Who was Alan Turing?", use_container_width=True):
            st.session_state.pending_prompt = "Who was Alan Turing?"
            st.rerun()
        if st.button("💬 Hello! What can you do?", use_container_width=True):
            st.session_state.pending_prompt = "Hello! What can you do?"
            st.rerun()

# Display chat history
for msg in st.session_state.chat_messages:
    avatar_icon = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if msg.get("tool_executions"):
            with st.expander("🛠️ Tool Executions", expanded=False):
                for idx, t_rec in enumerate(msg["tool_executions"], start=1):
                    tool_name = t_rec["tool_name"]
                    args = t_rec["arguments"]
                    success = t_rec["success"]
                    output = t_rec["output"]
                    duration = t_rec.get("execution_time_ms", 0.0)
                    status_icon = "✅" if success else "❌"

                    st.markdown(
                        f"""
                        <div class="tool-trace-card">
                            <div class="tool-trace-title">{status_icon} Execution {idx}: <code>{tool_name}</code> ({duration:.1f}ms)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Input Arguments (JSON):**")
                    st.json(args, expanded=False)
                    st.markdown("**Raw Tool Output:**")
                    st.json(output if isinstance(output, dict) else {"output": output}, expanded=False)

# Check for pending prompt or chat input
user_query = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    chat_input = st.chat_input("Ask a question, request a calculation, check weather, or search Wikipedia...")
    if chat_input:
        user_query = chat_input

# Execute turn
if user_query:
    # Append to UI history
    st.session_state.chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        if not st.session_state.api_key:
            err_text = "Please enter your Groq API Key in the sidebar or configure .env to activate the agent."
            st.error(err_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": err_text})
        else:
            with st.spinner("Agent is reasoning and orchestrating tools..."):
                response_text, tool_traces = st.session_state.agent.chat(user_query)

            # Display final synthesized text
            st.markdown(response_text)

            # Display tool trace expander if tools were invoked
            if tool_traces:
                with st.expander("🛠️ Tool Executions", expanded=False):
                    for idx, t_rec in enumerate(tool_traces, start=1):
                        tool_name = t_rec["tool_name"]
                        args = t_rec["arguments"]
                        success = t_rec["success"]
                        output = t_rec["output"]
                        duration = t_rec.get("execution_time_ms", 0.0)
                        status_icon = "✅" if success else "❌"

                        st.markdown(
                            f"""
                            <div class="tool-trace-card">
                                <div class="tool-trace-title">{status_icon} Execution {idx}: <code>{tool_name}</code> ({duration:.1f}ms)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Input Arguments (JSON):**")
                        st.json(args, expanded=False)
                        st.markdown("**Raw Tool Output:**")
                        st.json(output if isinstance(output, dict) else {"output": output}, expanded=False)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "tool_executions": tool_traces,
                }
            )
