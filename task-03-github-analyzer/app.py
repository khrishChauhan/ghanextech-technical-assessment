"""Streamlit Web Application for GitHub Repository AI Analyzer."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Add directory to sys.path for local imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from github_client import (
    GitHubClient,
    GitHubRateLimitError,
    InvalidRepositoryURLError,
    RepositoryNotFoundError,
)
from analyzer_engine import RepoAnalyzer

# Page configuration
st.set_page_config(
    page_title="GitHub Repository AI Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .main-header p {
        margin-top: 6px;
        margin-bottom: 0;
        color: #E2E8F0;
        font-size: 1.05rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .topic-pill {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
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
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_key(name: str) -> str:
    """Retrieve key from environment or Streamlit secrets safely."""
    val = os.getenv(name)
    if not val:
        try:
            if hasattr(st, "secrets"):
                val = st.secrets.get(name)
        except Exception:
            pass
    return val or ""


# Initialize session state
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = get_default_key("GROQ_API_KEY")

if "github_token" not in st.session_state:
    st.session_state.github_token = get_default_key("GITHUB_TOKEN")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "input_repo_val" not in st.session_state:
    st.session_state.input_repo_val = ""


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration & Keys")

    # Groq API Key Input
    has_groq = bool(st.session_state.groq_api_key)
    if has_groq:
        st.markdown('<span class="status-badge-active">● Groq Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-inactive">○ Groq Key Missing</span>', unsafe_allow_html=True)

    groq_input = st.text_input(
        "Groq Cloud API Key (`gsk_...`)",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="Enter gsk_...",
        help="Reads from .env or Streamlit secrets automatically.",
    )
    if groq_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = groq_input.strip()
        st.rerun()

    # GitHub Token Input
    has_github = bool(st.session_state.github_token)
    if has_github:
        st.markdown('<span class="status-badge-active">● GitHub Token Set (5,000 req/hr)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-inactive">○ GitHub Unauthenticated (60 req/hr)</span>', unsafe_allow_html=True)

    github_input = st.text_input(
        "GitHub Personal Access Token (Optional)",
        value=st.session_state.github_token,
        type="password",
        placeholder="Enter ghp_... or github_pat_...",
        help="Optional: Increases GitHub REST API rate limit from 60 req/hr to 5,000 req/hr. No permissions required for public repos.",
    )
    if github_input != st.session_state.github_token:
        st.session_state.github_token = github_input.strip()
        st.rerun()

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 LLM Inference Model")
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
    ]
    model_idx = (
        model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_options
        else 0
    )
    chosen_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=model_idx,
        help="Target Groq model for technical architectural synthesis.",
    )
    if chosen_model != st.session_state.selected_model:
        st.session_state.selected_model = chosen_model

    st.markdown("---")

    # Quick Demo Repositories
    st.markdown("### 🚀 Quick-Load Demo Repos")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if st.button("Flask", use_container_width=True):
            st.session_state.input_repo_val = "pallets/flask"
            st.session_state.trigger_analysis = True
            st.rerun()
    with col_d2:
        if st.button("FastAPI", use_container_width=True):
            st.session_state.input_repo_val = "tiangolo/fastapi"
            st.session_state.trigger_analysis = True
            st.rerun()
    with col_d3:
        if st.button("Requests", use_container_width=True):
            st.session_state.input_repo_val = "psf/requests"
            st.session_state.trigger_analysis = True
            st.rerun()


# ---------------- MAIN CONTENT ----------------
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 GitHub Repository AI Analyzer</h1>
        <p>Autonomous Codebase Ingestion, Architectural Synthesis & Open-Source Contribution Roadmap powered by Groq</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Repository Input Section
col_in, col_btn = st.columns([5, 1])
with col_in:
    target_repo = st.text_input(
        "GitHub Repository URL or shorthand",
        value=st.session_state.input_repo_val,
        placeholder="e.g. psf/requests, pallets/flask, or https://github.com/tiangolo/fastapi",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_clicked = st.button("🚀 Analyze", use_container_width=True, type="primary")

trigger = analyze_clicked or st.session_state.pop("trigger_analysis", False)

# Run Analysis Pipeline
if trigger:
    if not target_repo.strip():
        st.warning("Please enter a valid GitHub repository URL or 'owner/repo' identifier.")
    elif not st.session_state.groq_api_key:
        st.error("Groq Cloud API Key missing. Please provide it in the sidebar or set GROQ_API_KEY in .env.")
    else:
        st.session_state.input_repo_val = target_repo.strip()

        github_client = GitHubClient(token=st.session_state.github_token)
        analyzer = RepoAnalyzer(
            api_key=st.session_state.groq_api_key,
            model=st.session_state.selected_model,
        )

        try:
            with st.spinner("📥 Step 1/2: Fetching repository metadata, README, and file hierarchy via GitHub REST API..."):
                metadata, content = github_client.fetch_full_repository(target_repo.strip())

            with st.spinner(f"🧠 Step 2/2: Synthesizing architectural audit & contribution roadmap with Groq ({st.session_state.selected_model})..."):
                report = analyzer.analyze_repository(metadata, content)

            st.session_state.analysis_data = {
                "metadata": metadata,
                "content": content,
                "report": report,
            }
            st.success(f"Analysis complete for **{metadata.full_name}**!")
        except InvalidRepositoryURLError as e:
            st.error(f"❌ Invalid Repository URL: {e}")
        except RepositoryNotFoundError as e:
            st.error(f"❌ Repository Not Found: {e}")
        except GitHubRateLimitError as e:
            st.warning(f"⚠️ Rate Limit Intercepted: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error during analysis: {e}")


# Render Analysis Results
if st.session_state.analysis_data:
    meta = st.session_state.analysis_data["metadata"]
    content = st.session_state.analysis_data["content"]
    report = st.session_state.analysis_data["report"]

    st.markdown("---")

    # Header section with link and description
    st.markdown(f"## 📦 [{meta.full_name}]({meta.html_url})")
    st.markdown(f"*{meta.description}*")

    if meta.topics:
        topics_html = "".join(f'<span class="topic-pill">#{t}</span>' for t in meta.topics)
        st.markdown(topics_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric Cards Grid
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Stars</div><div class="metric-value">⭐ {meta.stars:,}</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Forks</div><div class="metric-value">🍴 {meta.forks:,}</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Open Issues</div><div class="metric-value">🐛 {meta.open_issues:,}</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Watchers</div><div class="metric-value">👁️ {meta.watchers:,}</div></div>""",
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Language</div><div class="metric-value">💻 {meta.primary_language}</div></div>""",
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">License</div><div class="metric-value">📜 {meta.license[:12]}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🤖 AI Technical Analysis",
        "📁 Codebase Tree & Manifests",
        "📖 Raw README",
    ])

    with tab1:
        # Download / Export Button
        download_report = f"# Technical Analysis: {meta.full_name}\n\n"
        download_report += f"- **GitHub URL:** {meta.html_url}\n"
        download_report += f"- **Stars:** {meta.stars:,} | **Forks:** {meta.forks:,} | **Language:** {meta.primary_language}\n"
        download_report += f"- **License:** {meta.license}\n\n---\n\n"
        download_report += report

        clean_slug = meta.full_name.replace("/", "_")
        st.download_button(
            label="📥 Download Complete AI Analysis Report (.md)",
            data=download_report,
            file_name=f"{clean_slug}_ai_technical_analysis.md",
            mime="text/markdown",
            type="secondary",
        )

        st.markdown("---")
        st.markdown(report)

    with tab2:
        st.markdown("### 🌳 Repository Structure")
        st.caption(f"Default branch: `{meta.default_branch}` | Total repository items: `{content.total_files:,}`")

        with st.expander(f"View File Hierarchy ({len(content.tree)} items listed)", expanded=True):
            st.code("\n".join(content.tree), language="text")

        st.markdown("### 📋 Detected Package Manifests & Configs")
        if content.manifests:
            for manifest_name, manifest_body in content.manifests.items():
                with st.expander(f"📄 `{manifest_name}`", expanded=False):
                    st.code(manifest_body, language="toml" if "toml" in manifest_name else "text")
        else:
            st.info("No standard package manifests (e.g. pyproject.toml, package.json) were found in the root.")

    with tab3:
        st.markdown("### 📄 Original README.md")
        if content.readme and content.readme.strip():
            st.markdown(content.readme)
        else:
            st.info("No README file available for this repository.")
