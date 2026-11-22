"""Unit tests for RepoAnalyzer and Groq LLM synthesis engine."""

from unittest.mock import MagicMock, patch
import pytest
from github_client import RepoContent, RepoMetadata
from analyzer_engine import RepoAnalyzer


@pytest.fixture
def sample_metadata():
    return RepoMetadata(
        owner="pallets",
        name="flask",
        full_name="pallets/flask",
        description="A Python microframework for web development.",
        stars=68000,
        forks=16000,
        open_issues=22,
        watchers=1900,
        primary_language="Python",
        license="BSD-3-Clause",
        topics=["python", "web", "framework"],
        html_url="https://github.com/pallets/flask",
    )


@pytest.fixture
def sample_content():
    return RepoContent(
        readme="# Flask\nFlask is a micro web framework written in Python.",
        tree=["src/flask/app.py", "tests/test_app.py", "pyproject.toml"],
        total_files=3,
        manifests={"pyproject.toml": '[project]\nname = "flask"'},
    )


def test_analyzer_initialization():
    """Verify RepoAnalyzer default settings."""
    analyzer = RepoAnalyzer(api_key="gsk_mock123")
    assert analyzer.model == "llama-3.3-70b-versatile"
    assert analyzer.temperature == 0.2


def test_build_analysis_prompt(sample_metadata, sample_content):
    """Verify prompt formatting contains metadata, file tree, and manifests."""
    analyzer = RepoAnalyzer(api_key="gsk_mock123")
    prompt = analyzer.build_analysis_prompt(sample_metadata, sample_content)

    assert "pallets/flask" in prompt
    assert "BSD-3-Clause" in prompt
    assert "src/flask/app.py" in prompt
    assert "pyproject.toml" in prompt
    assert "Flask is a micro web framework" in prompt


def test_missing_api_key_raises(sample_metadata, sample_content):
    """Verify ValueError when Groq API key is missing."""
    analyzer = RepoAnalyzer(api_key="")
    with pytest.raises(ValueError, match="Groq API Key"):
        analyzer.analyze_repository(sample_metadata, sample_content)


def test_analyze_repository_mocked(sample_metadata, sample_content):
    """Verify inference pipeline execution and report output."""
    analyzer = RepoAnalyzer(api_key="gsk_mock123")

    mock_choice = MagicMock()
    mock_choice.message.content = """## 1. Executive Summary
Flask is a WSGI web framework for Python.

## 2. Architecture & Design Patterns
Organized into modular src packages.

## 3. Three Practical, Actionable Recommendations
1. Increase pytest coverage.
2. Add architecture diagrams.
3. Add security scanning in GitHub Actions.

## 4. Good First Issues & Contribution Opportunities
- Add type annotations in request helpers.
"""

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch.object(analyzer._client.chat.completions, "create", return_value=mock_resp):
        report = analyzer.analyze_repository(sample_metadata, sample_content)

        assert "## 1. Executive Summary" in report
        assert "## 2. Architecture & Design Patterns" in report
        assert "## 3. Three Practical, Actionable Recommendations" in report
        assert "## 4. Good First Issues & Contribution Opportunities" in report


def test_model_fallback_on_404(sample_metadata, sample_content):
    """Verify automated model fallback when primary model is not found."""
    analyzer = RepoAnalyzer(api_key="gsk_mock123", model="llama-3.3-70b-versatile")

    mock_success_choice = MagicMock()
    mock_success_choice.message.content = "## 1. Executive Summary\nAnalysis generated via fallback."
    mock_success_resp = MagicMock()
    mock_success_resp.choices = [mock_success_choice]

    # First call raises 404 model_not_found, second call succeeds
    error_404 = Exception("Error code: 404 - model_not_found")

    with patch.object(
        analyzer._client.chat.completions,
        "create",
        side_effect=[error_404, mock_success_resp],
    ):
        report = analyzer.analyze_repository(sample_metadata, sample_content)

        assert "Analysis generated via fallback" in report
        assert analyzer.model == "qwen/qwen3.8-27b"
