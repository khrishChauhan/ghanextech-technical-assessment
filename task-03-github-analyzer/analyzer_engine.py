"""AI Technical Analysis Engine powered by Groq Cloud API for GitHub repositories."""

import os
from typing import Dict, List, Optional
from groq import Groq

from github_client import RepoContent, RepoMetadata

ANALYZER_SYSTEM_PROMPT = """You are a Principal Software Architect and GenAI Technical Lead conducting an in-depth codebase audit of an open-source GitHub repository.

Analyze the provided repository metadata, README documentation, file hierarchy tree, and package manifests.
Synthesize a comprehensive, rigorous, and professional technical analysis in clean GitHub-Flavored Markdown.

You MUST structure your response into the following EXACT four sections:

## 1. Executive Summary
- **Core Purpose & Problem Solved:** What problem does this project address, and why does it exist?
- **Target Audience:** Who are the primary consumers, engineers, or organizations using it?
- **Technology Stack:** Primary programming language, runtime, key dependencies, and ecosystem frameworks.
- **Project Health & Popularity:** Synthesis of community traction based on stars, forks, watchers, and issue counts.

## 2. Architecture & Design Patterns
- **Directory Layout & Modularity:** Assessment of directory structure, separation of concerns, and codebase organization.
- **Architectural Patterns:** Identifiable patterns (e.g., layered architecture, micro-framework, event-driven, plugin architecture, pipeline design).
- **Configuration & Build System:** Package management, manifest setup, and tooling.

## 3. Three Practical, Actionable Recommendations
Provide at least THREE distinct, highly actionable technical recommendations with rationale:
1. **Testing & Quality Assurance:** Gaps in unit testing, integration tests, mock harnesses, or edge-case validation.
2. **Documentation & Developer Experience:** Missing guides, API references, architecture decision records (ADRs), or onboarding friction.
3. **CI/CD & DevOps Automation:** Workflow automation, linting, matrix builds, release pipelines, or security scanning.

## 4. Good First Issues & Contribution Opportunities
Identify 3 to 4 concrete, accessible entry points for new open-source contributors wanting to contribute meaningfully to this repository (e.g. type hints, documentation polish, adding test fixtures, CLI helpers). Be specific and reference actual project components.
"""


class RepoAnalyzer:
    """Orchestrates Groq LLM inference to generate structured repository technical analyses."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
    ):
        """
        Args:
            api_key: Groq Cloud API Key (starts with gsk_...).
            model: Target LLM model identifier.
            temperature: Sampling temperature for balanced factual rigor.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client: Optional[Groq] = None

        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str):
        """Update Groq API key and reinitialize client."""
        self.api_key = api_key.strip() if api_key else None
        if self.api_key:
            self._init_client()
        else:
            self._client = None

    def _init_client(self):
        """Initialize official Groq client."""
        if not self.api_key:
            raise ValueError("Groq API Key is required for repository analysis.")
        self._client = Groq(api_key=self.api_key)

    def _ensure_client(self):
        """Ensure client is instantiated before inference."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Groq API Key missing. Please provide it in the sidebar or configure .env.")
            self._init_client()

    def build_analysis_prompt(self, metadata: RepoMetadata, content: RepoContent) -> str:
        """Construct structured user prompt containing all ingested repository context."""
        # Limit README to first 5000 chars to maintain context budget
        readme_preview = content.readme[:5000] if content.readme else "No README available."
        if len(content.readme) > 5000:
            readme_preview += "\n... [README truncated for length]"

        # Format file tree (top 80 paths)
        tree_sample = content.tree[:80]
        tree_formatted = "\n".join(f"- {p}" for p in tree_sample)
        if len(content.tree) > 80:
            tree_formatted += f"\n... and {len(content.tree) - 80} more files (Total {content.total_files} items)"

        # Format manifests
        manifests_text = ""
        if content.manifests:
            for name, manifest_content in content.manifests.items():
                manifests_text += f"\n### Manifest: `{name}`\n```\n{manifest_content[:1000]}\n```\n"
        else:
            manifests_text = "No standard package manifests detected in root."

        prompt = f"""Please perform a comprehensive technical architectural audit on the following GitHub repository:

### REPOSITORY METADATA:
- **Repository:** {metadata.full_name}
- **Description:** {metadata.description}
- **Primary Language:** {metadata.primary_language}
- **Stars:** {metadata.stars:,} | **Forks:** {metadata.forks:,} | **Watchers:** {metadata.watchers:,}
- **Open Issues:** {metadata.open_issues:,}
- **Default Branch:** {metadata.default_branch}
- **License:** {metadata.license}
- **Topics / Tags:** {', '.join(metadata.topics) if metadata.topics else 'None specified'}
- **GitHub URL:** {metadata.html_url}

### DIRECTORY & FILE HIERARCHY TREE:
{tree_formatted}

### PACKAGE MANIFESTS & BUILD CONFIGURATION:
{manifests_text}

### REPOSITORY README DOCUMENTATION (EXCERPT):
```markdown
{readme_preview}
```
"""
        return prompt

    def analyze_repository(self, metadata: RepoMetadata, content: RepoContent) -> str:
        """Execute Groq LLM inference to synthesize the complete technical report.

        Args:
            metadata: Repository metadata metrics.
            content: Ingested README, file tree, and manifests.

        Returns:
            Structured Markdown technical report.
        """
        self._ensure_client()
        user_prompt = self.build_analysis_prompt(metadata, content)

        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Model execution with automated fallback
        candidate_models = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        last_error = None
        for model_name in candidate_models:
            try:
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                self.model = model_name  # Lock to working model
                report = response.choices[0].message.content or "No analysis generated."
                return report
            except Exception as e:
                err_str = str(e)
                if "model_not_found" in err_str or "404" in err_str:
                    last_error = e
                    continue
                raise RuntimeError(f"Groq API error during repository analysis: {e}")

        raise RuntimeError(f"All candidate Groq models failed. Last error: {last_error}")
