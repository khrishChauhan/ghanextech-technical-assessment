"""GitHub REST API client for repository metadata, README, and tree ingestion."""

from datetime import datetime, timezone
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import requests


class GitHubAPIError(Exception):
    """Base exception for GitHub API interactions."""


class InvalidRepositoryURLError(GitHubAPIError):
    """Raised when repository input format or URL cannot be parsed."""


class RepositoryNotFoundError(GitHubAPIError):
    """Raised when repository does not exist or is private (HTTP 404)."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub REST API rate limit is exceeded (HTTP 403)."""

    def __init__(self, message: str, reset_timestamp: Optional[int] = None):
        super().__init__(message)
        self.reset_timestamp = reset_timestamp
        self.reset_datetime_utc = (
            datetime.fromtimestamp(reset_timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            if reset_timestamp
            else "Unknown"
        )


class RepoMetadata(BaseModel):
    """Structured repository metadata returned by GitHub REST API."""

    owner: str
    name: str
    full_name: str
    description: str = "No description provided."
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    watchers: int = 0
    default_branch: str = "main"
    primary_language: str = "Not specified"
    license: str = "No License"
    topics: List[str] = Field(default_factory=list)
    html_url: str = ""
    created_at: str = ""
    updated_at: str = ""
    size_kb: int = 0


class RepoContent(BaseModel):
    """Structured repository contents including README, file tree, and manifests."""

    readme: str = ""
    tree: List[str] = Field(default_factory=list)
    total_files: int = 0
    manifests: Dict[str, str] = Field(default_factory=dict)
    tree_truncated: bool = False


# Known package manifest filenames to inspect for dependencies
KNOWN_MANIFEST_NAMES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "Pipfile",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/tests.yml",
    ".github/workflows/main.yml",
}


class GitHubClient:
    """Robust client interfacing with the GitHub REST API v3."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None, timeout: int = 15):
        """
        Args:
            token: Optional GitHub Personal Access Token (PAT).
            timeout: Request timeout in seconds.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if self.token:
            self.token = self.token.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self._build_headers())

    def _build_headers(self) -> Dict[str, str]:
        """Construct standard HTTP headers for GitHub REST API."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GhanexTech-GitHubAnalyzer/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def set_token(self, token: Optional[str]):
        """Update GitHub PAT and refresh session headers."""
        self.token = token.strip() if token else None
        self.session.headers.update(self._build_headers())

    @staticmethod
    def parse_repo_identifier(url_or_repo: str) -> Tuple[str, str]:
        """Extract (owner, repo) from URL or shorthand string.

        Supports:
            - 'psf/requests'
            - 'https://github.com/psf/requests'
            - 'https://github.com/psf/requests.git'
            - 'github.com/psf/requests/'
            - 'git@github.com:psf/requests.git'

        Raises:
            InvalidRepositoryURLError: If input format is invalid or host is not github.com.
        """
        if not url_or_repo or not url_or_repo.strip():
            raise InvalidRepositoryURLError("Repository URL or identifier cannot be empty.")

        cleaned = url_or_repo.strip()

        # Reject non-GitHub domains if a full URL is supplied
        if cleaned.startswith(("http://", "https://")):
            match_domain = re.match(r"^https?:\/\/([^/]+)", cleaned, re.IGNORECASE)
            if match_domain and "github.com" not in match_domain.group(1).lower():
                raise InvalidRepositoryURLError(
                    f"Unsupported repository host '{match_domain.group(1)}'. Only GitHub repositories are supported."
                )

        # Strip .git suffix
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]

        # Strip protocols and domain prefixes
        cleaned = re.sub(r"^(https?:\/\/)?(www\.)?github\.com\/", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^git@github\.com:", "", cleaned, flags=re.IGNORECASE)

        # Split remaining path and take first two segments (owner, repo)
        segments = [s for s in cleaned.split("/") if s]
        if len(segments) >= 2:
            owner, repo = segments[0], segments[1]
            if re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) and re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
                return owner, repo

        raise InvalidRepositoryURLError(
            f"Could not parse repository from '{url_or_repo}'. "
            "Expected format: 'owner/repo' or 'https://github.com/owner/repo'."
        )

    def _check_response_status(self, response: requests.Response, owner: str, repo: str):
        """Inspect response status codes and rate limits, raising typed errors."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_epoch = response.headers.get("X-RateLimit-Reset")
        reset_int = int(reset_epoch) if reset_epoch and reset_epoch.isdigit() else None

        # Handle Rate Limit (HTTP 403 with 0 remaining or rate limit message)
        if response.status_code == 403 and (remaining == "0" or "rate limit" in response.text.lower()):
            err = GitHubRateLimitError(
                f"GitHub API rate limit exceeded. Reset time: "
                f"{datetime.fromtimestamp(reset_int, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if reset_int else 'soon'}. "
                "To increase limit from 60 to 5,000 req/hr, provide a GitHub Personal Access Token in the sidebar.",
                reset_timestamp=reset_int,
            )
            raise err

        # Handle 404 Not Found
        if response.status_code == 404:
            raise RepositoryNotFoundError(
                f"Repository '{owner}/{repo}' was not found on GitHub. "
                "Please verify that the owner and repository names are spelled correctly and the repository is public."
            )

        # Raise for other HTTP errors
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise GitHubAPIError(f"GitHub API HTTP error ({response.status_code}): {e}")

    def fetch_repository_metadata(self, owner: str, repo: str) -> RepoMetadata:
        """Fetch general repository metrics, description, and settings.

        Endpoint: GET /repos/{owner}/{repo}
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            self._check_response_status(resp, owner, repo)
            data = resp.json()
        except requests.RequestException as e:
            raise GitHubAPIError(f"Network error querying GitHub metadata: {e}")

        license_info = data.get("license") or {}
        license_name = license_info.get("name") if isinstance(license_info, dict) else "No License"

        return RepoMetadata(
            owner=data.get("owner", {}).get("login", owner),
            name=data.get("name", repo),
            full_name=data.get("full_name", f"{owner}/{repo}"),
            description=data.get("description") or "No description provided.",
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            open_issues=data.get("open_issues_count", 0),
            watchers=data.get("subscribers_count", data.get("watchers_count", 0)),
            default_branch=data.get("default_branch", "main"),
            primary_language=data.get("language") or "Not specified",
            license=license_name or "No License",
            topics=data.get("topics", []),
            html_url=data.get("html_url", f"https://github.com/{owner}/{repo}"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            size_kb=data.get("size", 0),
        )

    def fetch_readme(self, owner: str, repo: str) -> str:
        """Fetch repository README content as raw markdown text.

        Endpoint: GET /repos/{owner}/{repo}/readme
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme"
        headers = dict(self.session.headers)
        headers["Accept"] = "application/vnd.github.raw"

        try:
            resp = self.session.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 404:
                return "No README.md documentation file found in this repository."
            self._check_response_status(resp, owner, repo)
            return resp.text
        except RepositoryNotFoundError:
            return "No README.md documentation file found in this repository."
        except requests.RequestException as e:
            raise GitHubAPIError(f"Network error retrieving README: {e}")

    def fetch_file_tree(
        self, owner: str, repo: str, default_branch: str = "main", max_items: int = 150
    ) -> Tuple[List[str], int, bool]:
        """Fetch recursive repository git tree to map codebase organization.

        Endpoint: GET /repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1

        Returns:
            Tuple of:
                - List of file and directory paths (up to max_items)
                - Total items count in tree
                - Boolean indicating whether tree was truncated by GitHub
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            # If default_branch fails (e.g. master vs main), fallback to head
            if resp.status_code == 404 and default_branch != "HEAD":
                url_fallback = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
                resp = self.session.get(url_fallback, timeout=self.timeout)

            self._check_response_status(resp, owner, repo)
            data = resp.json()
        except requests.RequestException as e:
            raise GitHubAPIError(f"Network error querying repository tree: {e}")

        tree_items = data.get("tree", [])
        total_count = len(tree_items)
        is_truncated = data.get("truncated", False)

        # Collect paths, prioritizing important directories and root files
        paths: List[str] = [item["path"] for item in tree_items if "path" in item]
        selected_paths = paths[:max_items]

        return selected_paths, total_count, is_truncated

    def fetch_manifests(
        self, owner: str, repo: str, tree_paths: List[str], max_chars_per_file: int = 2500
    ) -> Dict[str, str]:
        """Identify and fetch package manifests and CI configuration files from the tree.

        Endpoint: GET /repos/{owner}/{repo}/contents/{path}
        """
        manifests: Dict[str, str] = {}
        headers = dict(self.session.headers)
        headers["Accept"] = "application/vnd.github.raw"

        matched_files = [p for p in tree_paths if p in KNOWN_MANIFEST_NAMES or os.path.basename(p) in KNOWN_MANIFEST_NAMES]

        # Limit to top 5 manifests to prevent request bloat
        for path in matched_files[:5]:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    content = resp.text
                    if len(content) > max_chars_per_file:
                        content = content[:max_chars_per_file] + "\n... [truncated for analysis]"
                    manifests[path] = content
            except Exception:
                continue

        return manifests

    def fetch_full_repository(self, url_or_repo: str) -> Tuple[RepoMetadata, RepoContent]:
        """Ingest complete repository metadata, README, file tree, and manifests in one call.

        Args:
            url_or_repo: Full URL or shorthand identifier.

        Returns:
            Tuple of (RepoMetadata, RepoContent).
        """
        owner, repo = self.parse_repo_identifier(url_or_repo)
        metadata = self.fetch_repository_metadata(owner, repo)
        readme = self.fetch_readme(owner, repo)
        tree_paths, total_files, is_truncated = self.fetch_file_tree(
            owner, repo, default_branch=metadata.default_branch
        )
        manifests = self.fetch_manifests(owner, repo, tree_paths)

        content = RepoContent(
            readme=readme,
            tree=tree_paths,
            total_files=total_files,
            manifests=manifests,
            tree_truncated=is_truncated,
        )

        return metadata, content
