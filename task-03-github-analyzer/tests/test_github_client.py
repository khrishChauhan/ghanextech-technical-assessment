"""Unit tests for GitHub REST API client."""

from unittest.mock import MagicMock, patch
import pytest
import requests
from github_client import (
    GitHubClient,
    GitHubRateLimitError,
    InvalidRepositoryURLError,
    RepositoryNotFoundError,
)


def test_parse_valid_github_urls():
    """Verify URL parsing handles full URLs, SSH formats, .git suffixes, and shorthands."""
    cases = [
        ("psf/requests", ("psf", "requests")),
        ("https://github.com/psf/requests", ("psf", "requests")),
        ("https://github.com/psf/requests/", ("psf", "requests")),
        ("https://github.com/psf/requests.git", ("psf", "requests")),
        ("github.com/psf/requests", ("psf", "requests")),
        ("git@github.com:psf/requests.git", ("psf", "requests")),
        ("https://github.com/tiangolo/fastapi/tree/master", ("tiangolo", "fastapi")),
    ]

    for input_str, expected in cases:
        assert GitHubClient.parse_repo_identifier(input_str) == expected


def test_parse_invalid_github_urls():
    """Verify invalid strings and non-GitHub hosts raise InvalidRepositoryURLError."""
    invalid_cases = [
        "",
        "   ",
        "single_word_no_slash",
        "https://gitlab.com/owner/repo",
        "https://bitbucket.org/owner/repo",
    ]

    for bad_input in invalid_cases:
        with pytest.raises(InvalidRepositoryURLError):
            GitHubClient.parse_repo_identifier(bad_input)


def test_fetch_metadata_mocked():
    """Verify metadata deserialization into RepoMetadata model."""
    client = GitHubClient(token="test_token")

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"X-RateLimit-Remaining": "50"}
    mock_resp.json.return_value = {
        "name": "flask",
        "full_name": "pallets/flask",
        "owner": {"login": "pallets"},
        "description": "A microframework for Python.",
        "stargazers_count": 65000,
        "forks_count": 15000,
        "open_issues_count": 25,
        "watchers_count": 1800,
        "default_branch": "main",
        "language": "Python",
        "license": {"name": "BSD-3-Clause"},
        "topics": ["python", "web", "wsgi"],
        "html_url": "https://github.com/pallets/flask",
        "size": 12000,
    }

    with patch.object(client.session, "get", return_value=mock_resp):
        meta = client.fetch_repository_metadata("pallets", "flask")

        assert meta.owner == "pallets"
        assert meta.name == "flask"
        assert meta.stars == 65000
        assert meta.forks == 15000
        assert meta.primary_language == "Python"
        assert meta.license == "BSD-3-Clause"
        assert "wsgi" in meta.topics


def test_fetch_readme_mocked():
    """Verify raw README retrieval and missing README fallback."""
    client = GitHubClient()

    # Success case
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"X-RateLimit-Remaining": "49"}
    mock_resp.text = "# Flask Documentation\nWelcome to Flask."

    with patch.object(client.session, "get", return_value=mock_resp):
        readme = client.fetch_readme("pallets", "flask")
        assert "Flask Documentation" in readme

    # 404 Missing case
    mock_resp_404 = MagicMock(spec=requests.Response)
    mock_resp_404.status_code = 404
    mock_resp_404.headers = {"X-RateLimit-Remaining": "48"}

    with patch.object(client.session, "get", return_value=mock_resp_404):
        readme_missing = client.fetch_readme("pallets", "flask")
        assert "No README.md" in readme_missing


def test_fetch_file_tree_mocked():
    """Verify recursive file tree extraction."""
    client = GitHubClient()

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"X-RateLimit-Remaining": "47"}
    mock_resp.json.return_value = {
        "tree": [
            {"path": "src/flask/app.py", "type": "blob"},
            {"path": "tests/test_basic.py", "type": "blob"},
            {"path": "pyproject.toml", "type": "blob"},
        ],
        "truncated": False,
    }

    with patch.object(client.session, "get", return_value=mock_resp):
        paths, total, truncated = client.fetch_file_tree("pallets", "flask", default_branch="main")

        assert len(paths) == 3
        assert total == 3
        assert truncated is False
        assert "src/flask/app.py" in paths


def test_rate_limit_error_handling():
    """Verify HTTP 403 rate limit raises GitHubRateLimitError with reset info."""
    client = GitHubClient()

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 403
    mock_resp.headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1700000000",
    }
    mock_resp.text = "API rate limit exceeded for IP."

    with patch.object(client.session, "get", return_value=mock_resp):
        with pytest.raises(GitHubRateLimitError, match="rate limit exceeded"):
            client.fetch_repository_metadata("pallets", "flask")


def test_repository_not_found_handling():
    """Verify HTTP 404 raises RepositoryNotFoundError."""
    client = GitHubClient()

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 404
    mock_resp.headers = {"X-RateLimit-Remaining": "40"}
    mock_resp.text = "Not Found"

    with patch.object(client.session, "get", return_value=mock_resp):
        with pytest.raises(RepositoryNotFoundError, match="was not found on GitHub"):
            client.fetch_repository_metadata("nonexistent_user", "nonexistent_repo")
