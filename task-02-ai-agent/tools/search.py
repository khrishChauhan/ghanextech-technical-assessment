"""Wikipedia Knowledge Search tool utilizing Wikipedia REST and OpenSearch APIs."""

from typing import Any, Dict
import requests


def search_wikipedia(query: str) -> Dict[str, Any]:
    """Search Wikipedia for verified encyclopedic summaries and reference links.

    Args:
        query: Topic, concept, person, or term to search for.

    Returns:
        Structured dict with keys: {title, extract, url}.

    Raises:
        ValueError: If query is empty or no matching Wikipedia article is found.
        RuntimeError: If Wikipedia API cannot be reached.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    clean_query = query.strip()
    headers = {"User-Agent": "GhanexTech-AIAgent/1.0 (developer.intern@ghanextech.com)"}

    # 1. Search for best matching title via opensearch
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={clean_query}&limit=1&namespace=0&format=json"
        search_resp = requests.get(search_url, headers=headers, timeout=10)
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error querying Wikipedia search service: {e}")

    titles = search_data[1] if len(search_data) > 1 else []
    if not titles:
        raise ValueError(f"No Wikipedia article found matching '{clean_query}'.")

    target_title = titles[0].replace(" ", "_")

    # 2. Fetch page summary
    try:
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{target_title}"
        summary_resp = requests.get(summary_url, headers=headers, timeout=10)
        if summary_resp.status_code == 404:
            raise ValueError(f"No summary available for article '{titles[0]}'.")
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error retrieving Wikipedia article summary: {e}")

    title = summary_data.get("title", titles[0])
    extract = summary_data.get("extract", "No summary excerpt available.")
    url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{target_title}")

    return {
        "title": title,
        "extract": extract,
        "url": url,
    }


# Standardized tool definition dictionary
TOOL_DEFINITION: Dict[str, Any] = {
    "name": "wikipedia_search",
    "description": (
        "Search Wikipedia for concise, verified factual knowledge, explanations of concepts, "
        "historical figures, scientific discoveries, or technology overviews."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search topic or entity, e.g. 'Alan Turing', 'Quantum computing'",
            }
        },
        "required": ["query"],
    },
    "function": search_wikipedia,
}
