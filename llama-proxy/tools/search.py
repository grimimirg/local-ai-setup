import logging
from proxy.config import MAX_RESULTS
from duckduckgo_search import DDGS

logger = logging.getLogger("llama-proxy.search")

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for up-to-date information, recent events, "
            "software releases, documentation, or anything not reliably covered "
            "by training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}


def do_search(query: str) -> str:
    """Execute a DuckDuckGo search and return formatted results."""
    logger.info("search query: %r", query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))
        logger.debug("search returned %d result(s)", len(results))
        if not results:
            return "No results found for this query."
        parts = [
            f"[{i}] {r['title']}\nURL: {r['href']}\n{r['body']}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning("search error: %s", e)
        return f"Search error: {e}"
