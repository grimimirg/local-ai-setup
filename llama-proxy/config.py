BACKEND_URL = "http://localhost:8080"
PROXY_PORT  = 8081
MAX_RESULTS = 5
MAX_LOOPS   = 6   # safeguard against infinite tool call chains

SEARCH_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You have access to a web_search tool. "
        "Use it whenever the user asks you to search online, or when the question "
        "involves current events, latest software versions, recent releases, or "
        "any information that may have changed after your training cutoff. "
        "When in doubt about the freshness of your knowledge, search."
    ),
}
