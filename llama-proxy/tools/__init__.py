from typing import Callable
from .search import do_search, SEARCH_TOOL

TOOL_HANDLERS: dict[str, Callable[[dict], str]] = {
    "web_search": lambda args: do_search(args.get("query", "")),
}

TOOL_DEFINITIONS = [SEARCH_TOOL]
