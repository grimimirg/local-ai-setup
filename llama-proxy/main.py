import json
import logging
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import BACKEND_URL, PROXY_PORT, MAX_LOOPS, SEARCH_SYSTEM_PROMPT
from tools import TOOL_HANDLERS, TOOL_DEFINITIONS

logger = logging.getLogger("llama-proxy")
app = FastAPI()


async def _as_sse(data: dict):
    """Convert a complete chat completion JSON response into an SSE stream.

    Clients that request stream=true (e.g. Crush) expect text/event-stream.
    The proxy resolves the full response internally, then re-wraps it here so
    the client receives a well-formed SSE sequence instead of a plain JSON body.
    """
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    base = {
        "id":      data.get("id", ""),
        "object":  "chat.completion.chunk",
        "created": data.get("created", 0),
        "model":   data.get("model", ""),
    }
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
    if tool_calls:
        tc_delta = [
            {"index": i, "id": tc.get("id", ""), "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
            for i, tc in enumerate(tool_calls)
        ]
        logger.debug("  _as_sse: forwarding %d tool_call(s) in SSE delta", len(tc_delta))
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': tc_delta}, 'finish_reason': None}]})}\n\n"
    else:
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': message.get('content', '')}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice.get('finish_reason', 'stop')}]})}\n\n"
    yield "data: [DONE]\n\n"


def _respond(data: dict, status_code: int, as_stream: bool):
    """Return either a StreamingResponse (SSE) or a plain JSONResponse."""
    if as_stream:
        return StreamingResponse(_as_sse(data), media_type="text/event-stream")
    return JSONResponse(data, status_code=status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not body:
        logger.warning("→ /v1/chat/completions — empty request body received from client")
        return JSONResponse({"error": {"message": "empty request body", "type": "proxy_error"}}, status_code=400)

    # Remember whether the client wants a streaming response
    client_wants_stream = body.get("stream", False)
    messages = body.get("messages", [])
    if not messages:
        logger.warning("→ /v1/chat/completions — request has no messages (empty conversation)")
    logger.info("→ /v1/chat/completions | stream=%s | messages=%d",
                client_wants_stream, len(messages))
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            preview = content[:300].replace("\n", " ") if isinstance(content, str) else repr(content)[:300]
            logger.info("  prompt: %s%s", preview, " [...]" if len(str(content)) > 300 else "")
            break

    # Force non-streaming for the tool loop; the final response is returned in full
    body["stream"] = False

    # Inject all registered tools if not already present
    existing_tools = {t.get("function", {}).get("name") for t in body.get("tools", [])}
    for tool in TOOL_DEFINITIONS:
        if tool["function"]["name"] not in existing_tools:
            body.setdefault("tools", []).append(tool)
    logger.debug("  active tools: %s", [t["function"]["name"] for t in body.get("tools", [])])

    # Inject system prompt if the conversation has none
    messages = body.get("messages", [])
    if not any(m.get("role") == "system" for m in messages):
        body["messages"] = [SEARCH_SYSTEM_PROMPT] + messages
        logger.debug("  system prompt injected")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            for loop_i in range(MAX_LOOPS):
                logger.debug("  [loop %d/%d] sending request to backend", loop_i + 1, MAX_LOOPS)
                logger.debug("  → backend | model=%s | messages=%d | tools=%d",
                             body.get("model", "?"), len(body.get("messages", [])),
                             len(body.get("tools", [])))
                resp = await client.post(
                    f"{BACKEND_URL}/v1/chat/completions", json=body
                )
                logger.debug("  [loop %d/%d] backend status: %d", loop_i + 1, MAX_LOOPS, resp.status_code)
                try:
                    data = resp.json()
                except Exception:
                    logger.warning("  backend returned non-JSON response (status=%d)", resp.status_code)
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/octet-stream"),
                    )

                if not data:
                    logger.warning("  [loop %d/%d] backend returned empty JSON body", loop_i + 1, MAX_LOOPS)
                if not data.get("choices"):
                    logger.warning("  [loop %d/%d] backend response has no 'choices' field — raw: %.300s",
                                   loop_i + 1, MAX_LOOPS, str(data))

                choice     = data.get("choices", [{}])[0]
                message    = choice.get("message", {})
                if not message:
                    logger.warning("  [loop %d/%d] backend choice has empty 'message'", loop_i + 1, MAX_LOOPS)
                tool_calls = message.get("tool_calls") or []

                if not tool_calls:
                    # No tool call — the model produced a final text answer
                    final_content = message.get("content") or ""
                    if not final_content:
                        logger.warning("← final answer has empty content — model returned no text and no tool calls")
                    logger.info("← final answer | loop=%d | stream=%s | status=%d | chars=%d",
                                loop_i + 1, client_wants_stream, resp.status_code, len(final_content))
                    logger.debug("  ← answer preview: %s%s",
                                 final_content[:300].replace("\n", " "),
                                 " [...]" if len(final_content) > 300 else "")
                    return _respond(data, resp.status_code, client_wants_stream)

                # If the model called a tool the proxy doesn't own, pass the
                # response back to the client immediately. Clients like Crush
                # send their own tools in the request; the proxy must not try
                # to handle those — the client will execute them and send the
                # result in the next request.
                unknown = [tc["function"]["name"] for tc in tool_calls
                           if tc["function"]["name"] not in TOOL_HANDLERS]
                if unknown:
                    logger.info("  [loop %d/%d] client-owned tool(s) %s — passing through",
                                loop_i + 1, MAX_LOOPS, unknown)
                    for tc in tool_calls:
                        if tc["function"]["name"] in unknown:
                            logger.debug("    ↳ tool=%s args=%.300s",
                                         tc["function"]["name"], tc["function"].get("arguments", ""))
                    return _respond(data, resp.status_code, client_wants_stream)

                # All tool calls are proxy-owned — execute them
                body["messages"].append(message)
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.info("  [loop %d/%d] tool call: %s | args=%s",
                                loop_i + 1, MAX_LOOPS, fn_name, fn_args)

                    result = await asyncio.get_running_loop().run_in_executor(
                        None, TOOL_HANDLERS[fn_name], fn_args
                    )
                    if not result or not result.strip():
                        logger.warning("  [loop %d/%d] tool '%s' returned empty result",
                                       loop_i + 1, MAX_LOOPS, fn_name)
                    logger.debug("  tool result (%d chars): %.120s", len(result), result)

                    body["messages"].append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      result,
                    })

        # MAX_LOOPS reached — return whatever the last response was
        logger.warning("← MAX_LOOPS (%d) reached, returning last response", MAX_LOOPS)
        return _respond(data, resp.status_code, client_wants_stream)
    except httpx.HTTPError as e:
        logger.error("backend connection error: %s", e)
        return JSONResponse(
            {"error": {"message": str(e), "type": "proxy_error"}},
            status_code=502,
        )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="llama-proxy: OpenAI-compatible tool proxy for llama-server")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=PROXY_PORT, help=f"Bind port (default: {PROXY_PORT})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    uvicorn.run(app, host=args.host, port=args.port,
                log_level="debug" if args.debug else "info")
