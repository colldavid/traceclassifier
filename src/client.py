import anthropic
from opentelemetry import trace

from src.cache import cache_get, cache_set, make_key

MODEL = "claude-sonnet-4-6"

_tracer = trace.get_tracer("modeltraceprep")


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _parse_response(response) -> tuple[str, str]:
    """Extract (thinking_trace, final_answer) from a response."""
    thinking_parts = []
    text_parts = []
    for block in response.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
    return "\n".join(thinking_parts), "\n".join(text_parts)


def call_claude(
    messages: list[dict],
    *,
    cache_key_parts: list[str],
    max_tokens: int = 4000,
    extended_thinking: bool = False,
    budget_tokens: int = 8000,
) -> dict:
    """Call Claude with caching.

    Two modes:
    - extended_thinking=False: temperature=0, no thinking (correctness gate,
      wrong answer gen, doc gen, judge). max_tokens as specified (1000 or 4000).
    - extended_thinking=True: temperature=1, thinking enabled, budget_tokens=8000,
      max_tokens=16000 (main pipeline, intervention calls).

    Returns dict with keys: answer, thinking, cached
    """
    key = make_key(*cache_key_parts)

    cached = cache_get(key)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "thinking": cached["thinking"],
            "cached": True,
        }

    client = get_client()

    if extended_thinking:
        # Main pipeline / intervention calls: temp 1, thinking on
        kwargs = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 1,
            "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
            "messages": messages,
        }
    else:
        # Correctness gate / wrong answer / doc gen / judge: temp 0, no thinking
        kwargs = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": messages,
        }

    # Wrap in our own span so we can attach the thinking_trace attribute.
    # The instrumentor's span (messages.create) becomes a child — but it
    # closes before returning, so we can't set attributes on it.  Our parent
    # span stays open until after we parse the response.
    with _tracer.start_as_current_span("call_claude") as span:
        response = client.messages.create(**kwargs)
        thinking, answer = _parse_response(response)

        if thinking:
            span.set_attribute("thinking_trace", thinking)

    cache_set(key, {"answer": answer, "thinking": thinking})

    return {
        "answer": answer,
        "thinking": thinking,
        "cached": False,
    }


def preflight_test() -> None:
    """Verify both calling modes work before running the pipeline."""
    client = get_client()

    # Test 1: temperature=0, no thinking
    print("  Test 1: temperature=0, no thinking...")
    r1 = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    _, answer1 = _parse_response(r1)
    print(f"    Answer: {answer1.strip()[:80]}")

    # Test 2: temperature=1, thinking enabled
    print("  Test 2: temperature=1, thinking enabled (budget_tokens=8000)...")
    r2 = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        temperature=1,
        thinking={"type": "enabled", "budget_tokens": 8000},
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    thinking2, answer2 = _parse_response(r2)
    print(f"    Thinking trace: {len(thinking2)} chars")
    print(f"    Answer: {answer2.strip()[:80]}")

    print("  Both modes OK.")
