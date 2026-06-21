"""Verify Phoenix receives thinking-mode traces before kicking off the main pipeline.

Runs a tiny end-to-end test: 2 extended-thinking calls to Claude (one with a
single document, one with three documents — same shape as Conditions 1 and 3
of the main pipeline), then explicitly flushes the OTel span processor so
nothing is left in the buffer when the script exits.

After this runs, open the Phoenix UI at http://localhost:6006 and look for:
1. Two spans named 'call_claude' (our parent wrapper) with a 'thinking_trace'
   attribute containing the model's reasoning text.
2. Child spans from the AnthropicInstrumentor named like 'messages.create'
   that record the API call itself.

Bypasses the Redis cache (uses a unique salt in the cache_key) so this always
hits the live API — that's the whole point of the verification.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from opentelemetry import trace
from src.tracing import init_tracing
from src.client import call_claude, MODEL


def build_prompt(question: str, documents: list[str]) -> str:
    blocks = ""
    for i, doc in enumerate(documents, 1):
        blocks += f"[DOCUMENT {i}]\n{doc}\n[/DOCUMENT {i}]\n\n"
    return blocks + f"Using the documents above as context, answer the following question:\n{question}"


def main():
    init_tracing()
    print(f"Tracing initialized. Endpoint: "
          f"{os.environ.get('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006/v1/traces')}")
    print()

    # Pull one real record from the dataset for an authentic shape
    with open("data/all_questions_with_documents.json", encoding="utf-8") as f:
        records = json.load(f)
    rec = records[0]
    print(f"Test question: {rec['question'][:60]}...")
    print(f"  correct: {rec['answer']}  |  wrong: {rec['wrong_answer']}")
    print(f"  domain: {rec['domain']}")
    print()

    # Salt the cache key so this always hits the live API
    salt = f"phoenix_verify_{int(time.time())}"

    # Condition 1: doc_a only
    prompt_c1 = build_prompt(rec["question"], [rec["doc_a"]])
    print("--- Condition 1 (doc_a only) — calling Claude with extended thinking ---")
    r1 = call_claude(
        messages=[{"role": "user", "content": prompt_c1}],
        cache_key_parts=[rec["question"], "verify", "c1", salt, MODEL],
        max_tokens=16000,
        extended_thinking=True,
        budget_tokens=8000,
    )
    print(f"  cached: {r1['cached']}")
    print(f"  thinking trace length: {len(r1['thinking'])} chars")
    print(f"  answer (first 200 chars): {r1['answer'][:200]}")
    print(f"  thinking preview: {r1['thinking'][:200]}...")
    print()

    # Condition 3: doc_a + doc_b + doc_c
    prompt_c3 = build_prompt(rec["question"], [rec["doc_a"], rec["doc_b"], rec["doc_c"]])
    print("--- Condition 3 (3 docs) — calling Claude with extended thinking ---")
    r3 = call_claude(
        messages=[{"role": "user", "content": prompt_c3}],
        cache_key_parts=[rec["question"], "verify", "c3", salt, MODEL],
        max_tokens=16000,
        extended_thinking=True,
        budget_tokens=8000,
    )
    print(f"  cached: {r3['cached']}")
    print(f"  thinking trace length: {len(r3['thinking'])} chars")
    print(f"  answer (first 200 chars): {r3['answer'][:200]}")
    print(f"  thinking preview: {r3['thinking'][:200]}...")
    print()

    # Force span flush so nothing is stuck in the BatchSpanProcessor buffer
    print("Flushing OTel spans...")
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=10000)
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    print("Flushed.")
    print()
    print("=" * 70)
    print("Now check Phoenix UI at http://localhost:6006")
    print("=" * 70)
    print("Look for:")
    print("  - 2 spans named 'call_claude' (our parent wrapper)")
    print("  - each with an attribute 'thinking_trace' containing reasoning text")
    print("  - child spans from AnthropicInstrumentor (messages.create) underneath")
    print(f"  - thinking_trace lengths should be ~{len(r1['thinking'])} and ~{len(r3['thinking'])} chars")


if __name__ == "__main__":
    main()
