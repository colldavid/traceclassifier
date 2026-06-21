"""Dump the OTel spans created by a thinking-mode call to stdout.

Same call shape as verify_phoenix.py but with an extra ConsoleSpanExporter so
we can see the exact span tree (names, parent/child relationships, attributes)
without relying on the Phoenix UI. If the AnthropicInstrumentor child span
isn't appearing in Phoenix, this tells us whether it's a UI thing or whether
the span isn't being created at all.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# Set up tracing BEFORE importing client modules so instrumentation is in place
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.anthropic import AnthropicInstrumentor

resource = Resource.create({"openinference.project.name": "hacktrace"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
trace.set_tracer_provider(provider)
AnthropicInstrumentor().instrument()

# Now we can import + use the cache wrapper
from src.client import call_claude, MODEL

salt = f"debug_spans_{int(time.time())}"

print("\n========== making thinking-mode call ==========\n", flush=True)
result = call_claude(
    messages=[{"role": "user", "content": "What is 2 + 2? Think briefly then answer."}],
    cache_key_parts=["debug", salt, MODEL],
    max_tokens=4000,
    extended_thinking=True,
    budget_tokens=2000,
)
print(f"\nanswer: {result['answer'][:200]}", flush=True)
print(f"thinking length: {len(result['thinking'])} chars", flush=True)

print("\n========== flushing spans ==========\n", flush=True)
provider.force_flush(timeout_millis=10000)
provider.shutdown()
print("\nDone — span JSON above shows the actual tree.", flush=True)
