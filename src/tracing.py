import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Suppress noisy retry/export warnings when Phoenix isn't running
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)

PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "hacktrace")

_initialized = False


def init_tracing(endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry tracing with Phoenix OTLP exporter.

    Sets the OpenInference project name so Hacktrace traces show up as a
    distinct project in the Phoenix UI (separate from any prior project
    that wrote to the same Phoenix instance).

    Uses BatchSpanProcessor so export failures don't block the main thread.
    When Phoenix isn't running, spans are silently dropped.
    """
    global _initialized
    if _initialized:
        return

    endpoint = endpoint or os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    resource = Resource.create({"openinference.project.name": PROJECT_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint),
            max_export_batch_size=64,
            schedule_delay_millis=5000,
        )
    )
    trace.set_tracer_provider(provider)

    AnthropicInstrumentor().instrument()
    _initialized = True
