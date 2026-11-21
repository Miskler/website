"""Optional OpenTelemetry tracing for the personal site."""

import atexit
import logging
import os

logger = logging.getLogger(__name__)


def setup_telemetry(app) -> bool:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        from opentelemetry.instrumentation.flask import FlaskInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "personal-site"),
            "deployment.environment": os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT", "production"),
        }))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        # This endpoint accepts a password in its URL and must never be traced.
        FlaskInstrumentor().instrument_app(app, excluded_urls="healthz,get/cv/ok")
        AioHttpClientInstrumentor().instrument()
        atexit.register(provider.shutdown)
        logger.info("OpenTelemetry tracing enabled for personal site")
        return True
    except Exception:
        logger.exception("Could not initialize OpenTelemetry tracing")
        return False
