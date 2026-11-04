"""Opt-in, detailed per-request tracing for local RAG development."""
from datetime import datetime, timezone
import json
import time


class RequestTrace:
    """Collect stage timings and optionally print full diagnostic payloads."""

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self.events = []

    def add(self, stage, component, status="complete", *, started=None, details=None, data=None):
        now = time.perf_counter()
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_from_request_start_ms": round((now - self.started) * 1000, 1),
            "stage": stage,
            "component": component,
            "status": status,
        }
        if started is not None:
            event["duration_ms"] = round((now - started) * 1000, 1)
        if details is not None:
            event["details"] = details
        if self.enabled and data is not None:
            event["data"] = data
        self.events.append(event)
        if self.enabled:
            print("[RAG_DEBUG_TRACE] " + json.dumps(event, ensure_ascii=False, default=str), flush=True)
        return event
