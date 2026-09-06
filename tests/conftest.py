from typing import Any

import pytest


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _log(self, level: str, message: str) -> None:
        self.records.append((level, message))

    def info(self, message: str) -> None:
        self._log("info", message)

    def warning(self, message: str) -> None:
        self._log("warning", message)

    def error(self, message: str) -> None:
        self._log("error", message)

    def critical(self, message: str) -> None:
        self._log("critical", message)

    def debug(self, message: str) -> None:
        self._log("debug", message)

    def exception(self, message: str) -> None:
        self._log("exception", message)

    def messages(self, level: str) -> list[str]:
        return [message for recorded, message in self.records if recorded == level]


@pytest.fixture
def logger() -> FakeLogger:
    return FakeLogger()


@pytest.fixture
def anyio_backend() -> Any:
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def _isolate_global_tracer_provider():
    """Neutralize the process-wide OpenTelemetry tracer-provider singleton for
    the whole test session.

    OpenTelemetryProvider.__init__ calls trace.set_tracer_provider(provider) --
    a global, meant to be set once per process. With ~200 tests each
    constructing (and some shutting down) their own provider, the MCP SDK's
    own internal otel_span() calls (used on every real MCP request, server
    and client side) eventually pick up a since-shut-down provider and break
    mid-request ("SSE stream ended without a response"). The provider object
    each DI/test constructs is still fully real, independently usable, and
    independently shutdownable -- only the *global* registration is disabled.
    """
    import opentelemetry.trace as otel_trace

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
        yield
