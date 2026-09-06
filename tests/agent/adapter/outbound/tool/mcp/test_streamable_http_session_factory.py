from typing import Literal

import pytest
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.no_auth import NoAuth
from pycraftcore.authentication.model.token_auth import TokenAuth

from agent.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent.domain.exception.tool_unavailable_exception import ToolUnavailableException


def make_connector(
    auth=None, transport: Literal["sse", "streamable_http"] = "streamable_http"
) -> McpConnector:
    return McpConnector(
        name="toolbox",
        type=ConnectorType.mcp,
        auth=auth or NoAuth(),
        base_url="http://localhost:8001/mcp",
        timeout=5,
        transport=transport,
    )


class FakeStack:
    def __init__(self, aclose_error: Exception | None = None) -> None:
        self.aclose_error = aclose_error
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        if self.aclose_error:
            raise self.aclose_error


def fake_session() -> object:
    return object()


def install_connect(factory: StreamableHttpSessionFactory, *outcomes) -> list:
    """Replace _connect() with a scripted sequence of exceptions / success.

    Each item in `outcomes` is either an Exception (raise it) or None (succeed:
    installs a FakeStack and a sentinel session, exactly as the real _connect
    would). Returns the list of FakeStacks created on each successful call.
    """
    calls = {"n": 0}
    created_stacks: list[FakeStack] = []

    async def fake_connect() -> None:
        outcome = outcomes[calls["n"]]
        calls["n"] += 1
        if outcome is not None:
            raise outcome
        stack = FakeStack()
        created_stacks.append(stack)
        factory._stack = stack  # ty: ignore[invalid-assignment]
        factory._session = fake_session()  # ty: ignore[invalid-assignment]

    factory._connect = fake_connect  # ty: ignore[invalid-assignment]
    return created_stacks


class _NullLogger:
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def critical(self, message: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def exception(self, message: str) -> None: ...


def fast_factory(
    connector: McpConnector | None = None,
    logger=None,
    max_attempts: int = 3,
    initial_delay: float = 0.001,
    max_delay: float = 0.001,
) -> StreamableHttpSessionFactory:
    return StreamableHttpSessionFactory(
        connector=connector or make_connector(),
        logger=logger or _NullLogger(),
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


async def test_start_connects_on_the_first_attempt() -> None:
    factory = fast_factory()
    install_connect(factory, None)

    await factory.start()

    assert (await factory.session()) is not None


async def test_start_is_a_no_op_once_already_connected() -> None:
    factory = fast_factory()
    calls = {"n": 0}

    async def fake_connect() -> None:
        calls["n"] += 1
        factory._stack = FakeStack()  # ty: ignore[invalid-assignment]
        factory._session = fake_session()  # ty: ignore[invalid-assignment]

    factory._connect = fake_connect  # ty: ignore[invalid-assignment]

    await factory.start()
    await factory.start()

    assert calls["n"] == 1


async def test_session_connects_lazily_when_never_started(logger) -> None:
    factory = fast_factory(logger=logger)
    install_connect(factory, None)

    session = await factory.session()

    assert session is not None


async def test_retries_after_transient_failures_then_succeeds(logger) -> None:
    factory = fast_factory(logger=logger, max_attempts=5)
    install_connect(factory, ConnectionError("down"), ConnectionError("down"), None)

    await factory.start()

    assert len(logger.messages("warning")) == 2
    assert logger.messages("info")


async def test_gives_up_after_max_attempts_and_raises_tool_unavailable(logger) -> None:
    factory = fast_factory(logger=logger, max_attempts=3)
    install_connect(factory, ConnectionError("a"), ConnectionError("b"), ConnectionError("c"))

    with pytest.raises(ToolUnavailableException) as excinfo:
        await factory.start()

    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert len(logger.messages("warning")) == 3


async def test_invalidate_tears_down_and_forces_a_reconnect_on_next_use(logger) -> None:
    factory = fast_factory(logger=logger)
    stacks = install_connect(factory, None, None)

    await factory.start()
    await factory.invalidate()

    assert stacks[0].closed is True
    assert factory._session is None

    await factory.session()
    assert len(stacks) == 2


async def test_close_before_start_is_a_no_op(logger) -> None:
    factory = fast_factory(logger=logger)

    await factory.close()  # must not raise


async def test_teardown_swallows_a_raising_aclose(logger) -> None:
    factory = fast_factory(logger=logger)
    calls = {"n": 0}

    async def fake_connect() -> None:
        calls["n"] += 1
        factory._stack = FakeStack(aclose_error=RuntimeError("close failed"))  # ty: ignore[invalid-assignment]
        factory._session = fake_session()  # ty: ignore[invalid-assignment]

    factory._connect = fake_connect  # ty: ignore[invalid-assignment]
    await factory.start()

    await factory.close()  # must not raise even though aclose() blew up

    assert logger.messages("warning")


def test_headers_for_token_auth() -> None:
    factory = fast_factory(
        connector=make_connector(auth=TokenAuth(key_name="Authorization", key_value="secret"))
    )

    assert factory._headers() == {"Authorization": "secret"}


def test_headers_for_basic_auth() -> None:
    factory = fast_factory(connector=make_connector(auth=BasicAuth(username="u", password="p")))

    headers = factory._headers()

    assert headers["Authorization"].startswith("Basic ")


def test_headers_for_no_auth_are_empty() -> None:
    factory = fast_factory(connector=make_connector(auth=NoAuth()))

    assert factory._headers() == {}
