from typing import Literal

from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.no_auth import NoAuth
from pycraftcore.authentication.model.token_auth import TokenAuth

from agent_orchestrator.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)


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
