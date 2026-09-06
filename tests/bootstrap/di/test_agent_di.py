import asyncio
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import uvicorn
from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.application_configuration.enum import ConnectorType, RunTypeEnvironment
from pycraftcore.application_configuration.enum.run_type_application import RunTypeApplication
from pycraftcore.application_configuration.model.connector import (
    ConnectorRegistry,
    ConnectorTyping,
    McpConnector,
)
from pycraftcore.application_configuration.model.operation import OperationRegistry
from pycraftcore.authentication.model.no_auth import NoAuth

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import EXTERNAL_MCP_PREFIX, MCP_CONNECTOR_NAME, AgentDI
from tests.support.mcp_test_server import build_mcp_asgi_app, build_mcp_server

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch, tmp_path):
    # The real config tree interpolates these with no default, so every test
    # that loads real configuration (i.e. everything except the fake-config
    # ones) needs them present -- individual tests can still override them.
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")


def make_settings(tmp_path: Path | None = None) -> ProcessSettings:
    return ProcessSettings(
        role="agent",
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
        host="0.0.0.0",
        port=8000,
    )


def _connector(name: str, base_url: str) -> McpConnector:
    return McpConnector(
        name=name,
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=5,
        transport="streamable_http",
    )


def _fake_config(mcp_connectors: dict[str, McpConnector]) -> ApplicationConfiguration:
    by_type: dict[str, ConnectorTyping] = dict(mcp_connectors)
    return ApplicationConfiguration(
        env=RunTypeEnvironment.debug,
        run=RunTypeApplication.asynchronous,
        connector=ConnectorRegistry({ConnectorType.mcp: by_type}),
        operation=OperationRegistry({}),
    )


def make_di_with_fake_config(**mcp_connectors: McpConnector) -> AgentDI:
    di = AgentDI(make_settings())
    di.__dict__["_configuration"] = _fake_config(mcp_connectors)
    return di


def test_mcp_session_factories_always_includes_the_default_toolbox_connector() -> None:
    di = make_di_with_fake_config(
        **{MCP_CONNECTOR_NAME: _connector("toolbox", "http://localhost:8001/mcp")}
    )

    factories = di._mcp_session_factories

    assert set(factories) == {"toolbox"}


def test_mcp_session_factories_merges_in_every_external_mcp_prefixed_connector() -> None:
    di = make_di_with_fake_config(
        **{
            MCP_CONNECTOR_NAME: _connector("toolbox", "http://localhost:8001/mcp"),
            f"{EXTERNAL_MCP_PREFIX}analytics": _connector("analytics", "http://localhost:9001/mcp"),
            f"{EXTERNAL_MCP_PREFIX}billing": _connector("billing", "http://localhost:9002/mcp"),
            "unrelated_other_connector": _connector("other", "http://localhost:9003/mcp"),
        }
    )

    factories = di._mcp_session_factories

    assert set(factories) == {
        "toolbox",
        f"{EXTERNAL_MCP_PREFIX}analytics",
        f"{EXTERNAL_MCP_PREFIX}billing",
    }


async def test_close_mcp_session_factories_is_a_no_op_when_never_built() -> None:
    di = make_di_with_fake_config(
        **{MCP_CONNECTOR_NAME: _connector("toolbox", "http://localhost:8001/mcp")}
    )

    await di._close_mcp_session_factories()  # must not raise


def test_llm_for_model_uses_the_real_configured_connector(monkeypatch) -> None:
    di = AgentDI(make_settings())

    llm = di._llm_for_model("gpt_oss_20b")

    assert llm.model_name == "gpt-oss-20b"
    assert llm.client._client.max_retries == 0
    # _llm_for_model builds _llm_transport_factory, which constructs a real
    # OpenTelemetryProvider (background export thread) -- shut it down.
    di._telemetry_provider.shutdown()


def test_llm_for_model_can_override_streaming() -> None:
    di = AgentDI(make_settings())

    streaming_llm = di._llm_for_model("gpt_oss_20b", use_streaming=True)
    non_streaming_llm = di._llm_for_model("gpt_oss_20b", use_streaming=False)
    di._telemetry_provider.shutdown()

    assert streaming_llm.streaming is True
    assert non_streaming_llm.streaming is False


async def test_close_llm_http_client_is_a_no_op_when_never_built() -> None:
    di = AgentDI(make_settings())

    await di._close_llm_http_client()  # must not raise


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def running_toolbox() -> AsyncIterator[str]:
    server = build_mcp_server(name="toolbox", version="1.0.0")

    @server.tool(name="echo", description="Echoes.")
    async def echo(text: str) -> str:
        return text

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/mcp"
    connector = _connector("self", base_url)
    app = build_mcp_asgi_app(server, connector)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(uv_server.serve())
    try:
        while not uv_server.started:
            await asyncio.sleep(0.01)
        yield base_url
    finally:
        uv_server.should_exit = True
        await serve_task


async def test_tool_registry_discovers_tools_from_a_real_toolbox(
    running_toolbox, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOOLBOX_URL", running_toolbox)
    di = AgentDI(make_settings())

    registry = await di._tool_registry()

    assert [spec.name for spec in registry.specifications()] == ["echo"]

    await di._close_mcp_session_factories()


async def test_checkpointer_opens_a_real_sqlite_connection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")
    di = AgentDI(make_settings())

    checkpointer = await di._checkpointer()

    assert checkpointer is not None
    await di._stop_factories()


async def test_build_graphs_wires_a_graph_per_configured_model(
    running_toolbox, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TOOLBOX_URL", running_toolbox)
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")
    di = AgentDI(make_settings())

    graphs, checkpointer = await di._build_graphs()

    assert set(graphs) == {
        "gpt-oss-20b",
        "mistralai/ministral-3-14b-reasoning",
        "mistralai/ministral-3-3b",
        "qwen/qwen3-1.7b",
    }
    assert checkpointer is not None

    await di._close_mcp_session_factories()
    await di._stop_factories()
    di._telemetry_provider.shutdown()
