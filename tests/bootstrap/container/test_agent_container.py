import asyncio
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import uvicorn
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.no_auth import NoAuth

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.container.agent_container import AgentContainer
from tests.support.mcp_test_server import build_mcp_asgi_app, build_mcp_server

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")


def make_settings() -> ProcessSettings:
    return ProcessSettings(
        role="agent",
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
        host="0.0.0.0",
        port=8000,
    )


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
    connector = McpConnector(
        name="self",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=5,
        transport="streamable_http",
    )
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


def test_is_not_ready_before_boot() -> None:
    container = AgentContainer(make_settings())

    assert container.is_ready is False


def test_logging_and_application_configuration_expose_the_di_internals(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")
    container = AgentContainer(make_settings())

    assert container.logging is container._logging
    assert container.application_configuration is container._configuration


async def test_boot_wires_both_routers_and_becomes_ready(running_toolbox, monkeypatch) -> None:
    monkeypatch.setenv("TOOLBOX_URL", running_toolbox)
    container = AgentContainer(make_settings())

    await container.boot()

    assert container.is_ready is True
    prefixes = {router.prefix for router in container.routers}
    assert prefixes == {"/v1", "/actuator"}

    await container.stop()


async def test_stop_after_boot_does_not_raise(running_toolbox, monkeypatch) -> None:
    monkeypatch.setenv("TOOLBOX_URL", running_toolbox)
    container = AgentContainer(make_settings())
    await container.boot()

    await container.stop()  # must not raise
