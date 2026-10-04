from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
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

from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.adapter.outbound.tool.toolbox import Toolbox
from agent_orchestrator.domain.workflow.step import Step
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import EXTERNAL_MCP_PREFIX, MCP_CONNECTOR_NAME, AgentDI

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_HOST", str(tmp_path))
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_NAME", "checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_HOST", "localhost")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PORT", "27017")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_NAME", "checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_USERNAME", "test")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PASSWORD", "test")
    monkeypatch.setenv("LLM_API_KEY", "test")


def make_settings(tmp_path: Path | None = None) -> ProcessSettings:
    return ProcessSettings(
        role="agent-orchestrator",
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
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

    await di._close_mcp_session_factories()


async def test_close_llm_http_client_is_a_no_op_when_never_built() -> None:
    di = AgentDI(make_settings())

    await di._close_llm_http_client()


class _FakeMcpSession:
    def __init__(self, tools: list) -> None:
        self._tools = tools

    async def list_tools(self):
        class Response:
            tools = self._tools

        return Response()


class _FakeMcpSessionFactory:
    def __init__(self, tools: list) -> None:
        self._tools = tools

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeMcpSession]:
        yield _FakeMcpSession(self._tools)


class _AdvertisedTool:
    def __init__(self, name: str, description: str, input_schema: dict) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = None


async def test_toolbox_discovers_tools_from_the_mcp_server() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {
        "toolbox": _FakeMcpSessionFactory([_AdvertisedTool("echo", "Echoes.", {"type": "object"})]),
    }

    toolbox = await di._toolbox()

    assert [spec.name for spec in toolbox.tools()] == ["echo"]


class _UnreachableMcpSessionFactory:
    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeMcpSession]:
        raise ConnectionError("connection refused")
        yield  # pragma: no cover


async def test_toolbox_skips_mcp_servers_that_are_down_or_advertise_nothing() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {
        "toolbox": _UnreachableMcpSessionFactory(),
        f"{EXTERNAL_MCP_PREFIX}empty": _FakeMcpSessionFactory([]),
        f"{EXTERNAL_MCP_PREFIX}analytics": _FakeMcpSessionFactory(
            [_AdvertisedTool("echo", "Echoes.", {"type": "object"})]
        ),
    }

    toolbox = await di._toolbox()

    assert [spec.name for spec in toolbox.tools()] == ["echo"]


async def test_toolbox_is_empty_but_usable_when_no_mcp_server_answers() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {"toolbox": _UnreachableMcpSessionFactory()}

    toolbox = await di._toolbox()

    assert toolbox.tools() == []


async def test_checkpointer_opens_a_real_sqlite_connection() -> None:
    di = AgentDI(make_settings())

    checkpointer = await di._checkpointer()

    assert checkpointer is not None
    await di._stop_factories()


def test_every_model_of_llm_yml_is_selectable_by_its_server_name() -> None:
    di = AgentDI(make_settings())
    operations = di._configuration.operation.by_name
    roles = {role.value for role in AgentRole}
    expected = {
        str(operation.parameters.get("model") or name)
        for name, operation in operations.items()
        if name not in roles
    }

    assert set(di._model_catalog.models) == expected


def test_only_the_roles_naming_a_model_in_agent_yml_are_frozen() -> None:
    di = AgentDI(make_settings())
    operations = di._configuration.operation.by_name

    for role in AgentRole:
        configured = operations[role.value].parameters.get("model")
        frozen = di._model_catalog.roles.get(role)
        assert (frozen[1].model_name if frozen else None) == configured, role


async def test_the_workflow_graph_has_one_node_per_step() -> None:
    di = AgentDI(make_settings())
    runner = await di._workflow_runner(Toolbox(ToolRegistry([]), di._logging), checkpointer=None)

    nodes = set(runner.graph.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {step.value for step in Step if step is not Step.END}
