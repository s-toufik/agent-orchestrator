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

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import EXTERNAL_MCP_PREFIX, MCP_CONNECTOR_NAME, AgentDI, AgentRole
from bootstrap.di.langgraph_di import LangGraphDI

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch, tmp_path):
    # The real config tree interpolates these with no default, so every test
    # that loads real configuration (i.e. everything except the fake-config
    # ones) needs them present -- individual tests can still override them.
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_HOST", str(tmp_path))
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_NAME", "checkpoint")
    # The mongodb_checkpointer connector is also eagerly resolved when the
    # config tree loads, even though most of these tests never touch it --
    # _mongo_connection() then fails to actually connect and falls back to
    # sqlite, which is what these tests expect.
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_HOST", "localhost")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PORT", "27017")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_NAME", "checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_USERNAME", "test")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PASSWORD", "test")


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

    await di._close_mcp_session_factories()  # must not raise


async def test_close_llm_http_client_is_a_no_op_when_never_built() -> None:
    di = AgentDI(make_settings())

    await di._close_llm_http_client()  # must not raise


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


async def test_tool_registry_discovers_tools_from_the_mcp_server() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {
        "toolbox": _FakeMcpSessionFactory([_AdvertisedTool("echo", "Echoes.", {"type": "object"})]),
    }

    registry = await di._tool_registry()

    assert [spec.name for spec in registry.specifications()] == ["echo"]


class _UnreachableMcpSessionFactory:
    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeMcpSession]:
        raise ConnectionError("connection refused")
        yield  # pragma: no cover


async def test_tool_registry_skips_mcp_servers_that_are_down_or_advertise_nothing() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {
        "toolbox": _UnreachableMcpSessionFactory(),
        f"{EXTERNAL_MCP_PREFIX}empty": _FakeMcpSessionFactory([]),
        f"{EXTERNAL_MCP_PREFIX}analytics": _FakeMcpSessionFactory(
            [_AdvertisedTool("echo", "Echoes.", {"type": "object"})]
        ),
    }

    registry = await di._tool_registry()

    assert [spec.name for spec in registry.specifications()] == ["echo"]


async def test_tool_registry_is_empty_but_usable_when_no_mcp_server_answers() -> None:
    di = AgentDI(make_settings())
    di.__dict__["_mcp_session_factories"] = {"toolbox": _UnreachableMcpSessionFactory()}

    registry = await di._tool_registry()

    assert registry.specifications() == []


async def test_checkpointer_opens_a_real_sqlite_connection() -> None:
    # No real MongoDB is running, so _checkpointer() falls back to sqlite,
    # which is what this test is exercising. _base_env already sets the
    # sqlite/mongo env vars the config tree needs to resolve.
    di = LangGraphDI(make_settings())

    checkpointer = await di._checkpointer()

    assert checkpointer is not None
    await di._stop_factories()


def test_each_role_reads_its_model_from_agent_yml() -> None:
    # Whatever agent.yml holds today: a named model is used, null means "follow the selection".
    di = LangGraphDI(make_settings())

    for role in AgentRole:
        configured = di._configuration.operation.api(role).parameters.get("model")
        settings = di._role_settings(role)

        assert (settings[1].model_name if settings else None) == configured, role


def _freeze(monkeypatch, di: LangGraphDI, frozen: dict[AgentRole, str]) -> None:
    # A role named here is frozen to that model; every other role is null.
    def role_settings(role: AgentRole):
        if role not in frozen:
            return None
        connector, parameters = di._model_settings(frozen[role])
        return connector, parameters.model_copy(update={"max_iterations": 3})

    monkeypatch.setattr(di, "_role_settings", role_settings)


def test_a_null_role_follows_the_selection_and_a_frozen_one_does_not(monkeypatch) -> None:
    di = LangGraphDI(make_settings())
    _freeze(monkeypatch, di, {AgentRole.CONTEXT: "qwen3-8b"})

    context = di._llm_for_role(AgentRole.CONTEXT, "qwen3-14b")
    act = di._llm_for_role(AgentRole.ACT, "qwen3-14b")

    assert context.model_name == "qwen3-8b"
    assert act.model_name == "qwen3-14b"
    assert not context.streaming and not act.streaming


def test_a_frozen_act_ignores_the_selection_and_sets_the_turn_budget(monkeypatch) -> None:
    di = LangGraphDI(make_settings())
    _freeze(monkeypatch, di, {AgentRole.ACT: "qwen3-8b"})
    captured: dict = {}
    monkeypatch.setattr(
        "bootstrap.di.langgraph_di.build_agent", lambda **kwargs: captured.update(kwargs)
    )

    di._build_graph("qwen3-14b", checkpointer=None, tool_registry=None)

    assert captured["act_llm"].model_name == "qwen3-8b"
    assert captured["plan_llm"].model_name == "qwen3-14b"
    assert captured["model_parameters"].max_iterations == 3
