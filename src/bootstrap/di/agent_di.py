from collections.abc import Mapping
from functools import cached_property
from typing import Any

import aiosqlite
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import (
    ApiConnector,
    DatabaseConnector,
    McpConnector,
)
from pycraftcore.application_configuration.model.operation import ApiOperation
from pycraftcore.circuit_breaker.configuration import CircuitBreakerSettings
from pycraftcore.http.configuration import HttpClientSettings, LimitsSettings
from pycraftcore.http.policy.http_error_policy import is_business_error, is_retryable
from pycraftcore.repository.adapter import SQLiteRepositoryFactory, SqliteSettingsMapper
from pycraftcore.repository.port import AsyncRepositoryFactory
from pycraftcore.resilient_http.adapter import ResilientTransportFactory
from pycraftcore.resilient_http.configuration import ResilientHttpSettings
from pycraftcore.retry.configuration import RetrySettings

from agent.adapter.outbound.langgraph.build_agent import build_agent
from agent.adapter.outbound.llm.factory import LLMChat
from agent.adapter.outbound.llm.mapper import ModelSettingsMapper
from agent.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent.adapter.outbound.tool.mcp.mcp_tool_provider import McpToolProvider
from agent.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent.adapter.outbound.tool.tool_registry import ToolRegistry
from agent.application.port.outbound.tool_port import ToolPort, ToolRegistryPort
from agent.application.use_case.stream_agent_usecase import on_token
from bootstrap.di.base_di import BaseDI

MCP_CONNECTOR_NAME: str = "toolbox"
# Any connector under `connector.mcp.*` whose name starts with this prefix is
# discovered alongside the toolbox and merged into the same tool catalogue --
# no code change needed to add one, just a new entry in connector/mcp.yml.
EXTERNAL_MCP_PREFIX: str = "external_mcp_"

MODEL_ALIASES: dict[str, str] = {
    "gpt-oss-20b": "gpt_oss_20b",
    "mistralai/ministral-3-14b-reasoning": "ministral_3_14b_reasoning",
    "mistralai/ministral-3-3b": "ministral_3_3b",
    "qwen/qwen3-1.7b": "qwen_3_1p7b",
}


class AgentDI(BaseDI):
    # ------------------------------------------------------------------ tools
    @cached_property
    def _mcp_session_factories(self) -> dict[str, StreamableHttpSessionFactory]:
        connectors: Mapping[str, McpConnector] = self._configuration.connector[ConnectorType.mcp]
        names: list[str] = [MCP_CONNECTOR_NAME] + sorted(
            name for name in connectors if name.startswith(EXTERNAL_MCP_PREFIX)
        )
        return {
            name: StreamableHttpSessionFactory(connector=connectors[name], logger=self._logging)
            for name in names
        }

    async def _close_mcp_session_factories(self) -> None:
        factories = self.__dict__.pop("_mcp_session_factories", None)
        if factories is not None:
            for factory in factories.values():
                await factory.close()

    async def _tool_registry(self) -> ToolRegistryPort:
        tools: list[ToolPort] = []
        for name, factory in self._mcp_session_factories.items():
            await factory.start()
            discovered: list[ToolPort] = await McpToolProvider(factory).tools()
            self._logging.info(
                f"Discovered {len(discovered)} MCP tools from '{name}': "
                f"{', '.join(tool.specification.name for tool in discovered)}"
            )
            tools.extend(discovered)
        return ToolRegistry(tools)

    # -------------------------------------------------------------------- llm
    @cached_property
    def _llm_transport_factory(self) -> ResilientTransportFactory:
        connector: ApiConnector = self._configuration.connector.api("llm")

        http_settings = HttpClientSettings(limits=LimitsSettings(timeout=connector.timeout))
        http_settings.client_params.base_url = connector.base_url
        http_settings.security.certificate = connector.certificate

        settings = ResilientHttpSettings(
            http=http_settings,
            retry=RetrySettings(
                retry_count=connector.retry,
                retry_delay=1,
                max_retry_delay=20,
                jitter=1,
                should_retry=is_retryable,
            ),
            circuit_breaker=CircuitBreakerSettings(
                failure_threshold=3,
                recovery_timeout=30,
                is_excluded=is_business_error,
                name="llm-gateway",
            ),
        )

        return ResilientTransportFactory(
            settings=settings,
            trace_manager=self._telemetry_provider.tracer("llm-gateway"),
            logger=self._logging,
        )

    @cached_property
    def _llm_http_client(self) -> AsyncClient:
        return self._llm_transport_factory.create_async_client()

    async def _close_llm_http_client(self) -> None:
        client = self.__dict__.pop("_llm_http_client", None)
        if client is not None:
            await client.aclose()

    def _model_settings(self, model_name: str) -> tuple[ModelConnector, ModelParameters]:
        operation: ApiOperation = self._configuration.operation.api(model_name)
        return ModelSettingsMapper(operation)()

    def _llm_for_model(self, model_name: str, use_streaming: bool | None = None) -> ChatOpenAI:
        connector, parameters = self._model_settings(model_name)
        if use_streaming is not None:
            parameters.use_streaming = use_streaming
        return LLMChat(connector, parameters, self._llm_http_client).create_chat_client()

    # ------------------------------------------------------------------ graph
    async def _checkpointer(self) -> AsyncSqliteSaver:
        connection: aiosqlite.Connection = await self._sqlite_connection("checkpointer")
        return AsyncSqliteSaver(connection)

    async def _sqlite_connection(self, connector_name: str) -> aiosqlite.Connection:
        connector: DatabaseConnector = self._configuration.connector.database(connector_name)
        factory: AsyncRepositoryFactory = SQLiteRepositoryFactory(SqliteSettingsMapper(connector)())
        self._register_repository(factory)
        return await factory.connection()

    def _build_graph(
        self, model_name: str, checkpointer: Any, tool_registry: ToolRegistryPort
    ) -> Any:
        _, parameters = self._model_settings(model_name)

        graph, _ = build_agent(
            planner_llm=self._llm_for_model(model_name),
            reflection_llm=self._llm_for_model(model_name, use_streaming=False),
            tool_registry=tool_registry,
            model_parameters=parameters,
            logger=self._logging,
            on_token=on_token,
            checkpointer=checkpointer,
        )
        return graph

    async def _build_graphs(self) -> tuple[dict[str, Any], Any]:
        checkpointer = await self._checkpointer()
        tool_registry: ToolRegistryPort = await self._tool_registry()

        graphs: dict[str, Any] = {
            alias: self._build_graph(operation_name, checkpointer, tool_registry)
            for alias, operation_name in MODEL_ALIASES.items()
        }
        return graphs, checkpointer
