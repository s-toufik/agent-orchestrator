import asyncio
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import aiosqlite
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
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
from pycraftcore.repository.adapter import (
    MongoRepositoryFactory,
    MongoSettingsMapper,
    SqliteRepositoryFactory,
    SqliteSettingsMapper,
)
from pycraftcore.repository.port import AsyncRepositoryFactory
from pycraftcore.resilient_http.adapter import ResilientTransportFactory
from pycraftcore.resilient_http.configuration import ResilientHttpSettings
from pycraftcore.retry.configuration import RetrySettings
from pymongo import MongoClient

from agent_orchestrator.adapter.outbound.langgraph.build_agent import build_agent
from agent_orchestrator.adapter.outbound.llm.factory import LLMChat
from agent_orchestrator.adapter.outbound.llm.mapper import ModelSettingsMapper
from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool_provider import McpToolProvider
from agent_orchestrator.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.application.port.outbound.tool_port import ToolPort, ToolRegistryPort
from agent_orchestrator.application.use_case.stream_agent_usecase import on_token
from bootstrap.di.base_di import BaseDI

MCP_CONNECTOR_NAME: str = "toolbox"
EXTERNAL_MCP_PREFIX: str = "external_mcp_"

SQLITE_CHECKPOINTER: str = "sqlite_checkpointer"
MONGODB_CHECKPOINTER: str = "mongodb_checkpointer"

MODEL_ALIASES: dict[str, str] = {"granite4-7b": "granite4-7b"}


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
        self.__dict__.pop("_mcp_session_factories", None)

    async def _tool_registry(self) -> ToolRegistryPort:
        tools: list[ToolPort] = []
        for name, factory in self._mcp_session_factories.items():
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
    async def _checkpointer(self) -> AsyncSqliteSaver | MongoDBSaver:
        mongo_saver: MongoDBSaver | None = await self._mongodb_checkpointer()
        if mongo_saver is not None:
            self._logging.info("Using MongoDB checkpointer with TTL")
            return mongo_saver

        self._logging.info("Using Sqlite checkpointer with no TTL")
        return await self._sqlite_checkpointer()

    async def _sqlite_checkpointer(self) -> AsyncSqliteSaver:
        connection: aiosqlite.Connection = await self._sqlite_connection(SQLITE_CHECKPOINTER)
        return AsyncSqliteSaver(connection)

    async def _mongodb_checkpointer(self) -> MongoDBSaver | None:
        if client := await self._mongo_connection(MONGODB_CHECKPOINTER):
            connector: DatabaseConnector = self._database_connector(MONGODB_CHECKPOINTER)
            return await asyncio.to_thread(
                MongoDBSaver,
                client,
                db_name=connector.default_name,
                ttl=connector.pool.get("ttl", 3600),
            )
        else:
            return None

    def _database_connector(self, connector_name: str) -> DatabaseConnector:
        return self._configuration.connector.database(connector_name)

    async def _sqlite_connection(self, connector_name: str) -> aiosqlite.Connection:
        connector = self._database_connector(connector_name)
        factory: AsyncRepositoryFactory = SqliteRepositoryFactory(SqliteSettingsMapper(connector)())
        self._register_repository(factory)
        return await factory.connection()

    async def _mongo_connection(self, connector_name: str) -> MongoClient | None:
        connector = self._database_connector(connector_name)
        factory: AsyncRepositoryFactory = MongoRepositoryFactory(MongoSettingsMapper(connector)())
        self._register_repository(factory)
        try:
            await factory.ping()
            return await factory.connection()
        except Exception as exception:
            self._logging.warning(f"MongoDB checkpointer unavailable: {exception}")
            await factory.disconnect()
            return None

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
