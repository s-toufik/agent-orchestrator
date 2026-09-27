from collections.abc import Mapping
from enum import StrEnum
from functools import cached_property

import aiosqlite
from httpx import AsyncClient
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

from agent_orchestrator.adapter.outbound.llm.mapper import ModelSettingsMapper
from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool_provider import McpToolProvider
from agent_orchestrator.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolPort, ToolRegistryPort
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from bootstrap.di.base_di import BaseDI

MCP_CONNECTOR_NAME: str = "toolbox"
EXTERNAL_MCP_PREFIX: str = "external_mcp_"
LLM_CONNECTOR_NAME: str = "llm"
MONGODB_CONNECTOR_NAME: str = "mongodb_checkpointer"

MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b": "qwen3-8b",
    "qwen3.5-0.8b": "qwen3_5-0_8b",
    "qwen3-1.7b": "qwen3-1_7b",
    "qwen3.5-2b": "qwen3_5-2b",
    "lfm2-8b-a1b": "lfm2-8b-a1b",
    "ministral-8b-instruct-2410": "ministral-8b-instruct-2410",
    "gigachat3.1-10b-a1.8b": "gigachat3_1-10b-a1_8b",
    "qwen3-14b": "qwen3-14b",
}


class AgentRole(StrEnum):
    CONTEXT = "agent_context"
    PLAN = "agent_plan"
    REFLECTION = "agent_reflection"
    SUMMARY = "agent_summary"


class AgentDI(BaseDI):
    # ------------------------------------------------------------------ tools
    @cached_property
    def _mcp_connectors(self) -> dict[str, McpConnector]:
        connectors: Mapping[str, McpConnector] = self._configuration.connector[ConnectorType.mcp]
        names: list[str] = [MCP_CONNECTOR_NAME] + sorted(
            name for name in connectors if name.startswith(EXTERNAL_MCP_PREFIX)
        )
        return {name: connectors[name] for name in names}

    @cached_property
    def _mcp_session_factories(self) -> dict[str, StreamableHttpSessionFactory]:
        return {
            name: StreamableHttpSessionFactory(connector=connector, logger=self._logging)
            for name, connector in self._mcp_connectors.items()
        }

    async def _close_mcp_session_factories(self) -> None:
        self.__dict__.pop("_mcp_session_factories", None)

    async def _tool_registry(self) -> ToolRegistryPort:
        tools: list[ToolPort] = []
        for name, factory in self._mcp_session_factories.items():
            tools.extend(await self._discover_tools(name, factory))
        if not tools:
            self._logging.warning("No MCP tools available: the agent will answer without tools")
        return ToolRegistry(tools)

    async def _discover_tools(
        self, name: str, factory: StreamableHttpSessionFactory
    ) -> list[ToolPort]:
        # Tools are optional: a server that is down or advertises nothing is skipped.
        try:
            discovered: list[ToolPort] = await McpToolProvider(factory, required=False).tools()
        except Exception as exception:
            self._logging.warning(f"MCP server '{name}' unavailable, skipping it: {exception}")
            return []
        self._logging.info(
            f"Discovered {len(discovered)} MCP tools from '{name}': "
            f"{', '.join(tool.specification.name for tool in discovered)}"
        )
        return discovered

    # -------------------------------------------------------------------- llm
    @cached_property
    def _llm_transport_factory(self) -> ResilientTransportFactory:
        connector: ApiConnector = self._configuration.connector.api(LLM_CONNECTOR_NAME)

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

    def _role_settings(self, role: AgentRole) -> tuple[ModelConnector, ModelParameters] | None:
        operation: ApiOperation = self._configuration.operation.api(role)
        if operation.parameters.get("model") is None:
            return None
        return ModelSettingsMapper(operation)()

    # --------------------------------------------------------------- database
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
            self._logging.warning(f"MongoDB '{connector_name}' unavailable: {exception}")
            await factory.disconnect()
            return None
