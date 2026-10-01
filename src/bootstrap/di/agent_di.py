import asyncio
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import aiosqlite
from httpx import AsyncClient
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

from agent_orchestrator.adapter.outbound.langgraph.langgraph_workflow_runner import (
    LangGraphWorkflowRunner,
)
from agent_orchestrator.adapter.outbound.llm.langchain.actor import LangChainActor
from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.intent_classifier import (
    LangChainIntentClassifier,
)
from agent_orchestrator.adapter.outbound.llm.langchain.planner import LangChainPlanner
from agent_orchestrator.adapter.outbound.llm.langchain.reviewer import LangChainReviewer
from agent_orchestrator.adapter.outbound.llm.langchain.summarizer import LangChainSummarizer
from agent_orchestrator.adapter.outbound.llm.langchain.token_counter import (
    EstimatedTokenCounter,
)
from agent_orchestrator.adapter.outbound.llm.mapper import ModelSettingsMapper
from agent_orchestrator.adapter.outbound.llm.model_catalog import (
    AgentRole,
    ModelCatalog,
    ModelSettings,
)
from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool_provider import McpToolProvider
from agent_orchestrator.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolPort
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.adapter.outbound.tool.toolbox import Toolbox
from agent_orchestrator.application.step.act_step import ActStep
from agent_orchestrator.application.step.clarify_step import ClarifyStep
from agent_orchestrator.application.step.feedback_step import FeedbackStep
from agent_orchestrator.application.step.finish_step import FinishStep
from agent_orchestrator.application.step.plan_step import PlanStep
from agent_orchestrator.application.step.review_step import ReviewStep
from agent_orchestrator.application.step.run_tools_step import RunToolsStep
from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.application.step.summarize_step import SummarizeStep
from agent_orchestrator.application.step.understand_step import UnderstandStep
from agent_orchestrator.application.use_case.handle_message import HandleMessage
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy
from bootstrap.di.base_di import BaseDI

MCP_CONNECTOR_NAME: str = "toolbox"
EXTERNAL_MCP_PREFIX: str = "external_mcp_"
LLM_CONNECTOR_NAME: str = "llm"
MONGODB_CONNECTOR_NAME: str = "mongodb_checkpointer"
SQLITE_CHECKPOINTER: str = "sqlite_checkpointer"


class AgentDI(BaseDI):
    """Wiring only: configuration in, adapters, steps, runner and use case out."""

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

    async def _toolbox(self) -> Toolbox:
        tools: list[ToolPort] = []
        for name, factory in self._mcp_session_factories.items():
            tools.extend(await self._discover_tools(name, factory))
        if not tools:
            self._logging.warning("No MCP tools available: the agent will answer without tools")
        return Toolbox(ToolRegistry(tools), self._logging)

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

    @cached_property
    def _model_catalog(self) -> ModelCatalog:
        # Every API operation is a selectable model, except the agent roles of agent.yml.
        roles: set[str] = {role.value for role in AgentRole}
        models: dict[str, ModelSettings] = {}
        for name, operation in self._configuration.operation.by_name.items():
            if name in roles or not isinstance(operation, ApiOperation):
                continue
            connector, parameters = ModelSettingsMapper(operation)()
            models[parameters.model_name] = (connector, parameters)
        return ModelCatalog(models=models, roles=self._frozen_roles())

    def _frozen_roles(self) -> dict[AgentRole, ModelSettings]:
        # A role with `model: null` is left out: it follows the model selected for the turn.
        frozen: dict[AgentRole, ModelSettings] = {}
        for role in AgentRole:
            operation = self._configuration.operation.by_name.get(role.value)
            if isinstance(operation, ApiOperation) and operation.parameters.get("model"):
                frozen[role] = ModelSettingsMapper(operation)()
        return frozen

    @cached_property
    def _chat_models(self) -> ChatModels:
        return ChatModels(self._model_catalog, self._llm_http_client)

    # ------------------------------------------------------------ persistence
    def _database_connector(self, connector_name: str) -> DatabaseConnector:
        return self._configuration.connector.database(connector_name)

    async def _checkpointer(self) -> AsyncSqliteSaver | MongoDBSaver:
        mongo_saver: MongoDBSaver | None = await self._mongodb_checkpointer()
        if mongo_saver is not None:
            self._logging.info("Using MongoDB checkpointer with TTL")
            return mongo_saver

        self._logging.info("Using Sqlite checkpointer with no TTL")
        return AsyncSqliteSaver(await self._sqlite_connection(SQLITE_CHECKPOINTER))

    async def _mongodb_checkpointer(self) -> MongoDBSaver | None:
        client = await self._mongo_connection(MONGODB_CONNECTOR_NAME)
        if client is None:
            return None
        connector: DatabaseConnector = self._database_connector(MONGODB_CONNECTOR_NAME)
        return await asyncio.to_thread(
            MongoDBSaver,
            client,
            db_name=connector.default_name,
            ttl=connector.pool.get("ttl", 3600),
        )

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

    # --------------------------------------------------------------- workflow
    async def _handle_message(self) -> HandleMessage:
        runner = await self._workflow_runner(await self._toolbox(), await self._checkpointer())
        return HandleMessage(runner, self._model_catalog, self._logging)

    async def _workflow_runner(
        self, toolbox: Toolbox, checkpointer: Any
    ) -> LangGraphWorkflowRunner:
        return LangGraphWorkflowRunner(
            self._steps(toolbox), TurnPolicy(), self._logging, checkpointer
        )

    def _steps(self, toolbox: Toolbox) -> list[StepHandler]:
        models, window, logger = self._chat_models, ContextWindow(), self._logging
        return [
            UnderstandStep(LangChainIntentClassifier(models, window, logger), logger),
            PlanStep(LangChainPlanner(models, window, logger), toolbox),
            ClarifyStep(),
            ActStep(LangChainActor(models, window, logger), toolbox),
            RunToolsStep(toolbox),
            ReviewStep(LangChainReviewer(models, logger), logger),
            FeedbackStep(logger),
            FinishStep(),
            SummarizeStep(LangChainSummarizer(models, logger), EstimatedTokenCounter(), logger),
        ]
