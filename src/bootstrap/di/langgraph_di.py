import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pycraftcore.application_configuration.model.connector import DatabaseConnector

from agent_orchestrator.adapter.outbound.langgraph.build_agent import build_agent
from agent_orchestrator.adapter.outbound.langgraph.langgraph_agent import LangGraphAgent
from agent_orchestrator.adapter.outbound.llm.langchain.chat_factory import LLMChat
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolRegistryPort
from agent_orchestrator.application.port.outbound.agent_port import AgentPort
from bootstrap.di.agent_di import (
    MODEL_ALIASES,
    MONGODB_CONNECTOR_NAME,
    SQLITE_CHECKPOINTER,
    AgentDI,
    AgentRole,
)


class LangGraphDI(AgentDI):
    async def _langgraph_agent(self) -> AgentPort:
        graphs, _ = await self._build_graphs()
        return LangGraphAgent(graphs)

    def _llm_for_model(self, model_name: str, use_streaming: bool | None = None) -> ChatOpenAI:
        connector, parameters = self._model_settings(model_name)
        if use_streaming is not None:
            parameters.use_streaming = use_streaming
        return LLMChat(connector, parameters, self._llm_http_client).create_chat_client()

    def _llm_for_role(self, role: AgentRole, model_name: str) -> ChatOpenAI:
        # A role set in agent.yml uses its own model; `model: null` uses the one selected.
        connector, parameters = self._role_settings(role) or self._model_settings(model_name)
        parameters.use_streaming = False
        return LLMChat(connector, parameters, self._llm_http_client).create_chat_client()

    async def _checkpointer(self) -> AsyncSqliteSaver | MongoDBSaver:
        mongo_saver: MongoDBSaver | None = await self._mongodb_checkpointer()
        if mongo_saver is not None:
            self._logging.info("Using MongoDB checkpointer with TTL")
            return mongo_saver

        self._logging.info("Using Sqlite checkpointer with no TTL")
        return await self._sqlite_checkpointer()

    async def _sqlite_checkpointer(self) -> AsyncSqliteSaver:
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

    def _build_graph(
        self, model_name: str, checkpointer: Any, tool_registry: ToolRegistryPort
    ) -> Any:
        _, parameters = self._role_settings(AgentRole.ACT) or self._model_settings(model_name)

        return build_agent(
            act_llm=self._llm_for_role(AgentRole.ACT, model_name),
            context_llm=self._llm_for_role(AgentRole.CONTEXT, model_name),
            plan_llm=self._llm_for_role(AgentRole.PLAN, model_name),
            reflection_llm=self._llm_for_role(AgentRole.REFLECTION, model_name),
            summary_llm=self._llm_for_role(AgentRole.SUMMARY, model_name),
            tool_registry=tool_registry,
            model_parameters=parameters,
            logger=self._logging,
            checkpointer=checkpointer,
        )

    async def _build_graphs(self) -> tuple[dict[str, Any], Any]:
        checkpointer = await self._checkpointer()
        tool_registry: ToolRegistryPort = await self._tool_registry()

        graphs: dict[str, Any] = {
            alias: self._build_graph(operation_name, checkpointer, tool_registry)
            for alias, operation_name in MODEL_ALIASES.items()
        }
        return graphs, checkpointer
