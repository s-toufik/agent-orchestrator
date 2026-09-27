from collections.abc import Callable
from functools import cached_property

from fastapi import APIRouter
from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.inbound.web.controller.stream_agent_controller import (
    StreamAgentController,
)
from agent_orchestrator.adapter.outbound.streaming.sse_queue import SSEQueue
from agent_orchestrator.application.port.inbound.stream_agent_port import StreamAgentPort
from agent_orchestrator.application.port.outbound.agent_port import AgentPort
from agent_orchestrator.application.port.outbound.event_stream_port import EventStreamPort
from agent_orchestrator.application.use_case.stream_agent_usecase import StreamAgentUseCase
from bootstrap.configuration.settings import AgentEngine
from bootstrap.di.anthropic_sdk_di import AnthropicSdkDI
from bootstrap.di.langgraph_di import LangGraphDI
from bootstrap.router.actuator.actuator_router import ActuatorRouter
from bootstrap.router.agent.stream_agent_router import StreamAgentRouter
from src import (
    APPLICATION_AUTHORS_EMAIL,
    APPLICATION_DEPLOYMENT_ENVIRONMENT,
    APPLICATION_NAME,
    APPLICATION_VERSION,
)


class AgentContainer(LangGraphDI, AnthropicSdkDI):
    @property
    def logging(self) -> Logger:
        return self._logging

    @property
    def application_configuration(self) -> ApplicationConfiguration:
        return self._configuration

    @cached_property
    def _routers(self) -> list[APIRouter]:
        return []

    async def boot(self) -> None:
        _ = self._llm_transport_factory
        await self._start_factories()
        await self._create_routers()
        self.logging.info(f"Agent container booted with the {self._settings.engine} engine")

    async def stop(self) -> None:
        await self._stop_factories()
        await self._stop_anthropic_sdk()
        await self._close_llm_http_client()
        await self._close_mcp_session_factories()
        await self._shutdown_telemetry()
        self.logging.info("Agent container shut down")

    @property
    def routers(self) -> list[APIRouter]:
        return self._routers

    async def _create_routers(self):
        self._routers.append(await self._stream_agent_router())
        self._routers.append(self._actuator_router())

    async def _agent(self) -> AgentPort:
        match self._settings.engine:
            case AgentEngine.LANGGRAPH:
                return await self._langgraph_agent()
            case AgentEngine.ANTHROPIC_SDK:
                return await self._anthropic_sdk_agent()

    async def _stream_agent_router(self) -> APIRouter:
        agent: AgentPort = await self._agent()
        use_case: StreamAgentPort = StreamAgentUseCase(agent, self._logging)
        events: Callable[[], EventStreamPort] = SSEQueue
        controller = StreamAgentController(
            use_case,
            events,
            self._logging,
            max_concurrent_streams=self._settings.max_concurrent_streams,
        )
        return StreamAgentRouter(controller).router

    @staticmethod
    def _actuator_router() -> APIRouter:
        return ActuatorRouter(
            app_name=APPLICATION_NAME,
            app_version=APPLICATION_VERSION,
            app_deployment_environment=APPLICATION_DEPLOYMENT_ENVIRONMENT,
            app_authors=APPLICATION_AUTHORS_EMAIL,
        ).router

    @property
    def is_ready(self) -> bool:
        return bool(self._routers)
