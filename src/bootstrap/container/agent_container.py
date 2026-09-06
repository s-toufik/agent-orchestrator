from collections.abc import Callable
from functools import cached_property

from fastapi import APIRouter
from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.logger.port import Logger

from agent.adapter.inbound.web.controller.stream_agent_controller import StreamAgentController
from agent.adapter.outbound.langgraph.lang_agent import LangAgent
from agent.adapter.outbound.streaming.sse_queue import SSEQueue
from agent.application.port.inbound.stream_agent_port import StreamAgentPort
from agent.application.port.outbound.agent_port import AgentPort
from agent.application.port.outbound.sse_queue_port import SSEQueuePort
from agent.application.use_case.stream_agent_usecase import StreamAgentUseCase
from bootstrap.di.agent_di import AgentDI
from bootstrap.router.actuator.actuator_router import ActuatorRouter
from bootstrap.router.agent.stream_agent_router import StreamAgentRouter
from src import (
    APPLICATION_API_ROOT_PATH,
    APPLICATION_AUTHORS_EMAIL,
    APPLICATION_DEPLOYMENT_ENVIRONMENT,
    APPLICATION_NAME,
    APPLICATION_VERSION,
)


class AgentContainer(AgentDI):
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
        self.logging.info("Agent container booted")

    async def stop(self) -> None:
        await self._stop_factories()
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

    async def _stream_agent_router(self) -> APIRouter:
        graphs, _ = await self._build_graphs()
        agent: AgentPort = LangAgent(graphs)
        use_case: StreamAgentPort = StreamAgentUseCase(agent, self._logging)
        sse_queue: Callable[[], SSEQueuePort] = SSEQueue
        controller = StreamAgentController(use_case, sse_queue, self._logging)
        return StreamAgentRouter(controller).router

    @staticmethod
    def _actuator_router() -> APIRouter:
        return ActuatorRouter(
            app_name=APPLICATION_NAME,
            app_version=APPLICATION_VERSION,
            app_deployment_environment=APPLICATION_DEPLOYMENT_ENVIRONMENT,
            app_authors=APPLICATION_AUTHORS_EMAIL,
            app_api_root_path=APPLICATION_API_ROOT_PATH,
        ).router

    @property
    def is_ready(self) -> bool:
        return bool(self._routers)
