from functools import cached_property

from fastapi import APIRouter
from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.inbound.web.controller.list_models_controller import (
    ListModelsController,
)
from agent_orchestrator.adapter.inbound.web.controller.stream_agent_controller import (
    StreamAgentController,
)
from agent_orchestrator.adapter.outbound.event.queue_turn_event_stream import QueueTurnEventStream
from bootstrap.di.agent_di import AgentDI
from bootstrap.router.actuator.actuator_router import ActuatorRouter
from bootstrap.router.agent.list_models_router import ListModelsRouter
from bootstrap.router.agent.stream_agent_router import StreamAgentRouter
from src import (
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
        self._routers.append(self._list_models_router())
        self._routers.append(self._actuator_router())

    async def _stream_agent_router(self) -> APIRouter:
        controller = StreamAgentController(
            await self._handle_message(),
            QueueTurnEventStream,
            self._logging,
            max_concurrent_streams=self._settings.max_concurrent_streams,
        )
        return StreamAgentRouter(controller).router

    def _list_models_router(self) -> APIRouter:
        controller = ListModelsController(self._list_models(), self._logging)
        return ListModelsRouter(controller).router

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
