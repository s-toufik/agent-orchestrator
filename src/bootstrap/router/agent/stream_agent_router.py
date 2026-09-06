from fastapi import APIRouter, Body
from starlette.responses import StreamingResponse

from agent.adapter.inbound.web.controller.stream_agent_controller import StreamAgentController
from agent.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema


class StreamAgentRouter:
    PREFIX: str = "/api/v1"
    ENDPOINT: str = "/agent/stream"

    def __init__(self, controller: StreamAgentController) -> None:
        self._controller = controller
        self._router: APIRouter = APIRouter(prefix=self.PREFIX)
        self._router_register()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _router_register(self) -> None:
        self._router.add_api_route(self.ENDPOINT, self._stream, methods=["POST"])

    async def _stream(self, request: AgentRequestSchema = Body(...)) -> StreamingResponse:
        return await self._controller.execute(request)
