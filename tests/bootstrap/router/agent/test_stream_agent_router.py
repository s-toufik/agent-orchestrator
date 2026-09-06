from typing import cast

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from starlette.testclient import TestClient

from agent.adapter.inbound.web.controller.stream_agent_controller import StreamAgentController
from agent.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from bootstrap.router.agent.stream_agent_router import StreamAgentRouter


class FakeController:
    def __init__(self) -> None:
        self.received: AgentRequestSchema | None = None

    async def execute(self, request: AgentRequestSchema) -> StreamingResponse:
        self.received = request

        async def body():
            yield b"event: final\ndata: {}\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")


def test_posting_to_the_endpoint_forwards_the_parsed_request_to_the_controller() -> None:
    controller = FakeController()
    app = FastAPI()
    app.include_router(StreamAgentRouter(cast(StreamAgentController, controller)).router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/agent/stream",
        json={"message": "hi", "model_name": "gpt-oss-20b", "request_id": "r1"},
    )

    assert response.status_code == 200
    assert controller.received is not None
    assert controller.received.message == "hi"
    assert controller.received.model_name == "gpt-oss-20b"
    assert controller.received.request_id == "r1"
    assert response.text == "event: final\ndata: {}\n\n"
