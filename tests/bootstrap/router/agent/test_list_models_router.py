from typing import cast

from fastapi import FastAPI
from starlette.testclient import TestClient

from agent_orchestrator.adapter.inbound.web.controller.list_models_controller import (
    ListModelsController,
)
from agent_orchestrator.adapter.inbound.web.schema.model_list_schema import ModelListSchema
from bootstrap.router.agent.list_models_router import ListModelsRouter
from tests.agent_orchestrator.application.fakes import LISTING


class FakeController:
    async def execute(self) -> ModelListSchema:
        return ModelListSchema.from_domain(LISTING)


def test_getting_the_endpoint_returns_the_controller_listing() -> None:
    app = FastAPI()
    app.include_router(ListModelsRouter(cast(ListModelsController, FakeController())).router)

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"name": "m", "context_tokens": 8000, "max_output_tokens": 1000, "thinking": False}
        ],
        "pinned_steps": {"understand": "small"},
    }
