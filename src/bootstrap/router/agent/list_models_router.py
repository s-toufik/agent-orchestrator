from fastapi import APIRouter

from agent_orchestrator.adapter.inbound.web.controller.list_models_controller import (
    ListModelsController,
)
from agent_orchestrator.adapter.inbound.web.schema.model_list_schema import ModelListSchema


class ListModelsRouter:
    PREFIX: str = "/v1"
    ENDPOINT: str = "/models"

    def __init__(self, controller: ListModelsController) -> None:
        self._controller = controller
        self._router: APIRouter = APIRouter(prefix=self.PREFIX)
        self._router_register()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _router_register(self) -> None:
        self._router.add_api_route(
            self.ENDPOINT, self._list, methods=["GET"], response_model=ModelListSchema
        )

    async def _list(self) -> ModelListSchema:
        return await self._controller.execute()
