from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.inbound.web.schema.model_list_schema import ModelListSchema
from agent_orchestrator.application.port.inbound.list_models_port import ListModelsPort


class ListModelsController:
    def __init__(self, use_case: ListModelsPort, logger: Logger) -> None:
        self._use_case = use_case
        self._logger = logger

    async def execute(self) -> ModelListSchema:
        listing = ModelListSchema.from_domain(await self._use_case.handle())
        self._logger.debug(f"listed {len(listing.models)} models")
        return listing
