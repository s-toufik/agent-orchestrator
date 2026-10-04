from agent_orchestrator.application.port.outbound.model_registry import ModelRegistry
from agent_orchestrator.domain.model.model_listing import ModelListing


class ListModels:
    def __init__(self, models: ModelRegistry) -> None:
        self._models = models

    async def handle(self) -> ModelListing:
        return self._models.listing()
