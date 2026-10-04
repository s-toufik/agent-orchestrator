from typing import Protocol

from agent_orchestrator.domain.model.model_listing import ModelListing


class ListModelsPort(Protocol):
    async def handle(self) -> ModelListing: ...
