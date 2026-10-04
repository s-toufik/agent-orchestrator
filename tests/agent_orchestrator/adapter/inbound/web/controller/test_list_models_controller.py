from agent_orchestrator.adapter.inbound.web.controller.list_models_controller import (
    ListModelsController,
)
from agent_orchestrator.application.use_case.list_models import ListModels
from tests.agent_orchestrator.application.fakes import FakeModels


async def test_the_controller_returns_the_listing_as_a_schema(logger) -> None:
    controller = ListModelsController(ListModels(FakeModels()), logger)

    listing = await controller.execute()

    assert [model.name for model in listing.models] == ["m"]
    assert listing.pinned_steps == {"understand": "small"}
