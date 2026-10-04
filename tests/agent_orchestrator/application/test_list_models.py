from agent_orchestrator.application.use_case.list_models import ListModels
from tests.agent_orchestrator.application.fakes import LISTING, FakeModels


async def test_the_listing_is_the_registry_s() -> None:
    assert await ListModels(FakeModels()).handle() == LISTING
