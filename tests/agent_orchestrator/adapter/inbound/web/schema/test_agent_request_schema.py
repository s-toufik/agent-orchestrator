from agent_orchestrator.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent_orchestrator.domain.model.agent_request import AgentRequest


def test_to_domain_maps_every_field() -> None:
    schema = AgentRequestSchema(message="hi", model_name="qwen3-14b", request_id="r1")

    request = schema.to_domain()

    assert request == AgentRequest(message="hi", model_name="qwen3-14b", request_id="r1")
