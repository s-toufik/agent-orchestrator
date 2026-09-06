from agent.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent.domain.model.agent_request import AgentRequest


def test_to_domain_maps_every_field() -> None:
    schema = AgentRequestSchema(message="hi", model_name="gpt-oss-20b", request_id="r1")

    request = schema.to_domain()

    assert request == AgentRequest(message="hi", model_name="gpt-oss-20b", request_id="r1")
