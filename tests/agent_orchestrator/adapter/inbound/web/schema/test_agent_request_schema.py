from agent_orchestrator.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent_orchestrator.application.port.inbound.agent_request import AgentRequest


def test_to_domain_maps_every_field() -> None:
    schema = AgentRequestSchema(message="hi", model_name="qwen3-14b", request_id="r1")

    request = schema.to_domain()

    assert request == AgentRequest(message="hi", model_name="qwen3-14b", request_id="r1")


def test_auto_approve_is_off_unless_the_request_asks_for_it() -> None:
    default = AgentRequestSchema(message="hi", model_name="m", request_id="r1")
    asked = AgentRequestSchema(message="hi", model_name="m", request_id="r1", auto_approve=True)

    assert default.to_domain().auto_approve is False
    assert asked.to_domain().auto_approve is True
