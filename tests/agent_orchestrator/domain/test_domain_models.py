from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest
from agent_orchestrator.domain.model.tool_outcome import ToolOutcome
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


def test_request_is_immutable() -> None:
    request = AgentRequest(message="hello", model_name="qwen3-14b", request_id="r1")

    try:
        request.message = "other"  # ty: ignore[invalid-assignment]
    except Exception as exception:
        assert isinstance(exception, (AttributeError, TypeError))
    else:  # pragma: no cover
        raise AssertionError("AgentRequest should be frozen")


def test_stream_events_are_built_by_type() -> None:
    assert AgentMessageStream.token("a") == AgentMessageStream(MessageStreamType.TOKEN, "a")
    assert AgentMessageStream.status("b").type is MessageStreamType.STATUS
    assert AgentMessageStream.reset() == AgentMessageStream(MessageStreamType.RESET, "")
    assert AgentMessageStream.final("c", {"k": "v"}).metadata == {"k": "v"}
    assert AgentMessageStream.final("c").metadata == {}


def test_successful_outcome_exposes_its_output() -> None:
    outcome = ToolOutcome(invocation_id="1", tool_name="t", output="rows")

    assert outcome.content == "rows"


def test_failed_outcome_prefixes_the_error() -> None:
    outcome = ToolOutcome.failure(invocation_id="1", tool_name="t", error="boom")

    assert outcome.content == "Error: boom"


def test_specification_carries_the_raw_json_schema() -> None:
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    specification = ToolSpecification(name="run_sql", description="SQL.", parameters=schema)

    assert specification.parameters is schema
