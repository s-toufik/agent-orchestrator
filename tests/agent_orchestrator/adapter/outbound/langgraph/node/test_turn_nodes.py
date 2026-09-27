from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.clarify_node import ClarifyNode
from agent_orchestrator.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent_orchestrator.adapter.outbound.langgraph.node.ingest_node import IngestNode
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import context, retry


async def test_ingest_records_the_user_message_and_stamps_the_limits() -> None:
    limits = AgentLimits(max_iterations=7, max_retries=1)
    state = AgentState(turn=TurnState(user_message="hello"))

    result = unpack_state(await IngestNode(limits)(pack_state(state)))

    assert result.limits == limits
    last = result.transcript.last()
    assert last is not None
    assert (last.role, last.content) == (Role.USER, "hello")


async def test_clarify_ends_the_turn_with_the_question() -> None:
    state = AgentState(
        turn=TurnState(context=context(Intent.AMBIGUOUS, clarification_question="Which desk?"))
    )

    result = unpack_state(await ClarifyNode()(pack_state(state)))

    assert result.turn.answer == "Which desk?"
    assert result.turn.outcome is TurnOutcome.CLARIFICATION


async def test_feedback_goes_to_the_scratch_not_the_transcript(logger) -> None:
    state = AgentState(turn=TurnState(reflection=retry("missing units")))

    result = unpack_state(await FeedbackNode(logger)(pack_state(state)))

    last = result.turn.scratch.last()
    assert last is not None
    assert last.role is Role.USER
    assert "missing units" in last.content
    assert result.turn.retries == 1
    assert result.transcript.messages == []
