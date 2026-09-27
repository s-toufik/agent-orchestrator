from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.finalize_node import (
    BUDGET_EXHAUSTED,
    NO_ANSWER,
    FinalizeNode,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import accept, retry


def _state(
    draft: ConversationMessage | None, reflection: ReflectionDecision | None = None
) -> AgentState:
    scratch = Conversation(messages=[draft] if draft else [])
    return AgentState(turn=TurnState(scratch=scratch, reflection=reflection))


def _draft(content: str = "the answer", **fields) -> ConversationMessage:
    return ConversationMessage(role=Role.ASSISTANT, content=content, **fields)


async def _run(state: AgentState) -> AgentState:
    return unpack_state(await FinalizeNode(stream_answer=True)(pack_state(state)))


async def test_an_accepted_draft_is_answered_and_committed_to_the_transcript() -> None:
    result = await _run(_state(_draft(), accept()))

    assert result.turn.answer == "the answer"
    assert result.turn.outcome is TurnOutcome.ANSWERED
    last = result.transcript.last()
    assert last is not None
    assert (last.role, last.content) == (Role.ASSISTANT, "the answer")


async def test_a_draft_still_rejected_after_the_last_retry_is_best_effort() -> None:
    result = await _run(_state(_draft(), retry()))

    assert result.turn.answer == "the answer"
    assert result.turn.outcome is TurnOutcome.BEST_EFFORT


async def test_pending_tool_calls_mean_the_step_budget_ran_out() -> None:
    draft = _draft("", tool_calls=[ToolCall(id="1", name="t")])

    result = await _run(_state(draft))

    assert result.turn.answer == BUDGET_EXHAUSTED
    assert result.turn.outcome is TurnOutcome.BUDGET_EXHAUSTED


async def test_no_draft_at_all_never_yields_an_empty_answer() -> None:
    result = await _run(_state(None))

    assert result.turn.answer == NO_ANSWER


async def test_an_answer_settled_upstream_is_kept_as_is() -> None:
    state = AgentState(turn=TurnState(answer="Which desk?", outcome=TurnOutcome.CLARIFICATION))

    result = await _run(state)

    assert result.turn.answer == "Which desk?"
    assert result.turn.outcome is TurnOutcome.CLARIFICATION
