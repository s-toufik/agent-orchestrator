from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import retry


def test_pack_then_unpack_round_trips_a_full_state() -> None:
    state = AgentState(
        session_id="s1",
        summary="earlier",
        transcript=Conversation(messages=[ConversationMessage(role=Role.USER, content="hi")]),
        pending_plan=Plan(task="t", steps="1. do"),
        turn=TurnState(
            user_message="hi",
            context=TurnContext(intent=Intent.TASK, standalone_query="hi", success_criteria=["a"]),
            scratch=Conversation(
                messages=[
                    ConversationMessage(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=[ToolCall(id="1", name="t", args={"a": 1})],
                    )
                ]
            ),
            reflection=retry(),
            iteration=2,
            answer="x",
            outcome=TurnOutcome.BEST_EFFORT,
        ),
    )

    assert unpack_state(pack_state(state)) == state


def test_the_packed_state_is_plain_json_for_the_checkpointer() -> None:
    packed = pack_state(
        AgentState(
            transcript=Conversation(messages=[ConversationMessage(role=Role.USER, content="hi")])
        )
    )

    assert packed["state"]["transcript"]["messages"][0]["role"] == "user"


def test_unpack_ignores_keys_from_older_checkpoints() -> None:
    legacy = {"state": {"session_id": "s", "conversation": [], "question": "q", "iteration": 3}}

    state = unpack_state(legacy)

    assert state.session_id == "s"
    assert state.transcript.messages == []
