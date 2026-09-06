from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state


def test_pack_then_unpack_round_trips_a_full_state() -> None:
    state = AgentState(
        conversation=Conversation(
            [
                ConversationMessage(
                    role=Role.ASSISTANT,
                    content="hi",
                    tool_calls=[ToolCall(id="1", name="t", args={"a": 1})],
                )
            ]
        ),
        planner=PlannerDecision(tool_calls=[], answer="done"),
        reflection=ReflectionDecision(action=ReflectionAction.ACCEPT, critique="ok"),
        last_node="final",
        session_id="s1",
        iteration=2,
        max_iterations=10,
        final_answer="the answer",
    )

    restored = unpack_state(pack_state(state))

    assert restored.session_id == "s1"
    assert restored.last_node == "final"
    assert restored.iteration == 2
    assert restored.max_iterations == 10
    assert restored.final_answer == "the answer"
    assert restored.planner is not None and restored.planner.answer == "done"
    assert restored.reflection is not None and restored.reflection.action is ReflectionAction.ACCEPT
    assert restored.conversation.messages[0].tool_calls[0].args == {"a": 1}


def test_unpack_state_applies_defaults_for_a_brand_new_thread() -> None:
    restored = unpack_state({"state": {}})

    assert restored.conversation.messages == []
    assert restored.planner is None
    assert restored.reflection is None
    assert restored.last_node == ""
    assert restored.session_id == ""
    assert restored.iteration == 0
    assert restored.max_iterations == 20
    assert restored.final_answer is None
