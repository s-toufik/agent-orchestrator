from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state

node = FeedbackNode(PromptService())


async def test_appends_the_critique_as_a_user_message() -> None:
    state = AgentState(
        reflection=ReflectionDecision(action=ReflectionAction.RETRY, critique="too vague")
    )

    result = unpack_state(await node(pack_state(state)))

    last = result.conversation.last()
    assert last is not None
    assert last.role is Role.USER
    assert "too vague" in last.content
    assert result.last_node == "feedback"


async def test_uses_a_default_message_when_there_is_no_reflection() -> None:
    state = AgentState(reflection=None)

    result = unpack_state(await node(pack_state(state)))

    last = result.conversation.last()
    assert last is not None
    assert "No specific critique provided." in last.content
