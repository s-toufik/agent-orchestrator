from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.final_node import FinalNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state

node = FinalNode()


async def test_final_answer_is_the_last_assistant_message() -> None:
    state = AgentState(
        conversation=Conversation(
            [
                ConversationMessage(role=Role.USER, content="hi"),
                ConversationMessage(role=Role.ASSISTANT, content="the answer"),
            ]
        )
    )

    result = unpack_state(await node(pack_state(state)))

    assert result.final_answer == "the answer"
    assert result.last_node == "final"


async def test_final_answer_defaults_when_no_assistant_message_exists() -> None:
    state = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.USER, content="hi")])
    )

    result = unpack_state(await node(pack_state(state)))

    assert result.final_answer == "No answer was produced."
