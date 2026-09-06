from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.memory_node import MemoryNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state


def _conversation(count: int) -> Conversation:
    return Conversation(
        [ConversationMessage(role=Role.USER, content=f"message {i}") for i in range(count)]
    )


async def test_keeps_every_message_when_well_under_budget() -> None:
    node = MemoryNode(max_context_tokens=10_000)
    state = AgentState(conversation=_conversation(5))

    result = unpack_state(await node(pack_state(state)))

    assert len(result.conversation.messages) == 5
    assert result.last_node == "memory"


async def test_trims_the_oldest_messages_first_under_a_tight_budget() -> None:
    node = MemoryNode(max_context_tokens=10)
    state = AgentState(conversation=_conversation(5))

    result = unpack_state(await node(pack_state(state)))

    assert len(result.conversation.messages) < 5
    # "last" strategy: whatever survives must be a suffix of the original conversation.
    kept_contents = [message.content for message in result.conversation.messages]
    assert kept_contents == [f"message {i}" for i in range(5 - len(kept_contents), 5)]
