from langchain_core.messages import BaseMessage, trim_messages

from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.tokens_service import count_message_tokens


class MemoryNode(Node):
    def __init__(self, max_context_tokens: int = 8_000) -> None:
        self._max_context_tokens = max_context_tokens

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        messages: list[BaseMessage] = agent_state.conversation.to_langchain()
        trimmed: list[BaseMessage] = trim_messages(
            messages,
            token_counter=count_message_tokens,
            max_tokens=self._max_context_tokens,
            strategy="last",
            include_system=False,
        )

        agent_state.conversation = Conversation.from_langchain(trimmed)
        agent_state.last_node = "memory"
        return self._pack(agent_state)
