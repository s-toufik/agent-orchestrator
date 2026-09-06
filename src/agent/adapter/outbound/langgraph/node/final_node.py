from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState


class FinalNode(Node):
    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        last: ConversationMessage | None = agent_state.conversation.last_assistant()
        agent_state.final_answer = last.content if last else "No answer was produced."
        agent_state.last_node = "final"
        return self._pack(agent_state)
