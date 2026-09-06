from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState


class RouterNode(Node):
    async def __call__(self, state: GraphState) -> str:
        agent_state: AgentState = self._unpack(state)

        if agent_state.iteration >= agent_state.max_iterations:
            return "final"

        match agent_state.last_node:
            case "planner":
                if agent_state.planner and agent_state.planner.wants_tools:
                    return "executor"
                return "reflection"
            case "reflection":
                if agent_state.reflection and agent_state.reflection.should_retry:
                    return "feedback"
                return "final"
            case _:
                return "final"
