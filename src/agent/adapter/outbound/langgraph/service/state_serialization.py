from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall


def pack_state(state: AgentState) -> GraphState:
    """AgentState -> GraphState, the plain dict LangGraph checkpoints."""
    return {
        "state": {
            "conversation": _serialize_conversation(state.conversation),
            "planner": state.planner.model_dump(mode="json") if state.planner else None,
            "reflection": state.reflection.model_dump(mode="json") if state.reflection else None,
            "last_node": state.last_node,
            "session_id": state.session_id,
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "final_answer": state.final_answer,
        }
    }


def unpack_state(graph_state: GraphState) -> AgentState:
    """GraphState -> AgentState."""
    payload = graph_state["state"]

    conversation = Conversation(
        [
            ConversationMessage(
                role=Role(message["role"]),
                content=message["content"],
                tool_calls=[ToolCall(**call) for call in (message.get("tool_calls") or [])],
                tool_call_id=message.get("tool_call_id"),
            )
            for message in (payload.get("conversation") or [])
        ]
    )

    return AgentState(
        conversation=conversation,
        planner=PlannerDecision(**payload["planner"]) if payload.get("planner") else None,
        reflection=(
            ReflectionDecision(**payload["reflection"]) if payload.get("reflection") else None
        ),
        last_node=payload.get("last_node", ""),
        session_id=payload.get("session_id", ""),
        iteration=payload.get("iteration", 0),
        max_iterations=payload.get("max_iterations", 20),
        final_answer=payload.get("final_answer"),
    )


def _serialize_conversation(conversation: Conversation) -> list[dict]:
    return [
        {
            "role": message.role.value,
            "content": message.content,
            "tool_calls": [call.model_dump(mode="json") for call in message.tool_calls],
            "tool_call_id": message.tool_call_id,
        }
        for message in conversation.messages
    ]
