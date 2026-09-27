from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

NO_ANSWER: str = "No answer was produced."
BUDGET_EXHAUSTED: str = (
    "I reached the step limit before finishing this task. "
    "Ask me to continue, or narrow the request."
)


class FinalizeNode(Node):
    def __init__(self, stream_answer: bool = False) -> None:
        self._stream_answer = stream_answer

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        turn = agent_state.turn

        if turn.outcome is None:
            turn.answer, turn.outcome = self._settle(turn)
        answer: str = turn.answer or NO_ANSWER
        turn.answer = answer

        agent_state.transcript.append(ConversationMessage(role=Role.ASSISTANT, content=answer))
        # Only settled answers are streamed, so a draft rejected by reflection is never shown.
        if self._stream_answer:
            self._emit(AgentMessageStream.token(answer))
        return self._pack(agent_state)

    @staticmethod
    def _settle(turn: TurnState) -> tuple[str, TurnOutcome]:
        draft = turn.scratch.last_assistant()
        if draft is None:
            return NO_ANSWER, TurnOutcome.BEST_EFFORT
        if draft.tool_calls:
            return BUDGET_EXHAUSTED, TurnOutcome.BUDGET_EXHAUSTED
        if turn.reflection and turn.reflection.should_retry:
            return draft.content, TurnOutcome.BEST_EFFORT
        return draft.content, TurnOutcome.ANSWERED
