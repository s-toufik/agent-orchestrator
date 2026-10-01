from typing import Any

from pydantic import TypeAdapter, ValidationError

from agent_orchestrator.adapter.outbound.langgraph.state.graph_state import GraphState
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn


class TurnStateCodec:
    def __init__(self) -> None:
        self._conversation = TypeAdapter(Conversation)
        self._turn = TypeAdapter(Turn)

    def encode(self, conversation: Conversation, turn: Turn) -> GraphState:
        return {
            "conversation": self._conversation.dump_python(conversation, mode="json"),
            "turn": self._turn.dump_python(turn, mode="json"),
        }

    def decode(self, state: GraphState) -> tuple[Conversation, Turn]:
        return (
            self._conversation.validate_python(state["conversation"]),
            self._turn.validate_python(state["turn"]),
        )

    def decode_turn(self, state: GraphState) -> Turn:
        return self._turn.validate_python(state["turn"])

    def stored_conversation(self, values: dict[str, Any]) -> Conversation | None:
        """The conversation of a stored checkpoint; None when absent or in an older format."""
        stored = values.get("conversation")
        if stored is None:
            return None
        try:
            return self._conversation.validate_python(stored)
        except ValidationError:
            return None
