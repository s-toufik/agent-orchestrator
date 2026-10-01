from typing import Protocol

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.understanding import Understanding


class IntentClassifier(Protocol):
    async def understand(self, conversation: Conversation, turn: Turn) -> Understanding | None:
        """How the latest message reads against the conversation; None when unreadable."""
        ...
