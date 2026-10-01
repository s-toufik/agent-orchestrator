from typing import Protocol

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.verdict import Verdict


class Reviewer(Protocol):
    async def review(self, conversation: Conversation, turn: Turn) -> Verdict | None:
        """Judge the turn's last draft; None when the judgement is unreadable."""
        ...
