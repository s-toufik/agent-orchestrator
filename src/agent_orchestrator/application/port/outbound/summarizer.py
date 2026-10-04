from typing import Protocol

from agent_orchestrator.domain.conversation.message import Message
from agent_orchestrator.domain.turn.turn import Turn


class Summarizer(Protocol):
    async def summarize(self, turn: Turn, summary: str, messages: list[Message]) -> str: ...
