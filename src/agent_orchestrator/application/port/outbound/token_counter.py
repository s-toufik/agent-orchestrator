from typing import Protocol

from agent_orchestrator.domain.conversation.message import Message


class TokenCounter(Protocol):
    def count(self, messages: list[Message]) -> int: ...
