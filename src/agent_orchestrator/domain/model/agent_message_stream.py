from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType


@dataclass(frozen=True, slots=True)
class AgentMessageStream:
    type: MessageStreamType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def token(cls, content: str) -> AgentMessageStream:
        return cls(MessageStreamType.TOKEN, content)

    @classmethod
    def status(cls, content: str) -> AgentMessageStream:
        return cls(MessageStreamType.STATUS, content)

    @classmethod
    def reset(cls) -> AgentMessageStream:
        return cls(MessageStreamType.RESET, "")

    @classmethod
    def final(cls, content: str, metadata: dict[str, Any] | None = None) -> AgentMessageStream:
        return cls(MessageStreamType.FINAL, content, metadata or {})

    @classmethod
    def error(cls, content: str) -> AgentMessageStream:
        return cls(MessageStreamType.ERROR, content)

    @classmethod
    def complete(cls) -> AgentMessageStream:
        return cls(MessageStreamType.COMPLETE, "")
