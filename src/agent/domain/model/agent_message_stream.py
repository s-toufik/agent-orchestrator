from dataclasses import dataclass, field
from typing import Any

from agent.domain.enum.agent_message_status import MessageStreamType


@dataclass(frozen=True, slots=True)
class AgentMessageStream:
    type: MessageStreamType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
