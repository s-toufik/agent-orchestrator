from dataclasses import dataclass, field

from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.enum.agent_message_type import AgentMessageType


@dataclass(frozen=True, slots=True)
class AgentMessage:
    session_id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    message_status: MessageStreamType = MessageStreamType.FINAL
    message_type: AgentMessageType = AgentMessageType.TEXT
