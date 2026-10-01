from pydantic import BaseModel, Field

from agent_orchestrator.adapter.inbound.web.schema.agent_message_type import AgentMessageType
from agent_orchestrator.adapter.inbound.web.schema.message_stream_type import MessageStreamType


class AgentMessageSchema(BaseModel):
    session_id: str
    content: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)
    error: str | None = None
    message_status: MessageStreamType = MessageStreamType.FINAL
    message_type: AgentMessageType = AgentMessageType.TEXT

    def serialize(self) -> bytes:
        # SSE framing
        return f"event: {self.message_status.value}\ndata: {self.model_dump_json()}\n\n".encode()
