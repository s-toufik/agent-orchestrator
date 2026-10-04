from pydantic import BaseModel

from agent_orchestrator.adapter.inbound.web.schema.message_stream_type import MessageStreamType


class AgentMessageStreamSchema(BaseModel):
    type: MessageStreamType
    content: str

    def serialize(self) -> bytes:
        return f"event: {self.type.value}\ndata: {self.model_dump_json()}\n\n".encode()
