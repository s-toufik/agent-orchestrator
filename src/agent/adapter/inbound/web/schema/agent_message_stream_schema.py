from pydantic import BaseModel

from agent.domain.enum.agent_message_status import MessageStreamType


class AgentMessageStreamSchema(BaseModel):
    type: MessageStreamType
    content: str

    def serialize(self) -> bytes:
        return f"event: {self.type.value}\ndata: {self.model_dump_json()}\n\n".encode()
