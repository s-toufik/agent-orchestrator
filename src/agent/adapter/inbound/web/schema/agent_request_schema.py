from pydantic import BaseModel, Field

from agent.domain.model.agent_request import AgentRequest


class AgentRequestSchema(BaseModel):
    message: str = Field(..., min_length=1)
    model_name: str
    request_id: str

    def to_domain(self) -> AgentRequest:
        return AgentRequest(
            message=self.message,
            model_name=self.model_name,
            request_id=self.request_id,
        )
