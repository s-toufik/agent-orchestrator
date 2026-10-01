from pydantic import BaseModel, Field

from agent_orchestrator.application.port.inbound.agent_request import AgentRequest


class AgentRequestSchema(BaseModel):
    message: str = Field(..., min_length=1)
    model_name: str
    request_id: str
    auto_approve: bool = Field(default=False)

    def to_domain(self) -> AgentRequest:
        return AgentRequest(
            message=self.message,
            model_name=self.model_name,
            request_id=self.request_id,
            auto_approve=self.auto_approve,
        )
