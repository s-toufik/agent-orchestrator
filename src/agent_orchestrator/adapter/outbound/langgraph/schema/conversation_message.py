from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall


class ConversationMessage(BaseModel):
    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
