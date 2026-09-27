from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState


class AgentState(BaseModel):
    session_id: str = ""
    transcript: Conversation = Field(default_factory=Conversation)
    summary: str = ""
    pending_plan: Plan | None = None
    limits: AgentLimits = Field(default_factory=AgentLimits)
    turn: TurnState = Field(default_factory=TurnState)
