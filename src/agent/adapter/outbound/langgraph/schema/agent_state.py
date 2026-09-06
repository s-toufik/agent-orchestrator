from pydantic import BaseModel, ConfigDict, Field

from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation: Conversation = Field(default_factory=Conversation)
    planner: PlannerDecision | None = None
    reflection: ReflectionDecision | None = None
    last_node: str = ""
    session_id: str = ""
    question: str = ""
    iteration: int = 0
    max_iterations: int = 20
    final_answer: str | None = None
