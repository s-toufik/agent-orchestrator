from pydantic import BaseModel

from agent_orchestrator.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction


class ReflectionDecision(BaseModel):
    action: ReflectionAction
    critique: str

    @property
    def should_retry(self) -> bool:
        return self.action is ReflectionAction.RETRY
