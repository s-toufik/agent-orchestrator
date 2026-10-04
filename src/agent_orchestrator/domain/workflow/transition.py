from dataclasses import dataclass

from agent_orchestrator.domain.workflow.conditions import Condition, always
from agent_orchestrator.domain.workflow.step import Step


@dataclass(frozen=True, slots=True)
class Transition:
    source: Step
    target: Step
    when: Condition = always
