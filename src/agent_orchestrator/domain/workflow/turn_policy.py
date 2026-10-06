from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.conditions import (
    asks_for_plan_without_one,
    asks_for_tools,
    has_approved_plan,
    is_ambiguous,
    is_plain_direct_answer,
    keeps_working_out_of_steps,
    leaves_plan_unfinished,
    needs_plan,
    retry_allowed,
)
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.transition import Transition

FIRST_STEP: Step = Step.UNDERSTAND


class TurnPolicy:
    TRANSITIONS: tuple[Transition, ...] = (
        Transition(Step.UNDERSTAND, Step.PLAN, needs_plan),
        Transition(Step.UNDERSTAND, Step.CLARIFY, is_ambiguous),
        Transition(Step.UNDERSTAND, Step.ACT),
        Transition(Step.PLAN, Step.ACT, has_approved_plan),
        Transition(Step.PLAN, Step.FINISH),
        Transition(Step.CLARIFY, Step.FINISH),
        Transition(Step.ACT, Step.PLAN, asks_for_plan_without_one),
        Transition(Step.ACT, Step.FINISH, keeps_working_out_of_steps),
        Transition(Step.ACT, Step.RUN_TOOLS, asks_for_tools),
        Transition(Step.ACT, Step.ACT, leaves_plan_unfinished),
        Transition(Step.ACT, Step.FINISH, is_plain_direct_answer),
        Transition(Step.ACT, Step.REVIEW),
        Transition(Step.RUN_TOOLS, Step.ACT),
        Transition(Step.REVIEW, Step.FEEDBACK, retry_allowed),
        Transition(Step.REVIEW, Step.FINISH),
        Transition(Step.FEEDBACK, Step.ACT),
        Transition(Step.FINISH, Step.SUMMARIZE),
        Transition(Step.SUMMARIZE, Step.END),
    )

    def next(self, done: Step, turn: Turn) -> Step:
        for transition in self.TRANSITIONS:
            if transition.source is done and transition.when(turn):
                return transition.target
        raise ValueError(f"No transition leaves the step '{done}'")

    def targets(self, source: Step) -> list[Step]:
        return list(dict.fromkeys(t.target for t in self.TRANSITIONS if t.source is source))
