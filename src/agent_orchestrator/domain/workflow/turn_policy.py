from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.conditions import (
    asks_for_plan_without_one,
    asks_for_tools,
    asks_for_tools_out_of_steps,
    is_ambiguous,
    is_plain_direct_answer,
    needs_plan,
    retry_allowed,
)
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.transition import Transition

FIRST_STEP: Step = Step.UNDERSTAND


class TurnPolicy:
    # Read top to bottom: after a step, the first transition whose condition holds is taken.
    TRANSITIONS: tuple[Transition, ...] = (
        Transition(Step.UNDERSTAND, Step.PLAN, needs_plan),
        Transition(Step.UNDERSTAND, Step.CLARIFY, is_ambiguous),
        Transition(Step.UNDERSTAND, Step.ACT),
        Transition(Step.PLAN, Step.FINISH),
        Transition(Step.CLARIFY, Step.FINISH),
        Transition(Step.ACT, Step.PLAN, asks_for_plan_without_one),
        Transition(Step.ACT, Step.FINISH, asks_for_tools_out_of_steps),
        Transition(Step.ACT, Step.RUN_TOOLS, asks_for_tools),
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
