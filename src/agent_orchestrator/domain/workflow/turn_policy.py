from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step

FIRST_STEP: Step = Step.UNDERSTAND


class TurnPolicy:

    def next(self, done: Step, turn: Turn) -> Step:
        match done:
            case Step.UNDERSTAND:
                return self._after_understanding(turn)
            case Step.ACT:
                return self._after_action(turn)
            case Step.REVIEW:
                return self._after_review(turn)
            case Step.RUN_TOOLS | Step.FEEDBACK:
                return Step.ACT
            case Step.PLAN | Step.CLARIFY:
                return Step.FINISH
            case Step.FINISH:
                return Step.SUMMARIZE
            case _:
                return Step.END

    @staticmethod
    def _after_understanding(turn: Turn) -> Step:
        intent = turn.intent
        if intent is None or intent.needs_plan:
            return Step.PLAN
        if intent is Intent.AMBIGUOUS:
            return Step.CLARIFY
        return Step.ACT

    @staticmethod
    def _after_action(turn: Turn) -> Step:
        draft = turn.last_draft
        if draft is not None and draft.asks_for_plan and turn.plan is None:
            # The route had no tools: a plan is how the answer gets them.
            return Step.PLAN
        if draft is not None and draft.asks_for_tools:
            if turn.steps_taken >= turn.settings.max_steps:
                return Step.FINISH
            return Step.RUN_TOOLS
        if turn.is_plain_direct_answer:
            return Step.FINISH
        return Step.REVIEW

    @staticmethod
    def _after_review(turn: Turn) -> Step:
        verdict = turn.last_verdict
        if (
            verdict is not None
            and verdict.rejects
            and turn.retries < turn.settings.max_retries
            and turn.steps_taken < turn.settings.max_steps
        ):
            return Step.FEEDBACK
        return Step.FINISH
