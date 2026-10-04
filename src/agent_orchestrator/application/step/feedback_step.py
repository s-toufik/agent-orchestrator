from pycraftcore.logger.port import Logger

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step

NO_CRITIQUE: str = "No critique provided."


class FeedbackStep:
    step = Step.FEEDBACK

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        verdict = turn.last_verdict
        critique = verdict.critique if verdict and verdict.critique else NO_CRITIQUE
        turn.give_feedback(critique)
        self._logger.debug(f"Retry {turn.retries} with critique: {critique}")
