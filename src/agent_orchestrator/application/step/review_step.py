from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.outbound.reviewer import Reviewer
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.verdict import Verdict
from agent_orchestrator.domain.workflow.step import Step


class ReviewStep:
    step = Step.REVIEW

    def __init__(self, reviewer: Reviewer, logger: Logger) -> None:
        self._reviewer = reviewer
        self._logger = logger

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        verdict = await self._reviewer.review(conversation, turn)
        if verdict is None:
            self._logger.warning("The review could not be read; accepting the draft")
            verdict = Verdict.accept()
        turn.reviewed(verdict)
