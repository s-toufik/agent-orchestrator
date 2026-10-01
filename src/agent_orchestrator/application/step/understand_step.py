from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.outbound.intent_classifier import IntentClassifier
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.workflow.step import Step


class UnderstandStep:
    step = Step.UNDERSTAND

    def __init__(self, classifier: IntentClassifier, logger: Logger) -> None:
        self._classifier = classifier
        self._logger = logger

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        understanding = await self._classifier.understand(conversation, turn)
        if understanding is None:
            self._logger.warning("The message could not be understood; treating it as a task")
            understanding = Understanding.fallback(turn.request)
        conversation.interpret(turn, understanding)
        self._logger.debug(f"Understood: {turn.understanding}")
