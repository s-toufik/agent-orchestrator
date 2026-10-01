from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.outbound.summarizer import Summarizer
from agent_orchestrator.application.port.outbound.token_counter import TokenCounter
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step

KEEP_LAST_MESSAGES: int = 4


class SummarizeStep:
    """Fold older messages into the summary once the history passes half the context."""

    step = Step.SUMMARIZE

    def __init__(
        self,
        summarizer: Summarizer,
        tokens: TokenCounter,
        logger: Logger,
        keep_last: int = KEEP_LAST_MESSAGES,
    ) -> None:
        self._summarizer = summarizer
        self._tokens = tokens
        self._logger = logger
        self._keep_last = keep_last

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        if len(conversation.messages) <= self._keep_last:
            return
        if self._tokens.count(conversation.messages) <= turn.settings.context_tokens // 2:
            return
        try:
            summary = await self._summarizer.summarize(
                turn, conversation.summary, conversation.older_than(self._keep_last)
            )
        except Exception as exception:
            # The answer is already given; a failed summary only delays compaction.
            self._logger.warning(f"Summary failed, history kept as is: {exception}")
            return
        conversation.compact(summary, self._keep_last)
