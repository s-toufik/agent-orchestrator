from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class FinishStep:
    step = Step.FINISH

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        if turn.answer is None:
            turn.finish(turn.conclude())
        conversation.close(turn)
