from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class ClarifyStep:
    step = Step.CLARIFY

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        understanding = turn.understanding
        question = understanding.clarification_question if understanding else None
        turn.finish(Answer.clarification(question or ""))
