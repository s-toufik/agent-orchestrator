import asyncio

from agent_orchestrator.application.port.outbound.tool_executor import ToolExecutor
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class RunToolsStep:
    step = Step.RUN_TOOLS

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        draft = turn.last_draft
        calls = draft.tool_calls if draft else ()
        turn.observed(list(await asyncio.gather(*(self._executor.run(call) for call in calls))))
