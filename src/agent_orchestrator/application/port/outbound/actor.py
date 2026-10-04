from typing import Protocol

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.turn import Turn


class Actor(Protocol):
    async def act(
        self, conversation: Conversation, turn: Turn, tools: list[ToolSpecification]
    ) -> Draft: ...
