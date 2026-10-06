from typing import Protocol

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn


class Planner(Protocol):
    async def plan(
        self, conversation: Conversation, turn: Turn, tools: list[ToolSpecification]
    ) -> Plan | None: ...
