from langchain_core.messages import HumanMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.messages import from_work, to_draft
from agent_orchestrator.adapter.outbound.llm.langchain.plan_request import (
    REQUEST_PLAN_TOOL,
    direct_draft,
)
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.act import act_system_prompt
from agent_orchestrator.adapter.outbound.llm.langchain.reply.reply_reader import ReplyReader
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.turn import Turn


class LangChainActor:
    def __init__(
        self,
        models: ChatModels,
        window: ContextWindow,
        logger: Logger,
        reply: ReplyReader,
        final_reply: ReplyReader,
    ) -> None:
        self._models = models
        self._window = window
        self._logger = logger
        self._reply = reply
        self._final_reply = final_reply

    async def act(
        self, conversation: Conversation, turn: Turn, tools: list[ToolSpecification]
    ) -> Draft:
        plan = turn.plan
        system = act_system_prompt(turn.query, plan.steps if plan else None, tools)
        tail = [HumanMessage(content=turn.request), *from_work(turn.work)]
        messages = self._window.build(system, conversation, turn.settings.context_tokens, tail)
        model = self._models.for_role(AgentRole.ACT, turn.model)
        self._logger.debug(f"Calling the act model (step {turn.steps_taken + 1})")
        if plan is None:
            reader = self._final_reply if turn.is_plain_direct_answer else self._reply
            reply = await reader.read(model.bind_tools([REQUEST_PLAN_TOOL]), messages)
            return direct_draft(to_draft(reply))
        if tools:
            model = model.bind_tools([_tool(spec) for spec in tools])
        return to_draft(await self._reply.read(model, messages))


def _tool(specification: ToolSpecification) -> dict:
    return {
        "name": specification.name,
        "description": specification.description,
        "parameters": specification.parameters,
    }
