from langchain_core.messages import HumanMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.messages import from_work, to_draft
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.act import (
    act_system_prompt,
    plan_request,
)
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.turn import Turn


class LangChainActor:
    def __init__(self, models: ChatModels, window: ContextWindow, logger: Logger) -> None:
        self._models = models
        self._window = window
        self._logger = logger

    async def act(
        self, conversation: Conversation, turn: Turn, tools: list[ToolSpecification]
    ) -> Draft:
        plan = turn.plan
        system = act_system_prompt(turn.query, plan.steps if plan else None, tools)
        tail = [HumanMessage(content=turn.request), *from_work(turn.work)]
        messages = self._window.build(system, conversation, turn.settings.context_tokens, tail)
        model = self._models.for_role(AgentRole.ACT, turn.model)
        # Tools are bound only under an approved plan; providers reject an empty tool list.
        if plan is not None and tools:
            model = model.bind_tools([_tool(spec) for spec in tools])
        self._logger.debug(f"Calling the act model (step {turn.steps_taken + 1})")
        draft = to_draft(await invoke(model, messages))
        if plan is None and (reason := plan_request(draft.text)) is not None:
            return Draft.asking_for_plan(reason)
        return draft


def _tool(specification: ToolSpecification) -> dict:
    return {
        "name": specification.name,
        "description": specification.description,
        "parameters": specification.parameters,
    }
