from langchain_core.messages import HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.dto import UnderstandingDto
from agent_orchestrator.adapter.outbound.llm.langchain.messages import render
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke_structured
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.context import (
    context_request,
    context_system_prompt,
)
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.understanding import Understanding


class LangChainIntentClassifier:
    def __init__(self, models: ChatModels, window: ContextWindow, logger: Logger) -> None:
        self._models = models
        self._window = window
        self._logger = logger

    async def understand(self, conversation: Conversation, turn: Turn) -> Understanding | None:
        pending = conversation.pending_plan
        history = self._window.history(conversation, turn.settings.context_tokens)
        messages = [
            SystemMessage(content=context_system_prompt(UnderstandingDto.model_json_schema())),
            HumanMessage(
                content=context_request(
                    render(history), pending.render() if pending else None, turn.request
                )
            ),
        ]
        self._logger.debug("Calling the context model")
        model = self._models.for_role(AgentRole.CONTEXT, turn.model)
        understood = await invoke_structured(model, UnderstandingDto, messages)
        return understood.to_domain() if understood else None
