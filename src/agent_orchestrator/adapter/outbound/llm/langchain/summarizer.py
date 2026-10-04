from langchain_core.messages import HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.messages import from_history, render
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.summary import (
    summary_request,
    summary_system_prompt,
)
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.message import Message
from agent_orchestrator.domain.turn.turn import Turn


class LangChainSummarizer:
    def __init__(self, models: ChatModels, logger: Logger) -> None:
        self._models = models
        self._logger = logger

    async def summarize(self, turn: Turn, summary: str, messages: list[Message]) -> str:
        prompt = [
            SystemMessage(content=summary_system_prompt()),
            HumanMessage(content=summary_request(summary, render(from_history(messages)))),
        ]
        self._logger.debug("Calling the summary model")
        reply = await invoke(self._models.for_role(AgentRole.SUMMARY, turn.model), prompt)
        return str(reply.content).strip()
