from langchain_core.messages import HumanMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.dto import PlanDto
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke_structured
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.plan import plan_system_prompt
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn


class LangChainPlanner:
    def __init__(self, models: ChatModels, window: ContextWindow, logger: Logger) -> None:
        self._models = models
        self._window = window
        self._logger = logger

    async def plan(
        self, conversation: Conversation, turn: Turn, tools: list[ToolSpecification]
    ) -> Plan | None:
        previous = conversation.pending_plan
        system = plan_system_prompt(
            turn.query, tools, PlanDto.model_json_schema(), previous.render() if previous else None
        )
        messages = self._window.build(
            system, conversation, turn.settings.context_tokens, [HumanMessage(content=turn.request)]
        )
        self._logger.debug("Calling the plan model")
        model = self._models.for_role(AgentRole.PLAN, turn.model)
        planned = await invoke_structured(model, PlanDto, messages)
        if planned is None:
            self._logger.warning("The plan could not be read")
            return None
        return planned.to_domain(turn.query, {tool.name for tool in tools})
