from langchain_core.messages import HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.dto import VerdictDto
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke_structured
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.review import (
    reflection_request,
    reflection_system_prompt,
)
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Speaker
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.verdict import Verdict

_PREVIOUS_MESSAGES: int = 2


class LangChainReviewer:
    def __init__(
        self,
        models: ChatModels,
        logger: Logger,
        max_evidence_chars: int = 2_000,
        max_message_chars: int = 4_000,
    ) -> None:
        self._models = models
        self._logger = logger
        self._max_evidence_chars = max_evidence_chars
        self._max_message_chars = max_message_chars

    async def review(self, conversation: Conversation, turn: Turn) -> Verdict | None:
        understanding = turn.understanding
        draft = turn.last_draft
        messages = [
            SystemMessage(
                content=reflection_system_prompt(
                    VerdictDto.model_json_schema(), has_evidence=bool(turn.evidence)
                )
            ),
            HumanMessage(
                content=reflection_request(
                    conversation=self._recent(conversation, turn),
                    request=turn.query,
                    criteria=list(understanding.success_criteria) if understanding else [],
                    plan=turn.plan.steps if turn.plan else None,
                    evidence=[
                        self._clip(result.content, self._max_evidence_chars)
                        for result in turn.evidence
                    ],
                    answer=draft.text if draft else "",
                )
            ),
        ]
        self._logger.debug("Calling the reflection model")
        model = self._models.for_role(AgentRole.REFLECTION, turn.model)
        verdict = await invoke_structured(model, VerdictDto, messages)
        return verdict.to_domain() if verdict else None

    def _recent(self, conversation: Conversation, turn: Turn) -> str:
        lines = [
            f"{'User' if m.speaker is Speaker.USER else 'Assistant'}: "
            f"{self._clip(m.text, self._max_message_chars)}"
            for m in conversation.messages[-_PREVIOUS_MESSAGES:]
        ]
        lines.append(f"User: {self._clip(turn.request, self._max_message_chars)}")
        return "\n".join(lines)

    @staticmethod
    def _clip(content: str, limit: int) -> str:
        return content if len(content) <= limit else content[:limit] + " [...]"
