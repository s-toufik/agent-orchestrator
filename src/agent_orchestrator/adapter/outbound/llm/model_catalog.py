from dataclasses import dataclass, field
from enum import StrEnum

from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.domain.exception.unknown_model_exception import UnknownModelException
from agent_orchestrator.domain.model.model_listing import ModelListing, SelectableModel
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.workflow.step import Step

ModelSettings = tuple[ModelConnector, ModelParameters]


class AgentRole(StrEnum):
    ACT = "agent_act"
    CONTEXT = "agent_context"
    PLAN = "agent_plan"
    REFLECTION = "agent_reflection"
    SUMMARY = "agent_summary"


ROLE_STEPS: dict[AgentRole, Step] = {
    AgentRole.CONTEXT: Step.UNDERSTAND,
    AgentRole.PLAN: Step.PLAN,
    AgentRole.ACT: Step.ACT,
    AgentRole.REFLECTION: Step.REVIEW,
    AgentRole.SUMMARY: Step.SUMMARIZE,
}


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    models: dict[str, ModelSettings]
    roles: dict[AgentRole, ModelSettings] = field(default_factory=dict)

    def settings_for(self, role: AgentRole, model: str) -> ModelSettings:
        frozen = self.roles.get(role)
        if frozen is not None:
            return frozen
        return self._selectable(model)

    def turn_settings(self, model: str) -> TurnSettings:
        self._selectable(model)
        _, parameters = self.settings_for(AgentRole.ACT, model)
        return TurnSettings(
            max_steps=parameters.max_iterations,
            max_retries=parameters.max_reflection_retries,
            context_tokens=parameters.max_context_tokens,
            stream_answer=parameters.use_streaming,
        )

    def listing(self) -> ModelListing:
        return ModelListing(
            models=tuple(
                SelectableModel(
                    name=name,
                    context_tokens=parameters.max_context_tokens,
                    max_output_tokens=parameters.max_output_tokens,
                    thinking=parameters.reasoning_effort is not None,
                )
                for name, (_, parameters) in self.models.items()
            ),
            pinned_steps={
                ROLE_STEPS[role]: parameters.model_name
                for role, (_, parameters) in self.roles.items()
            },
        )

    def _selectable(self, model: str) -> ModelSettings:
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelException(f"Unknown model '{model}'") from None
