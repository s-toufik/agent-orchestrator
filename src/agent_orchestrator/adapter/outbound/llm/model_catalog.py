from dataclasses import dataclass, field
from enum import StrEnum

from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.domain.exception.unknown_model_exception import UnknownModelException
from agent_orchestrator.domain.turn.turn_settings import TurnSettings

ModelSettings = tuple[ModelConnector, ModelParameters]


class AgentRole(StrEnum):
    """The model-calling steps; the value is the role's operation in agent.yml."""

    ACT = "agent_act"
    CONTEXT = "agent_context"
    PLAN = "agent_plan"
    REFLECTION = "agent_reflection"
    SUMMARY = "agent_summary"


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """The selectable models (by the name the UI sends) and the roles frozen to a model.

    A role absent from `roles` follows the model selected for the turn.
    """

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

    def _selectable(self, model: str) -> ModelSettings:
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelException(f"Unknown model '{model}'") from None
