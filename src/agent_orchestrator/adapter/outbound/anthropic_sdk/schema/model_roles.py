from dataclasses import dataclass

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_models import TurnModels


@dataclass(frozen=True, slots=True)
class ModelRoles:
    # A role left unset follows the model the user selected for the conversation.
    context: ModelProfile | None = None
    plan: ModelProfile | None = None
    reflection: ModelProfile | None = None

    def for_turn(self, selected: ModelProfile) -> TurnModels:
        return TurnModels(
            act=selected,
            context=self.context or selected,
            plan=self.plan or selected,
            reflection=self.reflection or selected,
        )
