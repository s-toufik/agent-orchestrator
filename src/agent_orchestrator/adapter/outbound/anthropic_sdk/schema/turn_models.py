from dataclasses import dataclass

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile


@dataclass(frozen=True, slots=True)
class TurnModels:
    act: ModelProfile
    context: ModelProfile
    plan: ModelProfile
    reflection: ModelProfile
