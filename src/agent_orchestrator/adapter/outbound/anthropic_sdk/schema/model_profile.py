from dataclasses import dataclass

from claude_agent_sdk.types import EffortLevel

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.agent_limits import AgentLimits


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    limits: AgentLimits
    max_context_tokens: int
    max_output_tokens: int
    temperature: float
    reasoning_effort: EffortLevel | None = None
