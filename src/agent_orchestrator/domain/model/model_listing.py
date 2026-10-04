from dataclasses import dataclass, field

from agent_orchestrator.domain.workflow.step import Step


@dataclass(frozen=True, slots=True)
class SelectableModel:
    name: str
    context_tokens: int
    max_output_tokens: int
    thinking: bool


@dataclass(frozen=True, slots=True)
class ModelListing:
    models: tuple[SelectableModel, ...]
    pinned_steps: dict[Step, str] = field(default_factory=dict)
