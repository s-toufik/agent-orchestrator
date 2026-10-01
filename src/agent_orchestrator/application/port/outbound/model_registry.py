from typing import Protocol

from agent_orchestrator.domain.turn.turn_settings import TurnSettings


class ModelRegistry(Protocol):
    def turn_settings(self, model: str) -> TurnSettings:
        """The turn budget for a selectable model; raises UnknownModelException otherwise."""
        ...
