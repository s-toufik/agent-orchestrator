from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnOptions:
    auto_approve: bool = False
