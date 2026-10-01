from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Plan:

    task: str
    steps: str
