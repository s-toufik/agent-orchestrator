from enum import StrEnum


class TurnMode(StrEnum):
    EXECUTE = "execute"
    PLAN = "plan"
    DIRECT = "direct"
    CLARIFY = "clarify"
