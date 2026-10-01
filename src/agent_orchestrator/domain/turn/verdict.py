from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VerdictAction(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class Verdict:
    action: VerdictAction
    critique: str = ""

    @property
    def rejects(self) -> bool:
        return self.action is VerdictAction.RETRY

    @classmethod
    def accept(cls) -> Verdict:
        return cls(VerdictAction.ACCEPT)
