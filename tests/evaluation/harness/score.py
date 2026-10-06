from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_THRESHOLD: float = 0.8


class ScoreKind(StrEnum):
    ROUTE = "route"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Score:
    name: str
    kind: ScoreKind
    value: float
    threshold: float
    failures: tuple[str, ...] = ()

    @classmethod
    def of_checks(cls, name: str, checks: Sequence[Check], threshold: float) -> Score:
        passed = sum(check.passed for check in checks)
        return cls(
            name=name,
            kind=ScoreKind.ROUTE,
            value=passed / len(checks) if checks else 1.0,
            threshold=threshold,
            failures=tuple(check.detail for check in checks if not check.passed),
        )

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold

    def explain(self) -> str:
        verdict = "at or above" if self.passed else "below"
        lines = [f"{self.name}: {self.value:.2f}, {verdict} the threshold {self.threshold:.2f}"]
        return "\n- ".join([*lines, *self.failures])


def threshold_from_env() -> float:
    raw = os.getenv("EVAL_THRESHOLD", str(DEFAULT_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"EVAL_THRESHOLD={raw!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"EVAL_THRESHOLD={raw!r} must be between 0 and 1")
    return value
