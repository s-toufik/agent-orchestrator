from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Feedback:
    critique: str
    kind: Literal["feedback"] = "feedback"
