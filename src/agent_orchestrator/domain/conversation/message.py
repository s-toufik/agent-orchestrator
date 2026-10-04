from dataclasses import dataclass
from enum import StrEnum


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    speaker: Speaker
    text: str
