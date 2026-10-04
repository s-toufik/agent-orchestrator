import re
from collections.abc import Iterable

from agent_orchestrator.domain.conversation.message import Message

_SEPARATOR = re.compile(r"[^a-zA-Z0-9]+")
_MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return sum(1 + len(_SEPARATOR.findall(word)) for word in text.split())


def count_texts(texts: Iterable[str]) -> int:
    return sum(count_tokens(text) + _MESSAGE_OVERHEAD for text in texts)


class EstimatedTokenCounter:
    def count(self, messages: list[Message]) -> int:
        return count_texts(message.text for message in messages)
