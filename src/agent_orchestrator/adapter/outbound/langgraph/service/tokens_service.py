import re

_SEPARATOR = re.compile(r"[^a-zA-Z0-9]+")
_MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    """Cheap, dependency-free approximation. Good enough to drive trimming."""
    if not text:
        return 0
    return sum(1 + len(_SEPARATOR.findall(word)) for word in text.split())


def count_message_tokens(messages: list) -> int:
    return sum(
        count_tokens(message.content if isinstance(message.content, str) else str(message.content))
        + _MESSAGE_OVERHEAD
        for message in messages
    )
