from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from claude_agent_sdk import ClaudeAgentOptions, Message


@runtime_checkable
class QueryPort(Protocol):
    def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Message]: ...
