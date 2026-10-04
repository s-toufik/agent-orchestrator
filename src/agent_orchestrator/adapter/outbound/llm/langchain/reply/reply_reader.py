from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable


class ReplyReader(Protocol):
    async def read(self, model: Runnable, messages: Sequence[BaseMessage]) -> BaseMessage: ...
