from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from mcp import ClientSession


@runtime_checkable
class McpSessionFactory(Protocol):
    def session(self) -> AbstractAsyncContextManager[ClientSession]: ...
