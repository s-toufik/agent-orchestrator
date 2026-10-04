import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole


class FakeLLM:
    def __init__(self, replies: list[Any] | None = None, parsed: list[Any] | None = None) -> None:
        self._replies = list(replies or [])
        self._parsed = list(parsed or [])
        self.calls: list[list[BaseMessage]] = []
        self.tool_bindings: list[list[dict]] = []

    def bind_tools(self, tools: list[dict]) -> FakeLLM:
        self.tool_bindings.append(tools)
        return self

    def with_structured_output(self, schema: type, include_raw: bool = True) -> _Structured:
        assert include_raw is True
        return _Structured(self)

    async def ainvoke(self, messages: list[BaseMessage]) -> Any:
        self.calls.append(messages)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        reply = await self.ainvoke(messages)
        for piece in re.findall(r"\S+\s*", reply.content):
            yield AIMessageChunk(content=piece)
        for index, call in enumerate(reply.tool_calls):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "id": call["id"],
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "index": index,
                    }
                ],
            )

    def prompt(self, call: int = -1) -> str:
        return "\n".join(str(message.content) for message in self.calls[call])


class _Structured:
    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm

    async def ainvoke(self, messages: list[BaseMessage]) -> dict:
        self._llm.calls.append(messages)
        return {"parsed": self._llm._parsed.pop(0)}


class FakeChatModels:
    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm
        self.requested: list[tuple[AgentRole, str]] = []

    def for_role(self, role: AgentRole, model: str) -> Any:
        self.requested.append((role, model))
        return self.llm


def text(content: str) -> AIMessage:
    return AIMessage(content=content)


def tool_request(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": "x", "name": name, "args": args}])
