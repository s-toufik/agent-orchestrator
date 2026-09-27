from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_context import TurnContext
from agent_orchestrator.domain.model.tool_invocation import ToolInvocation
from agent_orchestrator.domain.model.tool_outcome import ToolOutcome
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


class FakeLLM:
    def __init__(self, replies: list[Any] | None = None, parsed: list[Any] | None = None) -> None:
        self._replies = list(replies or [])
        self._parsed = list(parsed or [])
        self.calls: list[list[BaseMessage]] = []
        self.tool_bindings: list[list[dict]] = []

    def bind_tools(self, tools: list[dict]) -> FakeLLM:
        self.tool_bindings.append(tools)
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> Any:
        self.calls.append(messages)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def with_structured_output(self, schema: type, include_raw: bool = True) -> _Structured:
        assert include_raw is True
        return _Structured(self)

    def prompt(self, call: int = -1) -> str:
        return "\n".join(str(message.content) for message in self.calls[call])


class _Structured:
    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self._llm.calls.append(messages)
        return {"parsed": self._llm._parsed.pop(0), "raw": None, "parsing_error": None}


def text(content: str) -> AIMessage:
    return AIMessage(content=content)


def tool_request(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": "x", "name": name, "args": args}])


def context(intent: Intent, query: str = "the query", **fields: Any) -> TurnContext:
    return TurnContext(intent=intent, standalone_query=query, **fields)


def accept() -> ReflectionDecision:
    return ReflectionDecision(action=ReflectionAction.ACCEPT, critique="fine")


def retry(critique: str = "too short") -> ReflectionDecision:
    return ReflectionDecision(action=ReflectionAction.RETRY, critique=critique)


class EchoTool:
    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(name="echo", description="Echoes.", parameters={"type": "object"})

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome:
        return ToolOutcome(
            invocation_id=invocation.id,
            tool_name="echo",
            output=f"echo:{invocation.arguments.get('text')}",
        )
