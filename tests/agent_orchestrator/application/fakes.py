"""Fake ports for application tests: scripted answers, recorded calls, no LLM."""

import copy
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.exception.unknown_model_exception import UnknownModelException
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import FIRST_STEP, TurnPolicy

ECHO = ToolSpecification("echo", "Echoes its text.")


class Scripted:
    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[Conversation, Turn]] = []

    def _next(self, conversation: Conversation, turn: Turn) -> Any:
        # A snapshot: the live objects keep changing after the call.
        self.calls.append((copy.deepcopy(conversation), copy.deepcopy(turn)))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClassifier(Scripted):
    async def understand(self, conversation: Conversation, turn: Turn) -> Understanding | None:
        return self._next(conversation, turn)


class FakePlanner(Scripted):
    async def plan(self, conversation, turn, tools: list[ToolSpecification]) -> Plan:
        return Plan(task=turn.query, steps=self._next(conversation, turn))


class FakeActor(Scripted):
    async def act(self, conversation, turn, tools: list[ToolSpecification]) -> Draft:
        return self._next(conversation, turn)


class FakeReviewer(Scripted):
    async def review(self, conversation: Conversation, turn: Turn) -> Verdict | None:
        return self._next(conversation, turn)


class FakeSummarizer:
    def __init__(self, summary: str | Exception = "the summary") -> None:
        self._summary = summary
        self.calls: list[tuple[str, list[Message]]] = []

    async def summarize(self, turn: Turn, summary: str, messages: list[Message]) -> str:
        self.calls.append((summary, messages))
        if isinstance(self._summary, Exception):
            raise self._summary
        return self._summary


class FakeTokens:
    def __init__(self, tokens: int = 0) -> None:
        self.tokens = tokens

    def count(self, messages: list[Message]) -> int:
        return self.tokens


class FakeCatalog:
    def __init__(self, *tools: ToolSpecification) -> None:
        self._tools = list(tools)

    def tools(self) -> list[ToolSpecification]:
        return self._tools


class EchoExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def run(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call.id, call.name, f"{call.name}:{call.arguments.get('text', '')}")


class FakeModels:
    def __init__(self, settings: TurnSettings | None = None) -> None:
        self._settings = settings or TurnSettings()

    def turn_settings(self, model: str) -> TurnSettings:
        if model == "unknown":
            raise UnknownModelException(model)
        return self._settings


class ListEventStream:
    def __init__(self) -> None:
        self.events: list[TurnEvent] = []
        self.completed = False

    async def publish(self, event: TurnEvent) -> None:
        self.events.append(event)

    async def complete(self) -> None:
        self.completed = True

    async def __aiter__(self) -> AsyncIterator[TurnEvent]:
        for event in self.events:
            yield event


class InMemoryWorkflowRunner:
    """The workflow without LangGraph: the same steps and policy, conversations in a dict."""

    def __init__(self, steps: Sequence[StepHandler], policy: TurnPolicy) -> None:
        self._steps = {handler.step: handler for handler in steps}
        self._policy = policy
        self.conversations: dict[str, Conversation] = {}

    async def run(self, conversation_id: str, turn: Turn) -> AsyncIterator[TurnEvent]:
        conversation = self.conversations.setdefault(conversation_id, Conversation(conversation_id))
        step = FIRST_STEP
        while step is not Step.END:
            yield TurnEvent.entering(step, turn)
            await self._steps[step].run(conversation, turn)
            step = self._policy.next(step, turn)
        yield TurnEvent.finished(turn)
