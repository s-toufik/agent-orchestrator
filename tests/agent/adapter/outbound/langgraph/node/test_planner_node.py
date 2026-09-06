from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.planner_node import PlannerNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state
from agent.application.port.outbound.tool_port import ToolPort
from agent.domain.model.tool_specification import ToolSpecification


class FakeAIMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class FakeBoundLLM:
    def __init__(self, response=None, chunks=None) -> None:
        self._response = response
        self._chunks = chunks or []

    async def ainvoke(self, messages):
        return self._response

    async def astream(self, messages):
        for chunk in self._chunks:
            yield chunk


class FakeLLM:
    def __init__(self, bound: FakeBoundLLM) -> None:
        self._bound = bound
        self.bound_with: list | None = None

    def bind_tools(self, tools):
        self.bound_with = tools
        return self._bound


class StubRegistry:
    def specifications(self) -> list[ToolSpecification]:
        return [
            ToolSpecification(name="run_sql", description="SQL.", parameters={"type": "object"})
        ]

    def get(self, name: str) -> ToolPort:
        raise AssertionError("not expected to be called from PlannerNode")


def _node(llm: FakeLLM, **kwargs: Any) -> PlannerNode:
    return PlannerNode(cast(BaseChatModel, llm), StubRegistry(), PromptService(), **kwargs)


async def test_a_plain_answer_with_no_tool_calls_becomes_the_final_answer() -> None:
    llm = FakeLLM(FakeBoundLLM(response=FakeAIMessage(content="the answer")))
    node = _node(llm)

    result = unpack_state(await node(pack_state(AgentState())))

    assert result.planner is not None
    assert result.planner.answer == "the answer"
    assert result.planner.tool_calls == []
    assert result.iteration == 1
    assert result.last_node == "planner"
    last = result.conversation.last()
    assert last is not None
    assert last.role is Role.ASSISTANT


async def test_a_tool_call_response_has_no_answer_and_generates_call_ids() -> None:
    raw_call = {"name": "run_sql", "args": {"query": "select 1"}}
    llm = FakeLLM(FakeBoundLLM(response=FakeAIMessage(content="", tool_calls=[raw_call])))
    node = _node(llm)

    result = unpack_state(await node(pack_state(AgentState())))

    assert result.planner is not None
    assert result.planner.answer is None
    assert len(result.planner.tool_calls) == 1
    call = result.planner.tool_calls[0]
    assert call.name == "run_sql"
    assert call.args == {"query": "select 1"}
    assert call.id.startswith("call_")


async def test_tools_are_bound_from_the_registry_specifications() -> None:
    llm = FakeLLM(FakeBoundLLM(response=FakeAIMessage(content="ok")))
    node = _node(llm)

    await node(pack_state(AgentState()))

    assert llm.bound_with == [
        {"name": "run_sql", "description": "SQL.", "parameters": {"type": "object"}}
    ]


async def test_streaming_calls_on_token_per_chunk_and_accumulates_the_final_content() -> None:
    chunks = [AIMessageChunk(content="Hel"), AIMessageChunk(content="lo")]
    llm = FakeLLM(FakeBoundLLM(chunks=chunks))
    received: list[str] = []

    async def on_token(value: str) -> None:
        received.append(value)

    node = _node(llm, on_token=on_token)

    result = unpack_state(await node(pack_state(AgentState())))

    assert received == ["Hel", "lo"]
    assert result.planner is not None
    assert result.planner.answer == "Hello"


async def test_streaming_skips_on_token_for_empty_chunks() -> None:
    chunks = [AIMessageChunk(content=""), AIMessageChunk(content="hi")]
    llm = FakeLLM(FakeBoundLLM(chunks=chunks))
    received: list[str] = []

    async def on_token(value: str) -> None:
        received.append(value)

    node = _node(llm, on_token=on_token)

    await node(pack_state(AgentState()))

    assert received == ["hi"]
