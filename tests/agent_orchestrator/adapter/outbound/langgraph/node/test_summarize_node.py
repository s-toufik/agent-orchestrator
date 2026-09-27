from typing import cast

from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.summarize_node import SummarizeNode
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import FakeLLM, text


def _state(turns: int) -> AgentState:
    messages: list[ConversationMessage] = []
    for i in range(turns):
        messages.append(ConversationMessage(role=Role.USER, content=f"question {i} " * 10))
        messages.append(ConversationMessage(role=Role.ASSISTANT, content=f"answer {i} " * 10))
    return AgentState(summary="old summary", transcript=Conversation(messages=messages))


async def _run(llm: FakeLLM, state: AgentState, logger, trigger: int = 50) -> AgentState:
    node = SummarizeNode(cast(BaseChatModel, llm), logger, trigger_tokens=trigger)
    return unpack_state(await node(pack_state(state)))


async def test_nothing_happens_under_the_trigger(logger) -> None:
    llm = FakeLLM()

    result = await _run(llm, _state(3), logger, trigger=10_000)

    assert len(result.transcript.messages) == 6
    assert llm.calls == []


async def test_older_turns_are_folded_into_the_summary(logger) -> None:
    llm = FakeLLM(replies=[text("new summary")])

    result = await _run(llm, _state(3), logger)

    assert result.summary == "new summary"
    assert [m.content.split()[:2] for m in result.transcript.messages] == [
        ["question", "1"],
        ["answer", "1"],
        ["question", "2"],
        ["answer", "2"],
    ]
    assert "old summary" in llm.prompt()
    assert "question 0" in llm.prompt()


async def test_a_failed_summary_keeps_the_transcript_and_warns(logger) -> None:
    llm = FakeLLM(replies=[RuntimeError("down")])

    result = await _run(llm, _state(3), logger)

    assert len(result.transcript.messages) == 6
    assert result.summary == "old summary"
    assert logger.messages("warning")
