from typing import cast

from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.reflect_node import ReflectNode
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import (
    FakeLLM,
    context,
    retry,
)


def _state(turn_context: TurnContext) -> AgentState:
    return AgentState(
        transcript=Conversation(
            messages=[
                ConversationMessage(role=Role.USER, content="explain VaR"),
                ConversationMessage(role=Role.ASSISTANT, content="PREVIOUS ANSWER"),
                ConversationMessage(role=Role.USER, content="continue"),
            ]
        ),
        turn=TurnState(
            context=turn_context,
            plan=Plan(task="t", steps="APPROVED PLAN"),
            scratch=Conversation(
                messages=[
                    ConversationMessage(role=Role.TOOL, content="EVIDENCE" + "x" * 50),
                    ConversationMessage(role=Role.ASSISTANT, content="DRAFT"),
                ]
            ),
        ),
    )


async def _run(llm: FakeLLM, state: AgentState, logger) -> AgentState:
    node = ReflectNode(cast(BaseChatModel, llm), logger, max_evidence_chars=10)
    return unpack_state(await node(pack_state(state)))


async def test_the_judge_sees_the_conversation_request_plan_evidence_and_draft(logger) -> None:
    llm = FakeLLM(parsed=[retry("missing detail")])
    turn_context = context(
        Intent.CONTINUATION, "continue explaining VaR", success_criteria=["covers CVaR"]
    )

    result = await _run(llm, _state(turn_context), logger)

    prompt = llm.prompt()
    for expected in (
        "User: explain VaR",
        "Assistant: PREVIOUS ANSWER",
        "User: continue",
        "continue explaining VaR",
        "covers CVaR",
        "APPROVED PLAN",
        "DRAFT",
    ):
        assert expected in prompt
    assert "EVIDENCExx [...]" in prompt
    assert result.turn.reflection == retry("missing detail")


async def test_the_conversation_reaches_the_judge_even_when_the_request_was_misread(
    logger,
) -> None:
    llm = FakeLLM(parsed=[retry()])

    await _run(llm, _state(context(Intent.DIRECT, "continue")), logger)

    assert "Assistant: PREVIOUS ANSWER" in llm.prompt()


async def test_an_unparseable_verdict_accepts_the_draft_and_warns(logger) -> None:
    llm = FakeLLM(parsed=[None])

    result = await _run(llm, _state(context(Intent.TASK)), logger)

    assert result.turn.reflection is None
    assert logger.messages("warning")
