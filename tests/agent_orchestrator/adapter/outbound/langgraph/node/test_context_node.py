from typing import cast

from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.context_node import ContextNode
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import FakeLLM, context

PLAN = Plan(task="count EQD positions", steps="1. query positions (tool: run_sql)")


def _state(message: str, pending_plan: Plan | None = None) -> AgentState:
    return AgentState(
        pending_plan=pending_plan,
        transcript=Conversation(
            messages=[
                ConversationMessage(role=Role.USER, content="explain VaR"),
                ConversationMessage(role=Role.ASSISTANT, content="VaR is a loss quantile."),
                ConversationMessage(role=Role.USER, content=message),
            ]
        ),
        turn=TurnState(user_message=message),
    )


async def _run(llm: FakeLLM, state: AgentState, logger) -> AgentState:
    node = ContextNode(cast(BaseChatModel, llm), ContextWindow(max_tokens=1_000), logger)
    return unpack_state(await node(pack_state(state)))


async def test_the_model_sees_the_history_the_pending_plan_and_the_raw_message(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.CONTINUATION, "explain VaR in more depth")])

    result = await _run(llm, _state("explain more", PLAN), logger)

    prompt = llm.prompt()
    assert "Assistant: VaR is a loss quantile." in prompt
    assert PLAN.steps in prompt
    assert "explain more" in prompt
    assert result.turn.context is not None
    assert result.turn.context.standalone_query == "explain VaR in more depth"


async def test_approval_moves_the_pending_plan_into_the_turn(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.PLAN_APPROVAL, "yes")])

    result = await _run(llm, _state("yes", PLAN), logger)

    assert result.turn.plan == PLAN
    assert result.pending_plan is None
    assert result.turn.context is not None
    assert result.turn.context.standalone_query == PLAN.task


async def test_approval_without_a_pending_plan_is_answered_directly(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.PLAN_APPROVAL, "yes")])

    result = await _run(llm, _state("yes"), logger)

    assert result.turn.plan is None
    assert result.turn.context is not None
    assert result.turn.context.intent is Intent.DIRECT


async def test_a_revision_keeps_the_pending_plan_for_the_planner(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.PLAN_REVISION, "count EQD and FX positions")])

    result = await _run(llm, _state("add FX too", PLAN), logger)

    assert result.pending_plan == PLAN
    assert result.turn.plan is None


async def test_moving_on_to_something_else_drops_the_pending_plan(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.TASK, "list users")])

    result = await _run(llm, _state("list users instead", PLAN), logger)

    assert result.pending_plan is None


async def test_ambiguous_without_a_question_falls_back_to_a_task(logger) -> None:
    llm = FakeLLM(parsed=[context(Intent.AMBIGUOUS, "hmm")])

    result = await _run(llm, _state("hmm"), logger)

    assert result.turn.context is not None
    assert result.turn.context.intent is Intent.TASK


async def test_an_unparseable_answer_falls_back_to_a_task_on_the_raw_message(logger) -> None:
    llm = FakeLLM(parsed=[None])

    result = await _run(llm, _state("count users"), logger)

    assert result.turn.context is not None
    assert result.turn.context.intent is Intent.TASK
    assert result.turn.context.standalone_query == "count users"
    assert logger.messages("warning")
