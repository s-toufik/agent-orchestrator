from typing import cast

from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.plan_node import PlanNode
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import (
    EchoTool,
    FakeLLM,
    context,
    text,
)


async def _run(llm: FakeLLM, state: AgentState, logger) -> AgentState:
    node = PlanNode(
        cast(BaseChatModel, llm), ToolRegistry([EchoTool()]), ContextWindow(1_000), logger
    )
    return unpack_state(await node(pack_state(state)))


async def test_the_plan_is_stored_and_shown_to_the_user_for_approval(logger) -> None:
    llm = FakeLLM(replies=[text("## Plan\n1. echo hi (tool: echo)")])
    state = AgentState(turn=TurnState(context=context(Intent.TASK, "say hi")))

    result = await _run(llm, state, logger)

    assert result.pending_plan == Plan(task="say hi", steps="## Plan\n1. echo hi (tool: echo)")
    assert result.turn.outcome is TurnOutcome.AWAITING_APPROVAL
    assert result.turn.answer is not None
    assert result.turn.answer.startswith("## Plan")
    assert "yes" in result.turn.answer
    assert "- echo: Echoes." in llm.prompt()
    assert llm.tool_bindings == []


async def test_a_revision_shows_the_previous_plan_to_the_planner(logger) -> None:
    llm = FakeLLM(replies=[text("## Plan\n1. new")])
    state = AgentState(
        pending_plan=Plan(task="say hi", steps="1. old step"),
        turn=TurnState(context=context(Intent.PLAN_REVISION, "say hi twice")),
    )

    result = await _run(llm, state, logger)

    assert "1. old step" in llm.prompt()
    assert result.pending_plan is not None
    assert result.pending_plan.task == "say hi twice"
