from typing import cast

from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.node.act_node import ActNode
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
    tool_request,
)


async def _run(
    llm: FakeLLM, state: AgentState, logger, registry: ToolRegistry | None = None
) -> AgentState:
    tools = registry if registry is not None else ToolRegistry([EchoTool()])
    node = ActNode(cast(BaseChatModel, llm), tools, ContextWindow(1_000), logger)
    return unpack_state(await node(pack_state(state)))


def _approved() -> AgentState:
    return AgentState(
        turn=TurnState(
            context=context(Intent.PLAN_APPROVAL, "say hi"),
            plan=Plan(task="say hi", steps="1. echo"),
        )
    )


async def test_tools_are_bound_and_calls_recorded_when_a_plan_was_approved(logger) -> None:
    llm = FakeLLM(replies=[tool_request("echo", text="hi")])

    result = await _run(llm, _approved(), logger)

    assert llm.tool_bindings[0][0]["name"] == "echo"
    draft = result.turn.scratch.last_assistant()
    assert draft is not None
    assert draft.tool_calls[0].name == "echo"
    assert draft.tool_calls[0].args == {"text": "hi"}
    assert draft.tool_calls[0].id.startswith("call_")
    assert result.turn.iteration == 1


async def test_without_an_approved_plan_tools_are_listed_but_not_bound(logger) -> None:
    llm = FakeLLM(replies=[text("VaR, in more depth")])
    state = AgentState(turn=TurnState(context=context(Intent.CONTINUATION, "explain VaR more")))

    result = await _run(llm, state, logger)

    assert llm.tool_bindings == []
    assert "- echo: Echoes." in llm.prompt()
    assert "explain VaR more" in llm.prompt()
    draft = result.turn.scratch.last_assistant()
    assert draft is not None
    assert draft.content == "VaR, in more depth"
    assert result.transcript.messages == []


async def test_an_agent_without_tools_still_acts_and_binds_nothing(logger) -> None:
    llm = FakeLLM(replies=[text("done without tools")])

    result = await _run(llm, _approved(), logger, registry=ToolRegistry([]))

    assert llm.tool_bindings == []
    draft = result.turn.scratch.last_assistant()
    assert draft is not None
    assert draft.content == "done without tools"
