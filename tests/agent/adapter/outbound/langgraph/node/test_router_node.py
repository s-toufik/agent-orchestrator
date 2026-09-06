from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.node.router_node import RouterNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state

router = RouterNode()


async def test_max_iterations_always_wins_regardless_of_last_node() -> None:
    state = AgentState(last_node="planner", iteration=5, max_iterations=5)

    assert await router(pack_state(state)) == "final"


async def test_planner_with_tool_calls_routes_to_executor() -> None:
    state = AgentState(
        last_node="planner",
        planner=PlannerDecision(tool_calls=[ToolCall(id="1", name="t", args={})]),
    )

    assert await router(pack_state(state)) == "executor"


async def test_planner_without_tool_calls_routes_to_reflection() -> None:
    state = AgentState(last_node="planner", planner=PlannerDecision(tool_calls=[], answer="done"))

    assert await router(pack_state(state)) == "reflection"


async def test_planner_with_no_decision_at_all_routes_to_reflection() -> None:
    state = AgentState(last_node="planner", planner=None)

    assert await router(pack_state(state)) == "reflection"


async def test_reflection_asking_for_a_retry_routes_to_feedback() -> None:
    state = AgentState(
        last_node="reflection",
        reflection=ReflectionDecision(action=ReflectionAction.RETRY, critique="x"),
    )

    assert await router(pack_state(state)) == "feedback"


async def test_reflection_accepting_routes_to_final() -> None:
    state = AgentState(
        last_node="reflection",
        reflection=ReflectionDecision(action=ReflectionAction.ACCEPT, critique="x"),
    )

    assert await router(pack_state(state)) == "final"


async def test_reflection_with_no_decision_at_all_routes_to_final() -> None:
    state = AgentState(last_node="reflection", reflection=None)

    assert await router(pack_state(state)) == "final"


async def test_any_other_last_node_routes_to_final() -> None:
    state = AgentState(last_node="memory")

    assert await router(pack_state(state)) == "final"
