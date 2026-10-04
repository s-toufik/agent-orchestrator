from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent_orchestrator.adapter.outbound.langgraph.langgraph_event_publisher import (
    LangGraphEventPublisher,
)
from agent_orchestrator.domain.event.turn_event import TurnEvent

PUBLISHER = LangGraphEventPublisher()


class _State(TypedDict):
    done: bool


async def _speak(state: _State) -> _State:
    await PUBLISHER.publish(TurnEvent.answer_delta("hi"))
    return {"done": True}


async def test_an_event_published_inside_a_run_reaches_its_stream() -> None:
    graph = StateGraph(_State)
    graph.add_node("speak", _speak)
    graph.add_edge(START, "speak")
    graph.add_edge("speak", END)

    events = [
        chunk async for chunk in graph.compile().astream({"done": False}, stream_mode="custom")
    ]

    assert events == [TurnEvent.answer_delta("hi")]


async def test_an_event_published_outside_a_run_goes_nowhere() -> None:
    await PUBLISHER.publish(TurnEvent.answer_delta("hi"))
