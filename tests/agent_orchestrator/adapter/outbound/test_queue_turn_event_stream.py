from agent_orchestrator.adapter.outbound.event.queue_turn_event_stream import QueueTurnEventStream
from agent_orchestrator.domain.event.turn_event import TurnEvent


async def test_events_are_read_in_order_until_the_stream_completes() -> None:
    stream = QueueTurnEventStream()
    await stream.publish(TurnEvent.failed("a"))
    await stream.publish(TurnEvent.failed("b"))
    await stream.complete()

    assert [event.error async for event in stream] == ["a", "b"]
