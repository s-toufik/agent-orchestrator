from langgraph.config import get_stream_writer

from agent_orchestrator.domain.event.turn_event import TurnEvent


class LangGraphEventPublisher:
    async def publish(self, event: TurnEvent) -> None:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        writer(event)
