from collections.abc import Sequence

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable

from agent_orchestrator.adapter.outbound.llm.langchain.model_call import model_errors
from agent_orchestrator.application.port.outbound.turn_event_publisher import TurnEventPublisher
from agent_orchestrator.domain.event.turn_event import TurnEvent


class StreamedReply:
    def __init__(self, events: TurnEventPublisher) -> None:
        self._events = events

    async def read(self, model: Runnable, messages: Sequence[BaseMessage]) -> BaseMessage:
        reply = AIMessageChunk(content="")
        with model_errors():
            async for chunk in model.astream(list(messages)):
                reply += chunk
                if isinstance(chunk.content, str) and chunk.content:
                    await self._events.publish(TurnEvent.answer_delta(chunk.content))
        return reply
