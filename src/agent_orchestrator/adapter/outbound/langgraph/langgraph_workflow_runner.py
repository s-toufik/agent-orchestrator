from collections.abc import AsyncIterator
from typing import Any

from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.state.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.turn.turn import Turn


class LangGraphWorkflowRunner:
    def __init__(self, graph: Any, logger: Logger) -> None:
        self._graph = graph
        self._logger = logger
        self._codec = TurnStateCodec()

    @property
    def graph(self) -> Any:
        return self._graph

    async def run(self, conversation_id: str, turn: Turn) -> AsyncIterator[TurnEvent]:
        config = self._config(conversation_id, turn)
        conversation = await self._load(conversation_id, config)

        final: GraphState | None = None
        async for mode, chunk in self._graph.astream(
            self._codec.encode(conversation, turn), config=config, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                yield chunk
            else:
                final = chunk
        if final is None:
            raise RuntimeError("The workflow finished without producing a state")
        yield TurnEvent.finished(self._codec.decode_turn(final))

    @staticmethod
    def _config(conversation_id: str, turn: Turn) -> dict[str, Any]:
        settings = turn.settings
        # One superstep per step: acting and tools alternate, each retry adds a review cycle.
        limit = 10 + 2 * settings.max_steps + 3 * settings.max_retries
        return {"configurable": {"thread_id": conversation_id}, "recursion_limit": limit}

    async def _load(self, conversation_id: str, config: dict[str, Any]) -> Conversation:
        snapshot = await self._graph.aget_state(config)
        if not snapshot.values:
            return Conversation(conversation_id)
        stored = self._codec.stored_conversation(snapshot.values)
        if stored is None:
            self._logger.warning(
                f"Conversation {conversation_id} has an unreadable state: new start"
            )
            return Conversation(conversation_id)
        return stored
