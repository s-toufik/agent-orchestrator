from collections.abc import AsyncIterator, Callable, Coroutine, Hashable, Sequence
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.turn_state_codec import (
    GraphState,
    TurnStateCodec,
)
from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import FIRST_STEP, TurnPolicy

Node = Callable[[GraphState], Coroutine[Any, Any, GraphState]]


class LangGraphWorkflowRunner:

    def __init__(
        self,
        steps: Sequence[StepHandler],
        policy: TurnPolicy,
        logger: Logger,
        checkpointer: Any = None,
    ) -> None:
        self._policy = policy
        self._logger = logger
        self._codec = TurnStateCodec()
        self._graph = self._build(steps, checkpointer)

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

    # ------------------------------------------------------------------- graph
    def _build(self, steps: Sequence[StepHandler], checkpointer: Any) -> Any:
        graph = StateGraph(GraphState)  # ty: ignore[invalid-argument-type]
        targets: dict[Hashable, str] = {handler.step.value: handler.step.value for handler in steps}
        targets[Step.END.value] = END
        for handler in steps:
            graph.add_node(handler.step.value, self._node(handler))  # ty: ignore[invalid-argument-type]
            graph.add_conditional_edges(handler.step.value, self._route(handler.step), targets)
        graph.set_entry_point(FIRST_STEP.value)
        return graph.compile(checkpointer=checkpointer)

    def _node(self, handler: StepHandler) -> Node:
        async def node(state: GraphState) -> GraphState:
            conversation, turn = self._codec.decode(state)
            _emit(TurnEvent.entering(handler.step, turn))
            await handler.run(conversation, turn)
            return self._codec.encode(conversation, turn)

        return node

    def _route(self, done: Step) -> Callable[[GraphState], str]:
        def route(state: GraphState) -> str:
            return self._policy.next(done, self._codec.decode_turn(state)).value

        return route

    # ------------------------------------------------------------------- state
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


def _emit(event: TurnEvent) -> None:
    # Outside a streaming run (a node called directly) there is no writer.
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)
