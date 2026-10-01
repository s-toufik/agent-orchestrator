from typing import ClassVar

from langgraph.config import get_stream_writer

from agent_orchestrator.adapter.outbound.langgraph.state.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class AgentNode:
    """A graph node running one step. The work is the step handler's; a subclass overrides
    `work` only to use a LangGraph feature (interrupt, Send, a subgraph...)."""

    step: ClassVar[Step]

    def __init__(self, handler: StepHandler, codec: TurnStateCodec) -> None:
        if handler.step is not self.step:
            raise ValueError(f"{type(self).__name__} runs '{self.step}', not '{handler.step}'")
        self._handler = handler
        self._codec = codec

    @property
    def name(self) -> str:
        return self.step.value

    async def __call__(self, state: GraphState) -> GraphState:
        conversation, turn = self._codec.decode(state)
        _emit(TurnEvent.entering(self.step, turn))
        await self.work(conversation, turn)
        return self._codec.encode(conversation, turn)

    async def work(self, conversation: Conversation, turn: Turn) -> None:
        await self._handler.run(conversation, turn)


def _emit(event: TurnEvent) -> None:
    # Outside a streaming run (a node called directly) there is no writer.
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)
