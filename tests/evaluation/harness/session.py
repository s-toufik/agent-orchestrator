import uuid

from agent_orchestrator.adapter.outbound.langgraph.langgraph_workflow_runner import (
    LangGraphWorkflowRunner,
)
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.event.turn_event import TurnEventKind
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_options import TurnOptions
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from tests.evaluation.harness.exchange import Exchange
from tests.evaluation.harness.route_coverage import RouteCoverage


class EvalSession:
    def __init__(
        self,
        runner: LangGraphWorkflowRunner,
        model: str,
        settings: TurnSettings,
        coverage: RouteCoverage,
    ) -> None:
        self._runner = runner
        self._model = model
        self._settings = settings
        self._coverage = coverage
        self._codec = TurnStateCodec()
        self.id = str(uuid.uuid4())

    async def say(self, message: str, auto_approve: bool = False) -> Exchange:
        turn = Turn(message, self._model, self._settings, TurnOptions(auto_approve=auto_approve))
        route = tuple(
            [
                event.step
                async for event in self._runner.run(self.id, turn)
                if event.kind is TurnEventKind.STEP_STARTED and event.step is not None
            ]
        )
        snapshot = await self._runner.graph.aget_state({"configurable": {"thread_id": self.id}})
        exchange = Exchange(
            message=message,
            route=route,
            turn=self._codec.decode_turn(snapshot.values),
            conversation=self._codec.stored_conversation(snapshot.values) or Conversation(self.id),
        )
        self._coverage.record(exchange.route)
        return exchange
