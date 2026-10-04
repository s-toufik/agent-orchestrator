from collections.abc import Callable

from langgraph.graph import END

from agent_orchestrator.adapter.outbound.langgraph.state.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy


def node_name(step: Step) -> str:
    return END if step is Step.END else step.value


class PolicyRouter:
    def __init__(self, policy: TurnPolicy, codec: TurnStateCodec) -> None:
        self._policy = policy
        self._codec = codec

    def after(self, step: Step) -> Callable[[GraphState], str]:
        def route(state: GraphState) -> str:
            return node_name(self._policy.next(step, self._codec.decode_turn(state)))

        return route
