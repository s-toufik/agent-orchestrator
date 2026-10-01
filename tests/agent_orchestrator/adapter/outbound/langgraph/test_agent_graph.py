import pytest
from langgraph.graph import END, START

from agent_orchestrator.adapter.outbound.langgraph.graph.agent_graph_builder import (
    AgentGraphBuilder,
)
from agent_orchestrator.adapter.outbound.langgraph.graph.policy_router import (
    PolicyRouter,
    node_name,
)
from agent_orchestrator.adapter.outbound.langgraph.node.act_node import ActNode
from agent_orchestrator.adapter.outbound.langgraph.node.agent_nodes import NODE_TYPES, agent_nodes
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.application.step.clarify_step import ClarifyStep
from agent_orchestrator.application.step.finish_step import FinishStep
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy

CODEC = TurnStateCodec()
POLICY = TurnPolicy()


class RecordingHandler:
    def __init__(self, step: Step) -> None:
        self.step = step
        self.runs: list[Turn] = []

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        self.runs.append(turn)
        turn.drafted(Draft("done"))


def _handlers() -> list[RecordingHandler]:
    return [RecordingHandler(step) for step in Step if step is not Step.END]


def test_there_is_one_node_class_per_step() -> None:
    assert [node_type.step for node_type in NODE_TYPES] == [s for s in Step if s is not Step.END]


def test_a_node_refuses_the_handler_of_another_step() -> None:
    with pytest.raises(ValueError):
        ActNode(ClarifyStep(), CODEC)


async def test_a_node_delegates_to_its_handler_and_hands_back_the_state() -> None:
    handler = RecordingHandler(Step.ACT)
    state = CODEC.encode(Conversation("c1"), Turn("hi", "m"))

    new_state = await ActNode(handler, CODEC)(state)

    assert len(handler.runs) == 1
    assert CODEC.decode_turn(new_state).last_draft == Draft("done")


def _edges() -> dict[str, set[str]]:
    nodes = agent_nodes(_handlers(), CODEC)  # ty: ignore[invalid-argument-type]
    graph = AgentGraphBuilder(nodes, POLICY, PolicyRouter(POLICY, CODEC)).build().get_graph()
    edges: dict[str, set[str]] = {}
    for edge in graph.edges:
        edges.setdefault(edge.source, set()).add(edge.target)
    return edges


def test_the_graph_edges_are_exactly_the_policy_transitions() -> None:
    edges = _edges()

    assert edges.pop(START) == {"understand"}
    assert edges == {
        step.value: {node_name(target) for target in POLICY.targets(step)}
        for step in Step
        if step is not Step.END
    }
    assert sum(len(targets) for targets in edges.values()) == len(
        {(t.source, t.target) for t in POLICY.TRANSITIONS}
    )


def test_the_end_of_the_policy_is_the_end_of_the_graph() -> None:
    assert _edges()["summarize"] == {END}


async def test_the_router_asks_the_policy_with_the_domain_turn() -> None:
    turn = Turn("the numbers", "m")
    turn.understood(Understanding(Intent.AMBIGUOUS, "?", clarification_question="Which?"))
    state = CODEC.encode(Conversation("c1"), turn)
    router = PolicyRouter(POLICY, CODEC)

    assert router.after(Step.UNDERSTAND)(state) == "clarify"
    turn.finish(Answer.clarification("Which?"))
    await FinishStep().run(Conversation("c1"), turn)
    assert router.after(Step.SUMMARIZE)(CODEC.encode(Conversation("c1"), turn)) == END
