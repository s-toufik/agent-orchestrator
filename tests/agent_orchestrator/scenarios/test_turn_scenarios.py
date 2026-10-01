"""Whole turns, on both runners: the plain loop and the graph LangGraph generates from the
policy. Both must give the same answers, the same events and the same stored conversation."""

from collections.abc import AsyncIterator
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agent_orchestrator.adapter.outbound.langgraph.graph.agent_graph_builder import (
    AgentGraphBuilder,
)
from agent_orchestrator.adapter.outbound.langgraph.graph.policy_router import PolicyRouter
from agent_orchestrator.adapter.outbound.langgraph.langgraph_workflow_runner import (
    LangGraphWorkflowRunner,
)
from agent_orchestrator.adapter.outbound.langgraph.node.agent_nodes import agent_nodes
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.application.step.act_step import ActStep
from agent_orchestrator.application.step.clarify_step import ClarifyStep
from agent_orchestrator.application.step.feedback_step import FeedbackStep
from agent_orchestrator.application.step.finish_step import FinishStep
from agent_orchestrator.application.step.plan_step import PlanStep
from agent_orchestrator.application.step.review_step import ReviewStep
from agent_orchestrator.application.step.run_tools_step import RunToolsStep
from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.application.step.summarize_step import SummarizeStep
from agent_orchestrator.application.step.understand_step import UnderstandStep
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Speaker
from agent_orchestrator.domain.event.turn_event import TurnEvent, TurnEventKind
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.turn.answer import BUDGET_EXHAUSTED
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_options import TurnOptions
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy
from tests.agent_orchestrator.application.fakes import (
    ECHO,
    EchoExecutor,
    FakeActor,
    FakeCatalog,
    FakeClassifier,
    FakePlanner,
    FakeReviewer,
    FakeSummarizer,
    FakeTokens,
    InMemoryWorkflowRunner,
)
from tests.agent_orchestrator.domain.builders import accept, retry

CONVERSATION = "c1"


class Agent:
    """The steps wired on fake ports, run by the runner under test."""

    def __init__(
        self,
        runner_kind: str,
        logger,
        understandings: list[Understanding] | None = None,
        plans: list[str] | None = None,
        drafts: list[Draft] | None = None,
        verdicts: list | None = None,
    ) -> None:
        self.classifier = FakeClassifier(*(understandings or []))
        self.planner = FakePlanner(*(plans or []))
        self.actor = FakeActor(*(drafts or []))
        self.reviewer = FakeReviewer(*(verdicts or []))
        self.executor = EchoExecutor()
        catalog = FakeCatalog(ECHO)
        self.steps = steps = [
            UnderstandStep(self.classifier, logger),
            PlanStep(self.planner, catalog),
            ClarifyStep(),
            ActStep(self.actor, catalog),
            RunToolsStep(self.executor),
            ReviewStep(self.reviewer, logger),
            FeedbackStep(logger),
            FinishStep(),
            SummarizeStep(FakeSummarizer(), FakeTokens(0), logger),
        ]
        policy = TurnPolicy()
        self.runner = (
            InMemoryWorkflowRunner(steps, policy)
            if runner_kind == "memory"
            else _langgraph(steps, logger, InMemorySaver())
        )
        self.events: list[TurnEvent] = []

    async def say(
        self, message: str, settings: TurnSettings | None = None, auto_approve: bool = False
    ) -> TurnEvent:
        turn = Turn(
            message,
            "m",
            settings or TurnSettings(max_steps=6),
            options=TurnOptions(auto_approve=auto_approve),
        )
        self.events = [event async for event in self._run(turn)]
        return self.events[-1]

    async def _run(self, turn: Turn) -> AsyncIterator[TurnEvent]:
        async for event in self.runner.run(CONVERSATION, turn):
            yield event

    async def stored(self) -> Conversation:
        if isinstance(self.runner, InMemoryWorkflowRunner):
            return self.runner.conversations[CONVERSATION]
        snapshot = await self.runner.graph.aget_state({"configurable": {"thread_id": CONVERSATION}})
        stored = TurnStateCodec().stored_conversation(snapshot.values)
        assert stored is not None
        return stored

    def entered(self) -> list[Step | None]:
        return [e.step for e in self.events if e.kind is TurnEventKind.STEP_STARTED]


RUNNERS = pytest.mark.parametrize("runner_kind", ["memory", "langgraph"])


def _langgraph(steps: list[StepHandler], logger, checkpointer) -> LangGraphWorkflowRunner:
    codec, policy = TurnStateCodec(), TurnPolicy()
    builder = AgentGraphBuilder(agent_nodes(steps, codec), policy, PolicyRouter(policy, codec))
    return LangGraphWorkflowRunner(builder.build(checkpointer), logger)


def _understood(intent: Intent, query: str = "the query", **fields) -> Understanding:
    return Understanding(intent=intent, query=query, **fields)


def _calls(*names: str) -> Draft:
    return Draft("", tuple(ToolCall(f"call_{i}", n, {"text": "hi"}) for i, n in enumerate(names)))


@RUNNERS
async def test_a_task_is_planned_approved_then_executed_with_tools(runner_kind, logger) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[
            _understood(Intent.TASK, "say hi"),
            _understood(Intent.PLAN_APPROVAL, "yes"),
        ],
        plans=["## Plan\n1. echo hi (tool: echo)"],
        drafts=[_calls("echo"), Draft("echo said hi")],
        verdicts=[accept()],
    )

    proposal = await agent.say("say hi")

    assert proposal.answer is not None
    assert proposal.answer.outcome is Outcome.AWAITING_APPROVAL
    assert "1. echo hi" in proposal.answer.text
    assert agent.entered() == [Step.UNDERSTAND, Step.PLAN, Step.FINISH, Step.SUMMARIZE]

    answer = await agent.say("yes")

    assert answer.answer is not None
    assert (answer.answer.text, answer.answer.outcome) == ("echo said hi", Outcome.ANSWERED)
    assert (answer.steps_taken, answer.max_steps) == (2, 6)
    assert agent.entered() == [
        Step.UNDERSTAND,
        Step.ACT,
        Step.RUN_TOOLS,
        Step.ACT,
        Step.REVIEW,
        Step.FINISH,
        Step.SUMMARIZE,
    ]
    assert [e.tools for e in agent.events if e.step is Step.RUN_TOOLS] == [("echo",)]
    executed: Turn = agent.actor.calls[-1][1]
    assert executed.plan is not None and executed.query == "say hi"
    reviewed: Turn = agent.reviewer.calls[0][1]
    assert [result.output for result in reviewed.evidence] == ["echo:hi"]
    # The plan proposed in the first turn was stored, then taken by the approval.
    assert agent.classifier.calls[-1][0].pending_plan is not None
    stored = await agent.stored()
    assert [m.speaker for m in stored.messages] == [Speaker.USER, Speaker.ASSISTANT] * 2
    assert stored.pending_plan is None


@RUNNERS
async def test_a_follow_up_is_understood_against_the_conversation(runner_kind, logger) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[
            _understood(Intent.DIRECT, "what is VaR"),
            _understood(Intent.CONTINUATION, "explain VaR for AI agents"),
        ],
        drafts=[Draft("VaR is a loss quantile."), Draft("For agents, ...")],
        verdicts=[accept()],
    )

    first = await agent.say("what is VaR?")
    answer = await agent.say("I mean for agents")

    assert first.answer is not None and first.answer.outcome is Outcome.ANSWERED
    assert answer.answer is not None and answer.answer.text == "For agents, ..."
    assert len(agent.reviewer.calls) == 1  # the plain direct answer was not reviewed
    history = agent.classifier.calls[-1][0].messages
    assert [m.text for m in history] == ["what is VaR?", "VaR is a loss quantile."]


@RUNNERS
async def test_a_rejected_draft_is_retried_and_only_the_last_one_is_kept(
    runner_kind, logger
) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[_understood(Intent.CONTINUATION, "hi")],
        drafts=[Draft("first try"), Draft("second try")],
        verdicts=[retry("too short"), accept()],
    )

    answer = await agent.say("hi")

    assert answer.answer is not None and answer.answer.text == "second try"
    assert agent.entered().count(Step.REVIEW) == 2
    retried: Turn = agent.actor.calls[-1][1]
    assert retried.work[-1].critique == "too short"
    stored = await agent.stored()
    assert [m.text for m in stored.messages] == ["hi", "second try"]


@RUNNERS
async def test_an_ambiguous_message_gets_a_clarifying_question(runner_kind, logger) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[_understood(Intent.AMBIGUOUS, "?", clarification_question="Which desk?")],
    )

    answer = await agent.say("the numbers")

    assert answer.answer is not None
    assert (answer.answer.text, answer.answer.outcome) == ("Which desk?", Outcome.CLARIFICATION)
    assert agent.actor.calls == []


@RUNNERS
async def test_running_out_of_steps_ends_with_a_clear_message(runner_kind, logger) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[_understood(Intent.TASK, "loop"), _understood(Intent.PLAN_APPROVAL, "yes")],
        plans=["## Plan\n1. echo"],
        drafts=[_calls("echo"), _calls("echo")],
    )

    await agent.say("loop")
    answer = await agent.say("yes", TurnSettings(max_steps=2))

    assert answer.answer is not None
    assert (answer.answer.text, answer.answer.outcome) == (
        BUDGET_EXHAUSTED,
        Outcome.BUDGET_EXHAUSTED,
    )
    assert (answer.steps_taken, answer.max_steps) == (2, 2)
    assert agent.reviewer.calls == []


async def test_a_thread_stored_in_an_older_format_starts_a_new_conversation(logger) -> None:
    saver = InMemorySaver()
    old = StateGraph(_OldState)  # ty: ignore[invalid-argument-type]
    old.add_node("ingest", lambda state: {"state": {"transcript": ["older layout"]}})
    old.set_entry_point("ingest")
    old.add_edge("ingest", END)
    config = {"configurable": {"thread_id": CONVERSATION}}
    await old.compile(checkpointer=saver).ainvoke({"state": {}}, config)
    agent = Agent("langgraph", logger, [_understood(Intent.DIRECT, "hello")], drafts=[Draft("hi")])
    agent.runner = _langgraph(agent.steps, logger, saver)

    answer = await agent.say("hello")

    assert answer.answer is not None and answer.answer.text == "hi"
    assert agent.classifier.calls[0][0].messages == []


class _OldState(TypedDict):
    state: dict


def test_an_unreadable_stored_conversation_is_ignored() -> None:
    codec = TurnStateCodec()

    assert codec.stored_conversation({}) is None
    assert codec.stored_conversation({"conversation": {"transcript": []}}) is None


@RUNNERS
async def test_a_request_misread_as_direct_gets_a_plan_in_the_same_turn(
    runner_kind, logger
) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[
            _understood(Intent.DIRECT, "explain change.md"),
            _understood(Intent.PLAN_APPROVAL, "yes"),
        ],
        plans=["## Plan\n1. read change.md (tool: echo)"],
        drafts=[
            Draft.asking_for_plan("read change.md"),
            _calls("echo"),
            Draft("It lists changes."),
        ],
        verdicts=[accept()],
    )

    proposal = await agent.say("explain change.md")

    assert proposal.answer is not None
    assert proposal.answer.outcome is Outcome.AWAITING_APPROVAL
    assert agent.entered() == [Step.UNDERSTAND, Step.ACT, Step.PLAN, Step.FINISH, Step.SUMMARIZE]
    assert (await agent.stored()).pending_plan is not None

    answer = await agent.say("yes")

    assert answer.answer is not None and answer.answer.text == "It lists changes."
    assert agent.executor.calls[0].name == "echo"


@RUNNERS
async def test_with_auto_approve_a_task_is_planned_and_carried_out_in_one_turn(
    runner_kind, logger
) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[_understood(Intent.TASK, "say hi")],
        plans=["## Plan\n1. echo hi (tool: echo)"],
        drafts=[_calls("echo"), Draft("echo said hi")],
        verdicts=[accept()],
    )

    answer = await agent.say("say hi", auto_approve=True)

    assert answer.answer is not None
    assert (answer.answer.text, answer.answer.outcome) == ("echo said hi", Outcome.ANSWERED)
    assert agent.entered() == [
        Step.UNDERSTAND,
        Step.PLAN,
        Step.ACT,
        Step.RUN_TOOLS,
        Step.ACT,
        Step.REVIEW,
        Step.FINISH,
        Step.SUMMARIZE,
    ]
    assert (await agent.stored()).pending_plan is None


@RUNNERS
async def test_with_auto_approve_a_request_misread_as_direct_still_ends_with_the_tool_result(
    runner_kind, logger
) -> None:
    agent = Agent(
        runner_kind,
        logger,
        understandings=[_understood(Intent.DIRECT, "explain change.md")],
        plans=["## Plan\n1. read change.md (tool: echo)"],
        drafts=[
            Draft.asking_for_plan("read change.md"),
            _calls("echo"),
            Draft("It lists changes."),
        ],
        verdicts=[accept()],
    )

    answer = await agent.say("explain change.md", auto_approve=True)

    assert answer.answer is not None and answer.answer.text == "It lists changes."
    assert agent.executor.calls[0].name == "echo"
    executed: Turn = agent.actor.calls[1][1]
    assert executed.plan is not None and executed.work == []
