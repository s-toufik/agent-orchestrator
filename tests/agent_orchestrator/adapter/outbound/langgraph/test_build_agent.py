from langgraph.checkpoint.memory import InMemorySaver

from agent_orchestrator.adapter.outbound.langgraph.build_agent import build_agent
from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.langgraph_agent import LangGraphAgent
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    unpack_state,
)
from agent_orchestrator.adapter.outbound.llm.schema import ModelParameters
from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import (
    EchoTool,
    FakeLLM,
    accept,
    context,
    retry,
    text,
    tool_request,
)

THREAD = "thread-1"


class Harness:
    def __init__(
        self,
        logger,
        planner: FakeLLM | None = None,
        context_llm: FakeLLM | None = None,
        reflection: FakeLLM | None = None,
        summary: FakeLLM | None = None,
        max_iterations: int = 6,
        stream_answer: bool = False,
    ) -> None:
        self.planner = planner or FakeLLM()
        self.context = context_llm or FakeLLM()
        self.reflection = reflection or FakeLLM()
        self.graph = build_agent(
            act_llm=self.planner,  # ty: ignore[invalid-argument-type]
            context_llm=self.context,  # ty: ignore[invalid-argument-type]
            plan_llm=self.planner,  # ty: ignore[invalid-argument-type]
            reflection_llm=self.reflection,  # ty: ignore[invalid-argument-type]
            summary_llm=summary or FakeLLM(),  # ty: ignore[invalid-argument-type]
            tool_registry=ToolRegistry([EchoTool()]),
            model_parameters=ModelParameters(
                model_name="m",
                temperature=0.0,
                max_output_tokens=100,
                max_context_tokens=100_000,
                max_iterations=max_iterations,
                use_streaming=stream_answer,
            ),
            logger=logger,
            checkpointer=InMemorySaver(),
        )
        self.agent = LangGraphAgent({"m": self.graph})
        self.events: list[AgentMessageStream] = []

    async def say(self, message: str) -> AgentMessageStream:
        request = AgentRequest(message=message, model_name="m", request_id=THREAD)
        self.events = [event async for event in self.agent.stream(request)]
        return self.events[-1]

    def of_type(self, event_type: MessageStreamType) -> list[str]:
        return [event.content for event in self.events if event.type is event_type]

    async def state(self) -> AgentState:
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": THREAD}})
        return unpack_state(snapshot.values)


async def test_a_task_is_planned_approved_then_executed_with_tools(logger) -> None:
    harness = Harness(
        logger,
        context_llm=FakeLLM(
            parsed=[context(Intent.TASK, "say hi"), context(Intent.PLAN_APPROVAL, "yes")]
        ),
        planner=FakeLLM(
            replies=[
                text("## Plan\n1. echo hi (tool: echo)"),
                tool_request("echo", text="hi"),
                text("echo said hi"),
            ]
        ),
        reflection=FakeLLM(parsed=[accept()]),
    )

    proposal = await harness.say("say hi")

    assert proposal.metadata["outcome"] == "awaiting_approval"
    assert "1. echo hi" in proposal.content
    assert harness.planner.tool_bindings == []

    answer = await harness.say("yes")

    assert answer.content == "echo said hi"
    assert answer.metadata["outcome"] == "answered"
    state = await harness.state()
    assert state.pending_plan is None
    assert [m.role for m in state.transcript.messages] == [Role.USER, Role.ASSISTANT] * 2
    assert state.turn.scratch.of_role(Role.TOOL)[0].content == "echo:hi"
    assert "echo:hi" in harness.reflection.prompt()


async def test_a_follow_up_is_resolved_and_judged_against_the_conversation(logger) -> None:
    harness = Harness(
        logger,
        context_llm=FakeLLM(
            parsed=[
                context(Intent.DIRECT, "what is VaR"),
                context(Intent.CONTINUATION, "explain VaR for AI agents"),
            ]
        ),
        planner=FakeLLM(replies=[text("VaR is a loss quantile."), text("For agents, ...")]),
        reflection=FakeLLM(parsed=[accept()]),
    )

    first = await harness.say("what is VaR?")
    answer = await harness.say("I mean for agents")

    assert first.metadata["outcome"] == "answered"
    assert answer.content == "For agents, ..."
    assert len(harness.reflection.calls) == 1
    assert "Assistant: VaR is a loss quantile." in harness.context.prompt()
    judged = harness.reflection.prompt()
    assert "explain VaR for AI agents" in judged
    assert "User: I mean for agents" in judged
    assert harness.planner.tool_bindings == []


async def test_a_rejected_draft_is_retried_and_never_reaches_the_transcript(logger) -> None:
    harness = Harness(
        logger,
        context_llm=FakeLLM(parsed=[context(Intent.CONTINUATION, "hi")]),
        planner=FakeLLM(replies=[text("first try"), text("second try")]),
        reflection=FakeLLM(parsed=[retry("too short"), accept()]),
        stream_answer=True,
    )

    answer = await harness.say("hi")

    assert answer.content == "second try"
    assert harness.of_type(MessageStreamType.TOKEN) == ["second try"]
    assert harness.of_type(MessageStreamType.STATUS).count("Checking the answer") == 2
    assert harness.events[-1].type is MessageStreamType.FINAL
    state = await harness.state()
    assert [m.content for m in state.transcript.messages] == ["hi", "second try"]
    assert "too short" in harness.planner.prompt()


async def test_an_ambiguous_message_gets_a_clarifying_question(logger) -> None:
    harness = Harness(
        logger,
        context_llm=FakeLLM(
            parsed=[context(Intent.AMBIGUOUS, "?", clarification_question="Which desk?")]
        ),
    )

    answer = await harness.say("the numbers")

    assert answer.content == "Which desk?"
    assert answer.metadata["outcome"] == "clarification"
    assert harness.planner.calls == []


async def test_running_out_of_steps_ends_with_a_clear_message(logger) -> None:
    harness = Harness(
        logger,
        max_iterations=2,
        context_llm=FakeLLM(
            parsed=[context(Intent.TASK, "loop"), context(Intent.PLAN_APPROVAL, "yes")]
        ),
        planner=FakeLLM(replies=[text("## Plan\n1. echo"), *[tool_request("echo", text="x")] * 2]),
    )

    await harness.say("loop")
    answer = await harness.say("yes")

    assert "Running echo" in harness.of_type(MessageStreamType.STATUS)
    assert answer.metadata == {
        "iteration": "2",
        "max_iteration": "2",
        "outcome": "budget_exhausted",
    }
    assert answer.content
    assert harness.reflection.calls == []
