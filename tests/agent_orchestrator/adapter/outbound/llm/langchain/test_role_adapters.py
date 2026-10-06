from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_orchestrator.adapter.outbound.llm.langchain.actor import LangChainActor
from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.dto import (
    PlanDto,
    PlanStepDto,
    UnderstandingDto,
    VerdictDto,
)
from agent_orchestrator.adapter.outbound.llm.langchain.intent_classifier import (
    LangChainIntentClassifier,
)
from agent_orchestrator.adapter.outbound.llm.langchain.plan_request import REQUEST_PLAN_TOOL
from agent_orchestrator.adapter.outbound.llm.langchain.planner import LangChainPlanner
from agent_orchestrator.adapter.outbound.llm.langchain.reply.reply_reader import ReplyReader
from agent_orchestrator.adapter.outbound.llm.langchain.reply.streamed_reply import StreamedReply
from agent_orchestrator.adapter.outbound.llm.langchain.reply.whole_reply import WholeReply
from agent_orchestrator.adapter.outbound.llm.langchain.reviewer import LangChainReviewer
from agent_orchestrator.adapter.outbound.llm.langchain.summarizer import LangChainSummarizer
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.action import Action
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.plan import Plan, PlannedStep
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict, VerdictAction
from tests.agent_orchestrator.adapter.outbound.llm.langchain.fakes import (
    FakeChatModels,
    FakeLLM,
    text,
    tool_request,
)
from tests.agent_orchestrator.application.fakes import RecordingEvents
from tests.agent_orchestrator.domain.builders import PLAN, answer, calling, result, turn

ECHO = ToolSpecification("echo", "Echoes.", {"type": "object"})
WINDOW = ContextWindow()


def _models(llm: FakeLLM) -> tuple[FakeChatModels, ChatModels]:
    fake = FakeChatModels(llm)
    return fake, cast(ChatModels, fake)


def _actor(llm: FakeLLM, logger, final_reply: ReplyReader | None = None) -> LangChainActor:
    return LangChainActor(
        _models(llm)[1], WINDOW, logger, WholeReply(), final_reply or WholeReply()
    )


def _talked() -> Conversation:
    return Conversation(
        "c1",
        messages=[Message(Speaker.USER, "what is VaR?"), Message(Speaker.ASSISTANT, "A quantile.")],
    )


async def test_the_classifier_reads_the_history_the_pending_plan_and_the_message(logger) -> None:
    llm = FakeLLM(parsed=[UnderstandingDto(intent=Intent.PLAN_APPROVAL, standalone_query="yes")])
    fake, models = _models(llm)
    conversation = _talked()
    conversation.propose(PLAN)

    understood = await LangChainIntentClassifier(models, WINDOW, logger).understand(
        conversation, turn(None)
    )

    assert understood == Understanding(Intent.PLAN_APPROVAL, "yes")
    assert fake.requested == [(AgentRole.CONTEXT, "m")]
    prompt = llm.prompt()
    assert "User: what is VaR?\nAssistant: A quantile." in prompt
    assert PLAN.render() in prompt
    assert prompt.rstrip().endswith("msg")


async def test_an_unparsable_classification_is_none(logger) -> None:
    _, models = _models(FakeLLM(parsed=[None]))

    assert (
        await LangChainIntentClassifier(models, WINDOW, logger).understand(
            Conversation("c1"), turn(None)
        )
        is None
    )


async def test_the_planner_revises_the_pending_plan_and_ends_on_the_request(logger) -> None:
    llm = FakeLLM(
        parsed=[
            PlanDto(
                steps=[PlanStepDto(action="echo hi", tool="echo"), PlanStepDto(action="sum up")],
                expected_result="A greeting.",
            )
        ]
    )
    fake, models = _models(llm)
    conversation = _talked()
    conversation.propose(PLAN)

    plan = await LangChainPlanner(models, WINDOW, logger).plan(
        conversation, turn(Intent.PLAN_REVISION), [ECHO]
    )

    assert plan == Plan(
        "the query", (PlannedStep("echo hi", "echo"), PlannedStep("sum up")), "A greeting."
    )
    assert fake.requested == [(AgentRole.PLAN, "m")]
    system = llm.calls[0][0]
    assert isinstance(system, SystemMessage) and PLAN.render() in str(system.content)
    assert "- echo: Echoes." in str(system.content)
    assert llm.calls[0][-1] == HumanMessage(content="msg")


async def test_a_planned_tool_that_does_not_exist_becomes_a_step_without_a_tool(logger) -> None:
    llm = FakeLLM(
        parsed=[PlanDto(steps=[PlanStepDto(action="x", tool="nope")], expected_result="")]
    )

    plan = await LangChainPlanner(_models(llm)[1], WINDOW, logger).plan(
        Conversation("c1"), turn(Intent.TASK), [ECHO]
    )

    assert plan is not None and plan.steps == (PlannedStep("x"),)


async def test_an_unreadable_plan_is_none(logger) -> None:
    llm = FakeLLM(parsed=[None])

    plan = await LangChainPlanner(_models(llm)[1], WINDOW, logger).plan(
        Conversation("c1"), turn(Intent.TASK), [ECHO]
    )

    assert plan is None
    assert logger.messages("warning") == ["The plan could not be read"]


async def test_the_work_done_in_earlier_turns_is_shown_to_the_planner(logger) -> None:
    llm = FakeLLM(parsed=[None])
    conversation = Conversation(
        "c1", actions=[Action("file_writer", "file_path='report.md'", True, "written")]
    )

    await LangChainPlanner(_models(llm)[1], WINDOW, logger).plan(
        conversation, turn(Intent.TASK), [ECHO]
    )

    assert "- file_writer(file_path='report.md'): ok -> written" in str(llm.calls[0][0].content)


async def test_outside_a_plan_the_actor_lists_its_tools_but_binds_only_the_plan_request(
    logger,
) -> None:
    llm = FakeLLM(replies=[text("hello")])

    draft = await _actor(llm, logger).act(Conversation("c1"), turn(Intent.DIRECT), [ECHO])

    assert (draft.text, draft.tool_calls) == ("hello", ())
    assert llm.tool_bindings == [[REQUEST_PLAN_TOOL]]
    assert "- echo: Echoes." in llm.prompt()
    assert "cannot run tools" in llm.prompt()


async def test_under_a_plan_the_actor_binds_tools_and_replays_the_work(logger) -> None:
    llm = FakeLLM(replies=[tool_request("echo", text="hi")])
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo"))
    current.observed([result(output="echo:hi")])
    current.drafted(answer("draft"))
    current.give_feedback("shorter")

    draft = await _actor(llm, logger).act(Conversation("c1"), current, [ECHO])

    assert llm.tool_bindings == [
        [{"name": "echo", "description": "Echoes.", "parameters": {"type": "object"}}]
    ]
    assert [call.name for call in draft.tool_calls] == ["echo"]
    assert draft.tool_calls[0].arguments == {"text": "hi"}
    assert draft.tool_calls[0].id.startswith("call_")
    tail = llm.calls[0][-5:]
    assert tail[0] == HumanMessage(content="msg")
    assert isinstance(tail[1], AIMessage) and tail[1].tool_calls[0]["name"] == "echo"
    assert isinstance(tail[2], ToolMessage) and tail[2].tool_call_id == "c0"
    assert tail[3] == AIMessage(content="draft")
    assert isinstance(tail[4], HumanMessage) and "shorter" in str(tail[4].content)
    assert "✓ 1. count (tool: echo)" in llm.prompt()


async def test_while_a_step_is_open_the_actor_ends_on_the_next_step_to_do(logger) -> None:
    llm = FakeLLM(replies=[text("Step 1: I counted.")])
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(answer("I will count."))

    await _actor(llm, logger).act(Conversation("c1"), current, [ECHO])

    assert llm.calls[0][-1] == HumanMessage(
        content="Now do step 1: count\nCall echo to do it. "
        "Do not describe the step instead of running it."
    )


async def test_the_reviewer_judges_the_draft_against_the_evidence_and_the_exchange(logger) -> None:
    llm = FakeLLM(parsed=[VerdictDto(action=VerdictAction.RETRY, critique="wrong total")])
    fake, models = _models(llm)
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.understood(
        Understanding(Intent.PLAN_APPROVAL, "count rows", success_criteria=("the count",))
    )
    current.drafted(calling("echo"))
    current.observed([result(output="x" * 3_000)])
    current.drafted(answer("42 rows"))

    verdict = await LangChainReviewer(models, logger).review(_talked(), current)

    assert verdict == Verdict(VerdictAction.RETRY, "wrong total")
    assert fake.requested == [(AgentRole.REFLECTION, "m")]
    prompt = llm.prompt()
    assert "User: what is VaR?\nAssistant: A quantile.\nUser: msg" in prompt
    assert "- the count" in prompt and "✓ 1. count (tool: echo)" in prompt
    assert "- echo(): ok -> " + "x" * 300 + " [...]" in prompt
    assert "x" * 2_000 + " [...]" in prompt
    assert "42 rows" in prompt
    assert "not in the tool evidence" in prompt


async def test_without_evidence_the_reviewer_accepts_general_knowledge(logger) -> None:
    llm = FakeLLM(parsed=[VerdictDto(action=VerdictAction.ACCEPT, critique="")])
    current = turn(Intent.CONTINUATION)
    current.drafted(answer("A KV cache stores keys and values."))

    await LangChainReviewer(_models(llm)[1], logger).review(_talked(), current)

    assert "may use general knowledge" in llm.prompt()


async def test_the_summarizer_merges_the_summary_with_the_older_messages(logger) -> None:
    llm = FakeLLM(replies=[text("  merged  ")])
    fake, models = _models(llm)

    summary = await LangChainSummarizer(models, logger).summarize(
        turn(), "before", _talked().messages
    )

    assert summary == "merged"
    assert fake.requested == [(AgentRole.SUMMARY, "m")]
    assert "before" in llm.prompt() and "Assistant: A quantile." in llm.prompt()


async def test_outside_a_plan_a_plan_request_call_becomes_a_plan_request(logger) -> None:
    llm = FakeLLM(replies=[tool_request("request_plan", reason="read change.md")])

    draft = await _actor(llm, logger).act(Conversation("c1"), turn(Intent.DIRECT), [ECHO])

    assert draft == Draft.asking_for_plan("read change.md")


async def test_under_a_plan_the_plan_request_tool_is_not_offered(logger) -> None:
    llm = FakeLLM(replies=[text("done")])

    draft = await _actor(llm, logger).act(
        Conversation("c1"), turn(Intent.PLAN_APPROVAL, plan=PLAN), [ECHO]
    )

    assert draft == Draft("done")
    assert REQUEST_PLAN_TOOL not in llm.tool_bindings[0]


async def test_a_plain_direct_answer_is_read_by_the_final_reply(logger) -> None:
    llm, events = FakeLLM(replies=[text("hello there")]), RecordingEvents()
    actor = _actor(llm, logger, StreamedReply(events))

    draft = await actor.act(Conversation("c1"), turn(Intent.DIRECT), [ECHO])

    assert draft == Draft("hello there")
    assert [event.text for event in events.published] == ["hello ", "there"]


async def test_a_reply_that_will_be_reviewed_is_never_read_by_the_final_reply(logger) -> None:
    llm, events = FakeLLM(replies=[text("hello")]), RecordingEvents()
    actor = _actor(llm, logger, StreamedReply(events))

    draft = await actor.act(Conversation("c1"), turn(Intent.CONTINUATION), [ECHO])

    assert draft == Draft("hello")
    assert events.published == []
