from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_orchestrator.adapter.outbound.llm.langchain.actor import LangChainActor
from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.dto import UnderstandingDto, VerdictDto
from agent_orchestrator.adapter.outbound.llm.langchain.intent_classifier import (
    LangChainIntentClassifier,
)
from agent_orchestrator.adapter.outbound.llm.langchain.planner import LangChainPlanner
from agent_orchestrator.adapter.outbound.llm.langchain.reviewer import LangChainReviewer
from agent_orchestrator.adapter.outbound.llm.langchain.summarizer import LangChainSummarizer
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict, VerdictAction
from tests.agent_orchestrator.adapter.outbound.llm.langchain.fakes import (
    FakeChatModels,
    FakeLLM,
    text,
    tool_request,
)
from tests.agent_orchestrator.domain.builders import PLAN, answer, calling, result, turn

ECHO = ToolSpecification("echo", "Echoes.", {"type": "object"})
WINDOW = ContextWindow()


def _models(llm: FakeLLM) -> tuple[FakeChatModels, ChatModels]:
    fake = FakeChatModels(llm)
    return fake, cast(ChatModels, fake)


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
    assert PLAN.steps in prompt
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
    llm = FakeLLM(replies=[text("  ## Plan\n1. echo  ")])
    fake, models = _models(llm)
    conversation = _talked()
    conversation.propose(PLAN)

    plan = await LangChainPlanner(models, WINDOW, logger).plan(
        conversation, turn(Intent.PLAN_REVISION), [ECHO]
    )

    assert (plan.task, plan.steps) == ("the query", "## Plan\n1. echo")
    assert fake.requested == [(AgentRole.PLAN, "m")]
    system = llm.calls[0][0]
    assert isinstance(system, SystemMessage) and PLAN.steps in str(system.content)
    assert "- echo: Echoes." in str(system.content)
    assert llm.calls[0][-1] == HumanMessage(content="msg")


async def test_outside_a_plan_the_actor_lists_its_tools_but_binds_none(logger) -> None:
    llm = FakeLLM(replies=[text("hello")])

    draft = await LangChainActor(_models(llm)[1], WINDOW, logger).act(
        Conversation("c1"), turn(Intent.DIRECT), [ECHO]
    )

    assert (draft.text, draft.tool_calls) == ("hello", ())
    assert llm.tool_bindings == []
    assert "- echo: Echoes." in llm.prompt()
    assert "cannot run tools" in llm.prompt()


async def test_under_a_plan_the_actor_binds_tools_and_replays_the_work(logger) -> None:
    llm = FakeLLM(replies=[tool_request("echo", text="hi")])
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo"))
    current.observed([result(output="echo:hi")])
    current.drafted(answer("draft"))
    current.give_feedback("shorter")

    draft = await LangChainActor(_models(llm)[1], WINDOW, logger).act(
        Conversation("c1"), current, [ECHO]
    )

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
    assert PLAN.steps in llm.prompt()


async def test_the_reviewer_judges_the_draft_against_the_evidence_and_the_exchange(logger) -> None:
    llm = FakeLLM(parsed=[VerdictDto(action=VerdictAction.RETRY, critique="wrong total")])
    fake, models = _models(llm)
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.understood(
        Understanding(Intent.PLAN_APPROVAL, "count rows", success_criteria=("the count",))
    )
    current.observed([result(output="x" * 3_000)])
    current.drafted(answer("42 rows"))

    verdict = await LangChainReviewer(models, logger).review(_talked(), current)

    assert verdict == Verdict(VerdictAction.RETRY, "wrong total")
    assert fake.requested == [(AgentRole.REFLECTION, "m")]
    prompt = llm.prompt()
    assert "User: what is VaR?\nAssistant: A quantile.\nUser: msg" in prompt
    assert "- the count" in prompt and PLAN.steps in prompt
    assert "x" * 2_000 + " [...]" in prompt
    assert "42 rows" in prompt


async def test_the_summarizer_merges_the_summary_with_the_older_messages(logger) -> None:
    llm = FakeLLM(replies=[text("  merged  ")])
    fake, models = _models(llm)

    summary = await LangChainSummarizer(models, logger).summarize(
        turn(), "before", _talked().messages
    )

    assert summary == "merged"
    assert fake.requested == [(AgentRole.SUMMARY, "m")]
    assert "before" in llm.prompt() and "Assistant: A quantile." in llm.prompt()


async def test_outside_a_plan_a_reply_asking_for_one_becomes_a_plan_request(logger) -> None:
    llm = FakeLLM(replies=[text("I need to read it.\nNEEDS_PLAN: read change.md")])

    draft = await LangChainActor(_models(llm)[1], WINDOW, logger).act(
        Conversation("c1"), turn(Intent.DIRECT), [ECHO]
    )

    assert draft == Draft.asking_for_plan("read change.md")
    assert "NEEDS_PLAN:" in llm.prompt()


async def test_under_a_plan_the_reply_is_never_read_as_a_plan_request(logger) -> None:
    llm = FakeLLM(replies=[text("NEEDS_PLAN: more")])

    draft = await LangChainActor(_models(llm)[1], WINDOW, logger).act(
        Conversation("c1"), turn(Intent.PLAN_APPROVAL, plan=PLAN), [ECHO]
    )

    assert not draft.asks_for_plan
    assert "NEEDS_PLAN:" not in str(llm.calls[0][0].content)
