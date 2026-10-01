from agent_orchestrator.application.step.act_step import ActStep
from agent_orchestrator.application.step.clarify_step import ClarifyStep
from agent_orchestrator.application.step.feedback_step import NO_CRITIQUE, FeedbackStep
from agent_orchestrator.application.step.finish_step import FinishStep
from agent_orchestrator.application.step.plan_step import PlanStep
from agent_orchestrator.application.step.review_step import ReviewStep
from agent_orchestrator.application.step.run_tools_step import RunToolsStep
from agent_orchestrator.application.step.summarize_step import SummarizeStep
from agent_orchestrator.application.step.understand_step import UnderstandStep
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict
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
)
from tests.agent_orchestrator.domain.builders import PLAN, answer, calling, retry, turn


async def test_understanding_is_recorded_through_the_conversation(logger) -> None:
    conversation, current = Conversation("c1"), turn(None)
    conversation.propose(PLAN)
    step = UnderstandStep(FakeClassifier(Understanding(Intent.PLAN_APPROVAL, "yes")), logger)

    await step.run(conversation, current)

    assert current.plan == PLAN
    assert conversation.pending_plan is None


async def test_an_unreadable_message_is_understood_as_a_task(logger) -> None:
    current = turn(None)

    await UnderstandStep(FakeClassifier(None), logger).run(Conversation("c1"), current)

    assert current.intent is Intent.TASK
    assert current.query == "msg"
    assert logger.messages("warning")


async def test_a_plan_is_proposed_and_shown_for_approval() -> None:
    conversation, current = Conversation("c1"), turn(Intent.TASK)

    await PlanStep(FakePlanner("1. echo"), FakeCatalog(ECHO)).run(conversation, current)

    assert conversation.pending_plan is not None
    assert conversation.pending_plan.steps == "1. echo"
    assert current.answer is not None
    assert current.answer.outcome is Outcome.AWAITING_APPROVAL


async def test_a_clarifying_question_is_the_answer() -> None:
    current = turn(None)
    current.understood(Understanding(Intent.AMBIGUOUS, "?", clarification_question="Which desk?"))

    await ClarifyStep().run(Conversation("c1"), current)

    assert current.answer == Answer.clarification("Which desk?")


async def test_acting_records_the_draft_and_sees_the_catalog() -> None:
    actor, current = FakeActor(answer("hi")), turn()

    await ActStep(actor, FakeCatalog(ECHO)).run(Conversation("c1"), current)

    assert current.last_draft == answer("hi")


async def test_every_requested_tool_runs_and_its_result_is_kept() -> None:
    executor, current = EchoExecutor(), turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo", "echo"))

    await RunToolsStep(executor).run(Conversation("c1"), current)

    assert len(executor.calls) == 2
    assert [result.tool_name for result in current.evidence] == ["echo", "echo"]


async def test_an_unreadable_review_accepts_the_draft(logger) -> None:
    current = turn(Intent.CONTINUATION)

    await ReviewStep(FakeReviewer(None), logger).run(Conversation("c1"), current)

    assert current.last_verdict == Verdict.accept()
    assert logger.messages("warning")


async def test_feedback_hands_the_critique_back_to_the_actor(logger) -> None:
    current = turn(Intent.CONTINUATION)
    current.reviewed(retry("too short"))
    empty = turn(Intent.CONTINUATION)
    empty.reviewed(retry(""))

    await FeedbackStep(logger).run(Conversation("c1"), current)
    await FeedbackStep(logger).run(Conversation("c1"), empty)

    assert current.retries == 1
    assert current.work[-1].critique == "too short"
    assert empty.work[-1].critique == NO_CRITIQUE


async def test_finishing_settles_the_answer_and_keeps_the_exchange() -> None:
    conversation, current = Conversation("c1"), turn(Intent.DIRECT)
    current.drafted(answer("hello"))

    await FinishStep().run(conversation, current)

    assert current.answer == Answer.answered("hello")
    assert [m.text for m in conversation.messages] == ["msg", "hello"]


async def test_finishing_keeps_an_answer_already_settled() -> None:
    conversation, current = Conversation("c1"), turn(Intent.AMBIGUOUS)
    current.finish(Answer.clarification("Which desk?"))

    await FinishStep().run(conversation, current)

    assert current.answer == Answer.clarification("Which desk?")


def _long(count: int = 6) -> Conversation:
    return Conversation("c1", messages=[Message(Speaker.USER, str(i)) for i in range(count)])


async def test_a_long_history_is_summarized_keeping_the_last_messages(logger) -> None:
    conversation, summarizer = _long(), FakeSummarizer("the summary")
    step = SummarizeStep(summarizer, FakeTokens(10_000), logger)

    await step.run(conversation, turn())

    assert summarizer.calls[0][1] == [Message(Speaker.USER, "0"), Message(Speaker.USER, "1")]
    assert conversation.summary == "the summary"
    assert len(conversation.messages) == 4


async def test_a_short_or_small_history_is_left_alone(logger) -> None:
    summarizer = FakeSummarizer()

    await SummarizeStep(summarizer, FakeTokens(10_000), logger).run(_long(4), turn())
    await SummarizeStep(summarizer, FakeTokens(1), logger).run(_long(6), turn())

    assert summarizer.calls == []


async def test_a_failed_summary_keeps_the_history(logger) -> None:
    conversation = _long()
    step = SummarizeStep(FakeSummarizer(RuntimeError("down")), FakeTokens(10_000), logger)

    await step.run(conversation, turn())

    assert len(conversation.messages) == 6
    assert logger.messages("warning")
