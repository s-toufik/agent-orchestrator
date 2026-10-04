from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.understanding import Understanding


def test_an_approval_without_a_pending_plan_is_a_direct_message() -> None:
    approval = Understanding(Intent.PLAN_APPROVAL, "yes")

    assert approval.normalized(has_pending_plan=False).intent is Intent.DIRECT
    assert approval.normalized(has_pending_plan=True).intent is Intent.PLAN_APPROVAL


def test_an_ambiguous_message_without_a_question_becomes_a_task() -> None:
    vague = Understanding(Intent.AMBIGUOUS, "?")
    asked = Understanding(Intent.AMBIGUOUS, "?", clarification_question="Which desk?")

    assert vague.normalized(has_pending_plan=False).intent is Intent.TASK
    assert asked.normalized(has_pending_plan=False) == asked


def test_an_unreadable_message_falls_back_to_a_task() -> None:
    assert Understanding.fallback("count rows") == Understanding(Intent.TASK, "count rows")


def test_only_tasks_and_revisions_need_a_plan() -> None:
    assert {intent for intent in Intent if intent.needs_plan} == {Intent.TASK, Intent.PLAN_REVISION}


def test_only_a_revision_or_a_question_keeps_the_pending_plan() -> None:
    assert {intent for intent in Intent if intent.keeps_pending_plan} == {
        Intent.PLAN_REVISION,
        Intent.AMBIGUOUS,
    }
