import pytest

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.understanding import Understanding
from tests.agent_orchestrator.domain.builders import PLAN, turn


def _waiting() -> Conversation:
    conversation = Conversation(id="c1")
    conversation.propose(PLAN)
    return conversation


def test_approving_takes_the_pending_plan_into_the_turn() -> None:
    conversation, current = _waiting(), turn(None)

    conversation.interpret(current, Understanding(Intent.PLAN_APPROVAL, "yes"))

    assert current.plan == PLAN
    assert current.query == PLAN.task
    assert conversation.pending_plan is None


@pytest.mark.parametrize("intent", [Intent.PLAN_REVISION, Intent.AMBIGUOUS])
def test_discussing_the_plan_keeps_it_pending(intent: Intent) -> None:
    conversation, current = _waiting(), turn(None)

    conversation.interpret(current, Understanding(intent, "q", clarification_question="?"))

    assert conversation.pending_plan == PLAN
    assert current.plan is None


@pytest.mark.parametrize("intent", [Intent.TASK, Intent.DIRECT, Intent.CONTINUATION])
def test_moving_on_drops_the_pending_plan(intent: Intent) -> None:
    conversation, current = _waiting(), turn(None)

    conversation.interpret(current, Understanding(intent, "q"))

    assert conversation.pending_plan is None


def test_an_approval_with_nothing_pending_is_read_as_direct() -> None:
    conversation, current = Conversation(id="c1"), turn(None)

    conversation.interpret(current, Understanding(Intent.PLAN_APPROVAL, "yes"))

    assert current.intent is Intent.DIRECT
    assert current.plan is None


def test_closing_a_turn_keeps_the_message_and_its_answer() -> None:
    conversation, current = Conversation(id="c1"), turn()
    current.finish(Answer.answered("hello back"))

    conversation.close(current)

    assert conversation.messages == [
        Message(Speaker.USER, "msg"),
        Message(Speaker.ASSISTANT, "hello back"),
    ]


def test_compacting_keeps_only_the_last_messages_and_the_summary() -> None:
    messages = [Message(Speaker.USER, str(i)) for i in range(6)]
    conversation = Conversation(id="c1", messages=list(messages))

    assert conversation.older_than(4) == messages[:2]
    conversation.compact("earlier: 0, 1", keep_last=4)

    assert conversation.summary == "earlier: 0, 1"
    assert conversation.messages == messages[2:]
