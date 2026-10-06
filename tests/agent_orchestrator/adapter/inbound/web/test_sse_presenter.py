import json

import pytest

from agent_orchestrator.adapter.inbound.web.sse_presenter import APPROVAL_PROMPT, SsePresenter
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.workflow.step import Step

PRESENTER = SsePresenter()


def _frames(chunks: list[bytes]) -> list[tuple[str, dict]]:
    frames = []
    for chunk in chunks:
        event, data = chunk.decode("utf-8").strip().split("\n")
        frames.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return frames


@pytest.mark.parametrize(
    ("step", "status"),
    [
        (Step.UNDERSTAND, "Understanding your request"),
        (Step.PLAN, "Preparing a plan"),
        (Step.REVIEW, "Checking the answer"),
    ],
)
def test_visible_steps_become_a_status(step: Step, status: str) -> None:
    frames = _frames(PRESENTER.present(TurnEvent.entering(step, Turn("hi", "m")), "c1"))

    assert frames == [("status", {"type": "status", "content": status})]


def test_running_tools_names_them() -> None:
    turn = Turn("hi", "m")
    turn.drafted(Draft("", (ToolCall("1", "echo"), ToolCall("2", "file_reader"))))

    frames = _frames(PRESENTER.present(TurnEvent.entering(Step.RUN_TOOLS, turn), "c1"))

    assert frames[0][1]["content"] == "Running echo, file_reader"


@pytest.mark.parametrize("step", [Step.ACT, Step.FEEDBACK, Step.FINISH, Step.SUMMARIZE])
def test_other_steps_run_silently(step: Step) -> None:
    assert PRESENTER.present(TurnEvent.entering(step, Turn("hi", "m")), "c1") == []


def _finished() -> TurnEvent:
    turn = Turn("hi", "m", settings=TurnSettings(max_steps=6))
    turn.drafted(Draft("hello"))
    turn.finish(Answer.answered("hello"))
    return TurnEvent.finished(turn)


def test_the_answer_is_sent_as_the_final_event_with_its_metadata() -> None:
    frames = _frames(PRESENTER.present(_finished(), "c1"))

    assert [name for name, _ in frames] == ["final"]
    final = frames[0][1]
    assert (final["session_id"], final["content"]) == ("c1", "hello")
    assert final["metadata"] == {"iteration": "1", "max_iteration": "6", "outcome": "answered"}


def test_a_plan_awaiting_approval_tells_the_user_how_to_approve_it() -> None:
    turn = Turn("hi", "m")
    turn.finish(Answer.approval_request(Plan("task", "## Plan\n1. echo")))

    final = _frames(PRESENTER.present(TurnEvent.finished(turn), "c1"))[0][1]

    assert final["content"] == f"## Plan\n1. echo\n\n---\n{APPROVAL_PROMPT}"


def test_each_piece_of_a_streamed_answer_is_a_token_event() -> None:
    frames = _frames(PRESENTER.present(TurnEvent.answer_delta("hel"), "c1"))

    assert frames == [("token", {"type": "token", "content": "hel"})]


def test_a_failure_is_an_error_event() -> None:
    frames = _frames(PRESENTER.present(TurnEvent.failed("down"), "c1"))

    assert frames == [("error", {"type": "error", "content": "down"})]


def test_the_stream_ends_with_complete() -> None:
    assert _frames([PRESENTER.complete()]) == [("complete", {"type": "complete", "content": ""})]


def test_a_streaming_failure_is_an_error_event_the_ui_shows() -> None:
    assert _frames([PRESENTER.error("trace")]) == [("error", {"type": "error", "content": "trace"})]
