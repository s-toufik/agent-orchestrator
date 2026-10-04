from agent_orchestrator.adapter.outbound.llm.langchain.plan_request import direct_draft
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.turn.draft import Draft


def test_a_direct_reply_without_tool_calls_is_the_answer() -> None:
    assert direct_draft(Draft("hello")) == Draft("hello")


def test_a_plan_request_call_becomes_a_plan_request_with_its_reason() -> None:
    draft = Draft("", (ToolCall("1", "request_plan", {"reason": "read it"}),))

    assert direct_draft(draft) == Draft.asking_for_plan("read it")


def test_any_other_tool_call_means_the_answer_needs_a_plan() -> None:
    draft = Draft("", (ToolCall("1", "file_reader"), ToolCall("2", "echo")))

    assert direct_draft(draft) == Draft.asking_for_plan("run file_reader, echo")
