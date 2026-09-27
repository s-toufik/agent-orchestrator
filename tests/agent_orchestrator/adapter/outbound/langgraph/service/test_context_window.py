from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import (
    ContextWindow,
    render,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState


def _state(*contents: str, summary: str = "") -> AgentState:
    roles = [Role.USER, Role.ASSISTANT]
    return AgentState(
        summary=summary,
        transcript=Conversation(
            messages=[
                ConversationMessage(role=roles[i % 2], content=content)
                for i, content in enumerate(contents)
            ]
        ),
    )


def test_build_puts_the_system_prompt_first_and_the_tail_last() -> None:
    window = ContextWindow(max_tokens=1_000)
    tail = [AIMessage(content="draft")]

    messages = window.build("sys", _state("q1", "a1", "q2"), tail)

    assert isinstance(messages[0], SystemMessage)
    assert [m.content for m in messages[1:]] == ["q1", "a1", "q2", "draft"]


def test_the_summary_is_folded_into_the_single_system_message() -> None:
    messages = ContextWindow(max_tokens=1_000).build("sys", _state("q", summary="we met"))

    assert sum(isinstance(m, SystemMessage) for m in messages) == 1
    assert "we met" in str(messages[0].content)


def test_trimming_drops_the_oldest_turns_and_starts_on_a_user_message() -> None:
    state = _state("old question " * 20, "old answer " * 20, "new question")

    recent = ContextWindow(max_tokens=30).recent(state)

    assert [m.content for m in recent] == ["new question"]
    assert isinstance(recent[0], HumanMessage)


def test_the_latest_message_is_kept_even_when_it_alone_exceeds_the_budget() -> None:
    recent = ContextWindow(max_tokens=1).recent(_state("a very long question " * 50))

    assert len(recent) == 1


def test_trimming_never_mutates_the_saved_transcript() -> None:
    state = _state("q1 " * 50, "a1 " * 50, "q2")

    ContextWindow(max_tokens=10).recent(state)

    assert len(state.transcript.messages) == 3


def test_render_labels_each_speaker() -> None:
    assert render([HumanMessage(content="hi"), AIMessage(content="hello")]) == (
        "User: hi\nAssistant: hello"
    )
    assert render([]) == "(none)"
