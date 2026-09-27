from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages

from agent_orchestrator.adapter.outbound.langgraph.service.tokens_service import (
    count_message_tokens,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState

_SUMMARY_SECTION = "\n\nSummary of the earlier conversation:\n{summary}"
_SPEAKERS: dict[str, str] = {"human": "User", "ai": "Assistant", "tool": "Tool", "system": "System"}


class ContextWindow:
    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def build(
        self, system: str, state: AgentState, tail: Sequence[BaseMessage] = ()
    ) -> list[BaseMessage]:
        if state.summary:
            system += _SUMMARY_SECTION.format(summary=state.summary)
        head = SystemMessage(content=system)
        reserved: int = count_message_tokens([head, *tail])
        return [head, *self.recent(state, reserved), *tail]

    def recent(self, state: AgentState, reserved_tokens: int = 0) -> list[BaseMessage]:
        messages: list[BaseMessage] = state.transcript.to_langchain()
        trimmed: list[BaseMessage] = trim_messages(
            messages,
            token_counter=count_message_tokens,
            max_tokens=max(self._max_tokens - reserved_tokens, 0),
            strategy="last",
            start_on="human",
        )
        return trimmed or messages[-1:]


def render(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        return "(none)"
    return "\n".join(f"{_SPEAKERS.get(m.type, m.type)}: {m.content}" for m in messages)
