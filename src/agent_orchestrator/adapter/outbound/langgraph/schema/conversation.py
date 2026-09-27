from __future__ import annotations

from collections.abc import Callable, Mapping

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)


class Conversation(BaseModel):
    messages: list[ConversationMessage] = Field(default_factory=list)

    def append(self, message: ConversationMessage) -> None:
        self.messages.append(message)

    def last(self) -> ConversationMessage | None:
        return self.messages[-1] if self.messages else None

    def last_assistant(self) -> ConversationMessage | None:
        return next((m for m in reversed(self.messages) if m.role is Role.ASSISTANT), None)

    def of_role(self, role: Role) -> list[ConversationMessage]:
        return [m for m in self.messages if m.role is role]

    def to_langchain(self) -> list[BaseMessage]:
        return [_TO_LANGCHAIN[message.role](message) for message in self.messages]


def _to_assistant(message: ConversationMessage) -> AIMessage:
    return AIMessage(
        content=message.content,
        tool_calls=[
            {"id": call.id, "name": call.name, "args": call.args} for call in message.tool_calls
        ],
    )


_TO_LANGCHAIN: Mapping[Role, Callable[[ConversationMessage], BaseMessage]] = {
    Role.USER: lambda message: HumanMessage(content=message.content),
    Role.ASSISTANT: _to_assistant,
    Role.TOOL: lambda message: ToolMessage(
        content=message.content, tool_call_id=message.tool_call_id or ""
    ),
    Role.SYSTEM: lambda message: SystemMessage(content=message.content),
}
