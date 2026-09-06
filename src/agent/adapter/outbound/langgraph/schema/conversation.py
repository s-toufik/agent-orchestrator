from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall


class Conversation:
    def __init__(self, messages: list[ConversationMessage] | None = None) -> None:
        self.messages: list[ConversationMessage] = messages or []

    def append(self, message: ConversationMessage) -> None:
        self.messages.append(message)

    def last(self) -> ConversationMessage | None:
        return self.messages[-1] if self.messages else None

    def first_user(self) -> ConversationMessage | None:
        return next((m for m in self.messages if m.role is Role.USER), None)

    def last_assistant(self) -> ConversationMessage | None:
        return next((m for m in reversed(self.messages) if m.role is Role.ASSISTANT), None)

    def copy(self) -> Conversation:
        return Conversation(list(self.messages))

    def to_langchain(self) -> list[BaseMessage]:
        return [self._to_dispatcher[message.role](message) for message in self.messages]

    @classmethod
    def from_langchain(cls, messages: list[Any]) -> Conversation:
        instance = cls()
        return cls([instance._from_dispatcher[type(message)](message) for message in messages])

    @property
    def _to_dispatcher(self) -> Mapping[Role, Callable[[ConversationMessage], BaseMessage]]:
        return {
            Role.USER: self._to_human,
            Role.ASSISTANT: self._to_assistant,
            Role.TOOL: self._to_tool,
            Role.SYSTEM: self._to_system,
        }

    @property
    def _from_dispatcher(
        self,
    ) -> Mapping[type[BaseMessage], Callable[[Any], ConversationMessage]]:
        return {
            HumanMessage: self._from_human,
            AIMessage: self._from_assistant,
            ToolMessage: self._from_tool,
            SystemMessage: self._from_system,
        }

    @staticmethod
    def _to_system(message: ConversationMessage) -> SystemMessage:
        return SystemMessage(content=message.content)

    @staticmethod
    def _to_human(message: ConversationMessage) -> HumanMessage:
        return HumanMessage(content=message.content)

    @staticmethod
    def _to_tool(message: ConversationMessage) -> ToolMessage:
        return ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "")

    @staticmethod
    def _to_assistant(message: ConversationMessage) -> AIMessage:
        return AIMessage(
            content=message.content,
            tool_calls=[
                {"id": call.id, "name": call.name, "args": call.args} for call in message.tool_calls
            ],
        )

    @staticmethod
    def _from_system(message: SystemMessage) -> ConversationMessage:
        return ConversationMessage(role=Role.SYSTEM, content=str(message.content))

    @staticmethod
    def _from_human(message: HumanMessage) -> ConversationMessage:
        return ConversationMessage(role=Role.USER, content=str(message.content))

    @staticmethod
    def _from_tool(message: ToolMessage) -> ConversationMessage:
        return ConversationMessage(
            role=Role.TOOL,
            content=str(message.content),
            tool_call_id=message.tool_call_id,
        )

    @staticmethod
    def _from_assistant(message: AIMessage) -> ConversationMessage:
        return ConversationMessage(
            role=Role.ASSISTANT,
            content=str(message.content),
            tool_calls=[
                ToolCall(
                    id=call["id"] or "",
                    name=call["name"],
                    args=call.get("args", {}),
                )
                for call in (message.tool_calls or [])
            ],
        )
