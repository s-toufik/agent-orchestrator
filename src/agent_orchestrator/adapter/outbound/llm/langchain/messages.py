import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent_orchestrator.adapter.outbound.llm.langchain.prompts.act import act_feedback
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.feedback import Feedback
from agent_orchestrator.domain.turn.turn import WorkItem

_SPEAKERS: dict[str, str] = {"human": "User", "ai": "Assistant", "tool": "Tool", "system": "System"}


def from_history(messages: Sequence[Message]) -> list[BaseMessage]:
    return [
        HumanMessage(content=m.text) if m.speaker is Speaker.USER else AIMessage(content=m.text)
        for m in messages
    ]


def from_work(work: Sequence[WorkItem]) -> list[BaseMessage]:
    return [_from_item(item) for item in work]


def _from_item(item: WorkItem) -> BaseMessage:
    match item:
        case Draft():
            return AIMessage(
                content=item.text,
                tool_calls=[
                    {"id": call.id, "name": call.name, "args": call.arguments}
                    for call in item.tool_calls
                ],
            )
        case ToolResult():
            return ToolMessage(content=item.content, tool_call_id=item.call_id)
        case Feedback():
            return HumanMessage(content=act_feedback(item.critique))


def to_draft(reply: Any) -> Draft:
    return Draft(
        text=str(reply.content),
        tool_calls=tuple(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=call["name"],
                arguments=call.get("args", {}),
            )
            for call in getattr(reply, "tool_calls", None) or []
        ),
    )


def render(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        return "(none)"
    return "\n".join(f"{_SPEAKERS.get(m.type, m.type)}: {m.content}" for m in messages)
