from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages

from agent_orchestrator.adapter.outbound.llm.langchain.messages import from_history
from agent_orchestrator.adapter.outbound.llm.langchain.token_counter import count_texts
from agent_orchestrator.domain.conversation.conversation import Conversation

_SUMMARY_SECTION = "\n\nSummary of the earlier conversation:\n{summary}"
_ACTIONS_SECTION = (
    "\n\nTools you ran earlier in this conversation, oldest first "
    "(reuse their results instead of running them again):\n{actions}"
)


def count_message_tokens(messages: Sequence[BaseMessage]) -> int:
    return count_texts(
        m.content if isinstance(m.content, str) else str(m.content) for m in messages
    )


class ContextWindow:
    def build(
        self,
        system: str,
        conversation: Conversation,
        budget: int,
        tail: Sequence[BaseMessage] = (),
    ) -> list[BaseMessage]:
        if conversation.summary:
            system += _SUMMARY_SECTION.format(summary=conversation.summary)
        if conversation.actions:
            actions = "\n".join(f"- {action.render()}" for action in conversation.actions)
            system += _ACTIONS_SECTION.format(actions=actions)
        head = SystemMessage(content=system)
        reserved = count_message_tokens([head, *tail])
        return [head, *self.history(conversation, budget - reserved), *tail]

    @staticmethod
    def history(conversation: Conversation, budget: int) -> list[BaseMessage]:
        return trim_messages(
            from_history(conversation.messages),
            token_counter=count_message_tokens,
            max_tokens=max(budget, 0),
            strategy="last",
            start_on="human",
        )
