from typing import cast

from langchain_core.language_models import BaseChatModel

from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.reflection_node import ReflectionNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state


class FakeStructuredRunnable:
    def __init__(self, result: dict, sent: list) -> None:
        self._result = result
        self._sent = sent

    async def ainvoke(self, messages):
        self._sent.append(messages)
        return self._result


class FakeLLM:
    def __init__(self, result: dict) -> None:
        self._result = result
        self.sent_messages: list = []

    def with_structured_output(self, schema, include_raw=True):
        assert schema is ReflectionDecision
        assert include_raw is True
        return FakeStructuredRunnable(self._result, self.sent_messages)


def _node(llm: FakeLLM) -> ReflectionNode:
    return ReflectionNode(cast(BaseChatModel, llm), PromptService())


async def test_a_parsed_decision_is_stored_on_the_state() -> None:
    decision = ReflectionDecision(action=ReflectionAction.RETRY, critique="missing detail")
    llm = FakeLLM({"parsed": decision, "raw": None, "parsing_error": None})
    node = _node(llm)
    state = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.ASSISTANT, content="the answer")])
    )

    result = unpack_state(await node(pack_state(state)))

    assert result.reflection is decision or result.reflection == decision
    assert result.last_node == "reflection"


async def test_a_failed_parse_leaves_reflection_none() -> None:
    llm = FakeLLM({"parsed": None, "raw": None, "parsing_error": ValueError("bad json")})
    node = _node(llm)
    state = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.ASSISTANT, content="x")])
    )

    result = unpack_state(await node(pack_state(state)))

    assert result.reflection is None


async def test_uses_a_placeholder_when_there_is_no_assistant_answer_yet() -> None:
    llm = FakeLLM({"parsed": None})
    node = _node(llm)

    await node(pack_state(AgentState()))

    sent = llm.sent_messages[0]
    assert "(no assistant answer found)" in str(sent[-1].content)


async def test_the_assistant_answer_is_embedded_in_the_evaluation_prompt() -> None:
    llm = FakeLLM({"parsed": None})
    node = _node(llm)
    state = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.ASSISTANT, content="42 users")])
    )

    await node(pack_state(state))

    sent = llm.sent_messages[0]
    assert "42 users" in str(sent[-1].content)
