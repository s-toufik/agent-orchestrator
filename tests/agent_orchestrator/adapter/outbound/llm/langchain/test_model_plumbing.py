import pytest
from langchain_core.exceptions import ModelError
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from agent_orchestrator.adapter.outbound.llm.langchain.chat_models import ChatModels
from agent_orchestrator.adapter.outbound.llm.langchain.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke
from agent_orchestrator.adapter.outbound.llm.langchain.token_counter import (
    EstimatedTokenCounter,
    count_tokens,
)
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole, ModelCatalog
from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.exception.unknown_model_exception import UnknownModelException
from agent_orchestrator.domain.model.model_listing import ModelListing, SelectableModel
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.workflow.step import Step
from tests.agent_orchestrator.adapter.outbound.llm.langchain.fakes import FakeLLM

CONNECTOR = ModelConnector(base_url="http://llm:8080/v1", api_key=SecretStr("k"))


def _parameters(name: str, **fields) -> ModelParameters:
    defaults = {
        "temperature": 0.0,
        "max_output_tokens": 100,
        "max_context_tokens": 1_000,
        "max_iterations": 4,
    }
    return ModelParameters(model_name=name, **{**defaults, **fields})


CATALOG = ModelCatalog(
    models={
        "small": (CONNECTOR, _parameters("small", max_context_tokens=500)),
        "big": (CONNECTOR, _parameters("big", max_iterations=9)),
    },
    roles={AgentRole.CONTEXT: (CONNECTOR, _parameters("small"))},
)


def test_a_role_without_a_model_follows_the_selection_and_a_frozen_one_does_not() -> None:
    assert CATALOG.settings_for(AgentRole.ACT, "big")[1].model_name == "big"
    assert CATALOG.settings_for(AgentRole.CONTEXT, "big")[1].model_name == "small"


def test_the_turn_budget_comes_from_the_model_that_acts() -> None:
    assert CATALOG.turn_settings("big") == TurnSettings(
        max_steps=9, max_retries=2, context_tokens=1_000
    )


def test_a_frozen_act_model_sets_the_budget_whatever_the_selection() -> None:
    frozen = ModelCatalog(
        models=CATALOG.models, roles={AgentRole.ACT: (CONNECTOR, _parameters("small"))}
    )

    assert frozen.turn_settings("big").max_steps == 4


def test_an_unknown_model_is_refused() -> None:
    with pytest.raises(UnknownModelException):
        CATALOG.turn_settings("gpt-oss-20b")
    with pytest.raises(UnknownModelException):
        CATALOG.settings_for(AgentRole.ACT, "gpt-oss-20b")


def test_chat_clients_are_built_once_per_model() -> None:
    models = ChatModels(CATALOG)

    act = models.for_role(AgentRole.ACT, "big")

    assert models.for_role(AgentRole.ACT, "big") is act
    assert models.for_role(AgentRole.CONTEXT, "big").model_name == "small"  # ty: ignore[unresolved-attribute]


class _Unavailable(ModelError):
    is_retryable = True


async def test_a_retryable_model_error_means_the_agent_is_unavailable() -> None:
    with pytest.raises(AgentUnavailableException):
        await invoke(FakeLLM(replies=[_Unavailable("503")]), [])  # ty: ignore[invalid-argument-type]


async def test_a_non_retryable_model_error_propagates() -> None:
    with pytest.raises(ModelError):
        await invoke(FakeLLM(replies=[ModelError("bad request")]), [])  # ty: ignore[invalid-argument-type]


def _history(count: int) -> Conversation:
    return Conversation(
        "c1",
        messages=[
            Message(Speaker.USER if i % 2 == 0 else Speaker.ASSISTANT, f"message number {i}")
            for i in range(count)
        ],
        summary="earlier talk",
    )


def test_the_window_adds_the_summary_and_keeps_the_tail() -> None:
    tail = [HumanMessage(content="now")]

    built = ContextWindow().build("system", _history(2), 10_000, tail)

    assert built[0] == SystemMessage(
        content="system\n\nSummary of the earlier conversation:\nearlier talk"
    )
    assert [m.content for m in built[1:]] == ["message number 0", "message number 1", "now"]


def test_the_window_keeps_only_the_most_recent_history_that_fits() -> None:
    history = ContextWindow.history(_history(10), budget=20)

    assert history
    assert history[-1].content == "message number 9"
    assert history[0].type == "human"
    assert len(history) < 10


def test_tokens_are_estimated_from_words_and_punctuation() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
    assert count_tokens("file.md") == 2
    assert EstimatedTokenCounter().count([Message(Speaker.USER, "hello world")]) == 6


def test_the_listing_names_every_selectable_model_and_the_steps_pinned_to_one() -> None:
    thinking = ModelCatalog(
        models={**CATALOG.models, "deep": (CONNECTOR, _parameters("deep", reasoning_effort="low"))},
        roles=CATALOG.roles,
    )

    assert thinking.listing() == ModelListing(
        models=(
            SelectableModel("small", context_tokens=500, max_output_tokens=100, thinking=False),
            SelectableModel("big", context_tokens=1_000, max_output_tokens=100, thinking=False),
            SelectableModel("deep", context_tokens=1_000, max_output_tokens=100, thinking=True),
        ),
        pinned_steps={Step.UNDERSTAND: "small"},
    )
