from pydantic import SecretStr

from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort
from agent_orchestrator.adapter.outbound.llm.langchain.chat_factory import LLMChat
from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters


def test_create_chat_client_applies_every_configured_field() -> None:
    connector = ModelConnector(base_url="http://example.com", api_key=SecretStr("key"))
    parameters = ModelParameters(
        model_name="qwen3-14b",
        temperature=0.3,
        max_output_tokens=1234,
        max_context_tokens=8000,
        max_iterations=6,
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    client = LLMChat(connector, parameters).create_chat_client()

    assert client.model_name == "qwen3-14b"
    assert client.temperature == 0.3
    assert client.max_tokens == 1234
    assert client.openai_api_base == "http://example.com"
    assert client.reasoning_effort == "medium"
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_the_langchain_openai_sdk_retries_are_disabled() -> None:
    connector = ModelConnector(base_url="http://example.com", api_key=SecretStr("key"))
    parameters = ModelParameters(
        model_name="m",
        temperature=0.0,
        max_output_tokens=100,
        max_context_tokens=100,
        max_iterations=1,
    )

    client = LLMChat(connector, parameters).create_chat_client()

    assert client.client._client.max_retries == 0
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
