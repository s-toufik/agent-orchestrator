from pydantic import SecretStr

from agent.adapter.outbound.llm.factory import LLMChat
from agent.adapter.outbound.llm.schema import ModelConnector, ModelParameters


def test_create_chat_client_applies_every_configured_field() -> None:
    connector = ModelConnector(base_url="http://example.com", api_key=SecretStr("key"))
    parameters = ModelParameters(
        model_name="gpt-oss-20b",
        temperature=0.3,
        max_output_tokens=1234,
        max_context_tokens=8000,
        max_iterations=6,
        use_streaming=True,
    )

    client = LLMChat(connector, parameters).create_chat_client()

    assert client.model_name == "gpt-oss-20b"
    assert client.temperature == 0.3
    assert client.max_tokens == 1234
    assert client.streaming is True
    assert client.openai_api_base == "http://example.com"


def test_the_langchain_openai_sdk_retries_are_disabled() -> None:
    # Retry/circuit-breaking live in the resilient transport; the SDK's own
    # internal retry loop must be off or the two stack on top of each other.
    connector = ModelConnector(base_url="http://example.com", api_key=SecretStr("key"))
    parameters = ModelParameters(
        model_name="m",
        temperature=0.0,
        max_output_tokens=100,
        max_context_tokens=100,
        max_iterations=1,
        use_streaming=False,
    )

    client = LLMChat(connector, parameters).create_chat_client()

    assert client.client._client.max_retries == 0
