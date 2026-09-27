from dataclasses import replace

import pytest
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.no_auth import NoAuth
from pycraftcore.authentication.model.token_auth import TokenAuth

from bootstrap.configuration.anthropic_sdk_settings import AnthropicSdkSettings
from bootstrap.di.anthropic_sdk_di import AnthropicSdkDI, _mcp_server
from tests.bootstrap.di.test_agent_di import make_settings


def _connector(auth, transport: str = "streamable_http") -> McpConnector:
    return McpConnector(
        name="toolbox",
        type=ConnectorType.mcp,
        auth=auth,
        base_url="http://toolbox/mcp",
        timeout=5,
        transport=transport,
    )


def test_mcp_connectors_become_sdk_servers_with_their_auth() -> None:
    token = _mcp_server("toolbox", _connector(TokenAuth(key_name="X-Key", key_value="secret")))
    basic = _mcp_server("toolbox", _connector(BasicAuth(username="u", password="p"), "sse"))
    open_ = _mcp_server("toolbox", _connector(NoAuth()))

    assert (token.url, token.transport, token.headers) == (
        "http://toolbox/mcp",
        "http",
        {"X-Key": "secret"},
    )
    assert basic.transport == "sse"
    assert basic.headers == {"Authorization": "Basic dTpw"}
    assert open_.headers == {}


def _di(**overrides) -> AnthropicSdkDI:
    di = AnthropicSdkDI(make_settings())
    di.__dict__["_anthropic_sdk_settings"] = replace(AnthropicSdkSettings.from_env(), **overrides)
    return di


async def test_without_lite_llm_the_native_endpoint_is_used() -> None:
    di = _di(lite_llm_enabled=False, anthropic_base_url="http://homelab:8090")

    assert await di._anthropic_base_url() == "http://homelab:8090"
    assert di._lite_llm_gateway is None


async def test_without_lite_llm_an_endpoint_is_required() -> None:
    di = _di(lite_llm_enabled=False, anthropic_base_url="")

    with pytest.raises(ValueError, match="ANTHROPIC_BASE_URL"):
        await di._anthropic_base_url()


def test_every_configured_model_gets_a_profile() -> None:
    profiles = AnthropicSdkDI(make_settings())._model_profiles()

    qwen = profiles["qwen3-8b"]
    assert qwen.name == "qwen3-8b"
    assert qwen.limits.max_iterations == 20
    assert qwen.limits.max_retries == 2


async def test_stopping_an_engine_that_never_started_is_safe() -> None:
    await AnthropicSdkDI(make_settings())._stop_anthropic_sdk()


def test_the_cli_telemetry_goes_to_the_application_collector(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_HOST", "sirius")
    monkeypatch.setenv("OTEL_PORT", "4317")

    assert AnthropicSdkDI(make_settings())._cli_otlp_endpoint() == "http://sirius:4317"


def test_no_collector_means_no_cli_telemetry(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_HOST", "")

    assert AnthropicSdkDI(make_settings())._cli_otlp_endpoint() is None


def test_roles_come_from_agent_yml_and_null_follows_the_selected_model() -> None:
    roles = AnthropicSdkDI(make_settings())._model_roles()
    selected = AnthropicSdkDI(make_settings())._model_profiles()["qwen3-14b"]

    turn = roles.for_turn(selected)

    assert roles.context is not None and roles.context.name == "qwen3-8b"
    assert roles.context.reasoning_effort is None
    assert (turn.act, turn.plan, turn.reflection) == (selected, selected, selected)
    assert turn.context.name == "qwen3-8b"


def test_the_gateway_serves_each_model_once() -> None:
    names = [model.model_name for model in AnthropicSdkDI(make_settings())._gateway_models()]

    assert len(names) == len(set(names))
    assert "qwen3-8b" in names
