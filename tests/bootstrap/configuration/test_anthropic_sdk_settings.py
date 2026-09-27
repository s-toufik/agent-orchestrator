from pathlib import Path

import pytest

from bootstrap.configuration.anthropic_sdk_settings import (
    DEFAULT_LITE_LLM_COMMAND,
    AnthropicSdkSettings,
)

KEYS = (
    "WORKING_DIRECTORY",
    "AGENT_SDK_TOOLS",
    "LITELLM_ENABLED",
    "LITELLM_COMMAND",
    "LITELLM_HOST",
    "LITELLM_PORT",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
)


def test_defaults_start_lite_llm_transparently(monkeypatch) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = AnthropicSdkSettings.from_env()

    assert settings.lite_llm_enabled is True
    assert settings.lite_llm_command == tuple(DEFAULT_LITE_LLM_COMMAND.split())
    assert (settings.lite_llm_host, settings.lite_llm_port) == ("127.0.0.1", 4000)
    assert settings.working_directory == (Path.cwd() / "working_directory").resolve()
    assert settings.builtin_tools == ()
    assert settings.anthropic_auth_token == "none"


def test_lite_llm_can_be_removed_for_a_native_anthropic_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://homelab:8090")
    monkeypatch.setenv("LITELLM_COMMAND", "/opt/litellm/bin/litellm --num_workers 2")

    settings = AnthropicSdkSettings.from_env()

    assert settings.lite_llm_enabled is False
    assert settings.anthropic_base_url == "http://homelab:8090"
    assert settings.lite_llm_command == ("/opt/litellm/bin/litellm", "--num_workers", "2")


def test_the_working_directory_and_the_built_in_tools_come_from_the_deployment(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("AGENT_SDK_TOOLS", " Glob, Grep ,Glob")

    settings = AnthropicSdkSettings.from_env()

    assert settings.working_directory == tmp_path.resolve()
    assert settings.builtin_tools == ("Glob", "Grep")


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ("Glob,Bash", "unsupported built-in tool\\(s\\) Bash"),
        ("Edit,Glob", "Edit needs Read"),
    ],
)
def test_a_wrong_tool_list_stops_the_boot(monkeypatch, tools: str, message: str) -> None:
    monkeypatch.setenv("AGENT_SDK_TOOLS", tools)

    with pytest.raises(ValueError, match=message):
        AnthropicSdkSettings.from_env()
