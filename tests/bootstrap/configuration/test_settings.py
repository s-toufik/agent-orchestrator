import pytest

import bootstrap.configuration.settings as settings_module
from agent_orchestrator.adapter.outbound.llm.enum.answer_delivery import AnswerDelivery
from bootstrap.configuration.settings import ProcessSettings


def test_for_role_uses_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CONFIGURATION_DIR", "/custom/config")
    monkeypatch.setenv("MAX_CONCURRENT_STREAMS", "12")
    monkeypatch.setenv("AGENT_ANSWER_DELIVERY", "whole")

    settings = ProcessSettings.for_role("agent-orchestrator")

    assert settings.role == "agent-orchestrator"
    assert settings.environment == "prod"
    assert str(settings.configuration_directory) == "/custom/config"
    assert settings.max_concurrent_streams == 12
    assert settings.answer_delivery is AnswerDelivery.WHOLE


def test_for_role_falls_back_to_defaults_when_nothing_is_set(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    for key in ("APP_ENV", "CONFIGURATION_DIR", "MAX_CONCURRENT_STREAMS", "AGENT_ANSWER_DELIVERY"):
        monkeypatch.delenv(key, raising=False)

    settings = ProcessSettings.for_role("toolbox")

    assert settings.environment == "debug"
    assert str(settings.configuration_directory) == "config"
    assert settings.max_concurrent_streams == 200
    assert settings.answer_delivery is AnswerDelivery.STREAM


def test_an_unknown_answer_delivery_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("AGENT_ANSWER_DELIVERY", "sometimes")

    with pytest.raises(ValueError):
        ProcessSettings.for_role("agent-orchestrator")
