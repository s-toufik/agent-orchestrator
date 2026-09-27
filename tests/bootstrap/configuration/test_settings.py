import bootstrap.configuration.settings as settings_module
from bootstrap.configuration.settings import AgentEngine, ProcessSettings


def test_for_role_uses_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CONFIGURATION_DIR", "/custom/config")
    monkeypatch.setenv("AGENT_ENGINE", "anthropic_sdk")
    monkeypatch.setenv("MAX_CONCURRENT_STREAMS", "12")

    settings = ProcessSettings.for_role("agent-orchestrator")

    assert settings.role == "agent-orchestrator"
    assert settings.environment == "prod"
    assert str(settings.configuration_directory) == "/custom/config"
    assert settings.engine is AgentEngine.ANTHROPIC_SDK
    assert settings.max_concurrent_streams == 12


def test_for_role_falls_back_to_defaults_when_nothing_is_set(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    for key in ("APP_ENV", "CONFIGURATION_DIR", "AGENT_ENGINE", "MAX_CONCURRENT_STREAMS"):
        monkeypatch.delenv(key, raising=False)

    settings = ProcessSettings.for_role("toolbox")

    assert settings.environment == "debug"
    assert str(settings.configuration_directory) == "config"
    assert settings.engine is AgentEngine.LANGGRAPH
    assert settings.max_concurrent_streams == 200
