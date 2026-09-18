import bootstrap.configuration.settings as settings_module
from bootstrap.configuration.settings import ProcessSettings


def test_for_role_uses_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CONFIGURATION_DIR", "/custom/config")

    settings = ProcessSettings.for_role("agent-orchestrator")

    assert settings.role == "agent-orchestrator"
    assert settings.environment == "prod"
    assert str(settings.configuration_directory) == "/custom/config"


def test_for_role_falls_back_to_defaults_when_nothing_is_set(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    for key in ("APP_ENV", "CONFIGURATION_DIR"):
        monkeypatch.delenv(key, raising=False)

    settings = ProcessSettings.for_role("toolbox")

    assert settings.environment == "debug"
    assert str(settings.configuration_directory) == "config"
