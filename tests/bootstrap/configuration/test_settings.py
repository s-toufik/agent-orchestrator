import bootstrap.configuration.settings as settings_module
from bootstrap.configuration.settings import ProcessSettings


def test_for_role_uses_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CONFIGURATION_DIR", "/custom/config")
    monkeypatch.setenv("AGENT_HOST", "127.0.0.1")
    monkeypatch.setenv("AGENT_PORT", "9000")

    settings = ProcessSettings.for_role("agent", default_port=8000)

    assert settings.role == "agent"
    assert settings.environment == "prod"
    assert str(settings.configuration_directory) == "/custom/config"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000


def test_for_role_falls_back_to_defaults_when_nothing_is_set(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    for key in ("APP_ENV", "CONFIGURATION_DIR", "TOOLBOX_HOST", "TOOLBOX_PORT"):
        monkeypatch.delenv(key, raising=False)

    settings = ProcessSettings.for_role("toolbox", default_port=8001)

    assert settings.environment == "debug"
    assert str(settings.configuration_directory) == "config"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8001


def test_each_role_reads_its_own_host_and_port_prefix(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.delenv("AGENT_HOST", raising=False)
    monkeypatch.setenv("TOOLBOX_HOST", "10.0.0.5")

    agent_settings = ProcessSettings.for_role("agent", default_port=8000)
    toolbox_settings = ProcessSettings.for_role("toolbox", default_port=8001)

    assert agent_settings.host == "0.0.0.0"
    assert toolbox_settings.host == "10.0.0.5"
