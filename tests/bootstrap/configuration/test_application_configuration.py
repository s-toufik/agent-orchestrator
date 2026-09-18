from pathlib import Path

import pytest
from pycraftcore.application_configuration.adapter import LoadApplicationConfiguration

from bootstrap.configuration.application_configuration import SetApplicationConfiguration
from bootstrap.configuration.settings import ProcessSettings

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def make_settings(directory: Path, role: str = "agent-orchestrator") -> ProcessSettings:
    return ProcessSettings(role=role, environment="debug", configuration_directory=directory)


def _set_required_env(monkeypatch) -> None:
    # The real config tree interpolates these with no default -- they must be
    # present for OmegaConf to resolve it at all, even though this test only
    # cares about the mcp/operation values, not the database ones.
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_HOST", "/tmp")
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_NAME", "test-checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_HOST", "localhost")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PORT", "27017")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_NAME", "test-checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_USERNAME", "test")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PASSWORD", "test")


def test_loads_the_real_configuration_tree(logger, monkeypatch) -> None:
    _set_required_env(monkeypatch)

    configuration = SetApplicationConfiguration(make_settings(REAL_CONFIG_DIR), logger)()

    assert configuration.connector.mcp("toolbox").transport == "streamable_http"
    assert configuration.connector.database("sqlite_checkpointer").engine == "sqlite"
    assert configuration.connector.database("mongodb_checkpointer").engine == "mongodb"


def test_raises_when_the_configuration_directory_does_not_exist(logger, tmp_path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        SetApplicationConfiguration(make_settings(missing), logger)()

    assert logger.messages("critical")


def test_raises_a_value_error_when_the_loader_returns_nothing(logger, monkeypatch) -> None:
    monkeypatch.setattr(LoadApplicationConfiguration, "load", lambda self: None)

    with pytest.raises(ValueError, match="No configuration loaded"):
        SetApplicationConfiguration(make_settings(REAL_CONFIG_DIR), logger)()


def test_uses_a_default_stdlib_logger_when_none_given(monkeypatch) -> None:
    _set_required_env(monkeypatch)

    configuration = SetApplicationConfiguration(make_settings(REAL_CONFIG_DIR))()

    assert configuration is not None
