from pathlib import Path

import pytest
from pycraftcore.application_configuration.adapter import LoadApplicationConfiguration

from bootstrap.configuration.application_configuration import SetApplicationConfiguration
from bootstrap.configuration.settings import ProcessSettings

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def make_settings(directory: Path, role: str = "agent-orchestrator") -> ProcessSettings:
    return ProcessSettings(
        role=role, environment="debug", configuration_directory=directory, host="0.0.0.0", port=8000
    )


def _set_required_env(monkeypatch) -> None:
    # The real config tree interpolates these with no default -- they must be
    # present for OmegaConf to resolve it at all, even though this test only
    # cares about the mcp/operation values, not the database ones.
    monkeypatch.setenv("CHECKPOINT_DB_HOST", "/tmp")
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "test-checkpoint")


def test_loads_the_real_configuration_tree(logger, monkeypatch) -> None:
    _set_required_env(monkeypatch)

    configuration = SetApplicationConfiguration(make_settings(REAL_CONFIG_DIR), logger)()

    assert configuration.connector.mcp("toolbox").transport == "streamable_http"
    assert configuration.connector.database("checkpointer").engine == "sqlite"
    assert configuration.operation.api("gpt_oss_20b").name == "gpt-oss-20b"


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
