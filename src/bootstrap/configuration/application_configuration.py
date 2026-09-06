from pathlib import Path

from pycraftcore.application_configuration import (
    ApplicationConfiguration,
    Configuration,
    ConfigurationReader,
)
from pycraftcore.application_configuration.adapter import (
    LoadApplicationConfiguration,
    OmegaConfigurationReader,
)
from pycraftcore.application_configuration.enum import RunTypeEnvironment
from pycraftcore.logger.port import Logger

from bootstrap.configuration.application_logger import create_logger
from bootstrap.configuration.settings import ProcessSettings


class SetApplicationConfiguration:
    """Loads the YAML configuration tree for one process role."""

    def __init__(self, settings: ProcessSettings, logger: Logger | None = None) -> None:
        self._settings = settings
        self._logger = create_logger(logger)

    def __call__(self) -> ApplicationConfiguration:
        environment = RunTypeEnvironment(self._settings.environment)
        directory: Path = self._settings.configuration_directory

        self._logger.info(
            f"Loading '{self._settings.role}' configuration ({environment.value}) from {directory}"
        )

        if not directory.exists():
            exception = FileNotFoundError(f"Configuration directory not found: {directory}")
            self._logger.critical(str(exception))
            raise exception

        reader: ConfigurationReader = OmegaConfigurationReader(environment, directory)
        loader: Configuration = LoadApplicationConfiguration(reader, self._logger)

        configuration = loader.load()
        if configuration is None:
            exception = ValueError("No configuration loaded")
            self._logger.critical(str(exception))
            raise exception

        return configuration
