import asyncio
import logging
from functools import cached_property

from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.application_configuration.model.connector import TelemetryConnector
from pycraftcore.http.port import AsyncHttpFactory
from pycraftcore.logger import configure_logging
from pycraftcore.logger.adapter import StandardLogger
from pycraftcore.logger.port import Logger
from pycraftcore.repository.port import AsyncRepositoryFactory
from pycraftcore.telemetry.adapter import OpenTelemetryProvider
from pycraftcore.telemetry.port import TelemetryProvider

from bootstrap.configuration.application_configuration import SetApplicationConfiguration
from bootstrap.configuration.application_logger import create_logger
from bootstrap.configuration.settings import ProcessSettings


class BaseDI:
    def __init__(self, settings: ProcessSettings) -> None:
        self._settings = settings
        self._clients: list[AsyncHttpFactory] = []
        self._repositories: list[AsyncRepositoryFactory] = []
        self._telemetry_log_handler: logging.Handler | None = None

    @cached_property
    def _logging(self) -> Logger:
        configure_logging(self._settings.log_level)
        return create_logger(StandardLogger())

    @cached_property
    def _configuration(self) -> ApplicationConfiguration:
        return SetApplicationConfiguration(self._settings, self._logging)()

    @cached_property
    def _telemetry_provider(self) -> TelemetryProvider:
        connector: TelemetryConnector = self._configuration.connector.telemetry("open_telemetry")
        provider = OpenTelemetryProvider(
            service_name=f"{self._settings.role}-service",
            environment=self._configuration.env,
            otlp_endpoint=f"{connector.host}:{connector.port}"
            if all([connector.host, connector.port])
            else None,
        )
        self._telemetry_log_handler = provider.log_handler()
        logging.getLogger().addHandler(self._telemetry_log_handler)
        return provider

    def _register_client(self, client: AsyncHttpFactory) -> AsyncHttpFactory:
        self._clients.append(client)
        return client

    def _register_repository(self, repository: AsyncRepositoryFactory) -> AsyncRepositoryFactory:
        self._repositories.append(repository)
        return repository

    async def _start_factories(self) -> None:
        for client in self._clients:
            if hasattr(client, "start"):
                await client.start()
        for repository in self._repositories:
            if hasattr(repository, "connect"):
                await repository.connect()

    async def _stop_factories(self) -> None:
        for client in self._clients:
            if hasattr(client, "close"):
                await client.close()
        for repository in self._repositories:
            if hasattr(repository, "disconnect"):
                await repository.disconnect()

    async def _shutdown_telemetry(self) -> None:
        provider = self.__dict__.pop("_telemetry_provider", None)
        if provider is not None:
            if self._telemetry_log_handler is not None:
                logging.getLogger().removeHandler(self._telemetry_log_handler)
            await asyncio.to_thread(provider.shutdown)
