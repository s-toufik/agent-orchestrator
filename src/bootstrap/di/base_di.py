import asyncio
from functools import cached_property

from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.http.port import AsyncHttpFactory
from pycraftcore.logger.adapter import LoguruLogger
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

    @cached_property
    def _logging(self) -> Logger:
        return create_logger(LoguruLogger())

    @cached_property
    def _configuration(self) -> ApplicationConfiguration:
        return SetApplicationConfiguration(self._settings, self._logging)()

    @cached_property
    def _telemetry_provider(self) -> TelemetryProvider:
        return OpenTelemetryProvider(
            service_name=f"{self._settings.role}-service",
            environment=self._configuration.env,
        )

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
            # TracerProvider.shutdown() flushes buffered spans and can block on I/O.
            await asyncio.to_thread(provider.shutdown)
