from typing import Protocol, runtime_checkable

from pycraftcore.application_configuration import ApplicationConfiguration
from pycraftcore.logger.port import Logger


@runtime_checkable
class Container(Protocol):
    @property
    def logging(self) -> Logger: ...

    @property
    def application_configuration(self) -> ApplicationConfiguration: ...

    async def boot(self) -> None: ...

    async def stop(self) -> None: ...
