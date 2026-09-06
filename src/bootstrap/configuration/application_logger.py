from typing import cast

from pycraftcore.logger.port import Logger


def create_logger(logger: Logger | None) -> Logger:
    if logger is None:
        from logging import getLogger

        return cast(Logger, getLogger(__name__))
    return logger
