import logging

from bootstrap.configuration.application_logger import create_logger


def test_returns_the_given_logger_unchanged(logger) -> None:
    assert create_logger(logger) is logger


def test_falls_back_to_a_stdlib_logger_when_none_given() -> None:
    result = create_logger(None)

    assert isinstance(result, logging.Logger)
