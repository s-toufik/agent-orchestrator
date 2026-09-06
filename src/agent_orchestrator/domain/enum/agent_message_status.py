from enum import StrEnum


class MessageStreamType(StrEnum):
    TOKEN = "token"
    COMPLETE = "complete"
    ERROR = "error"
    FINAL = "final"
