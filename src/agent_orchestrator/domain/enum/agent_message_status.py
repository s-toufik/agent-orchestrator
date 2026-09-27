from enum import StrEnum


class MessageStreamType(StrEnum):
    TOKEN = "token"
    STATUS = "status"
    RESET = "reset"
    FINAL = "final"
    ERROR = "error"
    COMPLETE = "complete"
