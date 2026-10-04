from enum import StrEnum


class MessageStreamType(StrEnum):
    TOKEN = "token"
    STATUS = "status"
    FINAL = "final"
    ERROR = "error"
    COMPLETE = "complete"
