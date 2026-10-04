from typing import Any, TypedDict


class GraphState(TypedDict):
    conversation: dict[str, Any]
    turn: dict[str, Any]
