from deepeval.test_case import ToolCall

from agent_orchestrator.domain.turn.turn import Turn


def tools_called(turn: Turn) -> list[ToolCall]:
    return [ToolCall(name=action.tool) for action in turn.actions]


def retrieval_context(turn: Turn) -> list[str]:
    return [result.content for result in turn.evidence]
