from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict, VerdictAction


def turn(
    intent: Intent | None = Intent.DIRECT,
    max_steps: int = 5,
    max_retries: int = 2,
    plan: Plan | None = None,
) -> Turn:
    created = Turn(
        request="msg",
        model="m",
        settings=TurnSettings(max_steps=max_steps, max_retries=max_retries),
    )
    if intent is not None:
        created.understood(Understanding(intent=intent, query="the query"))
    if plan is not None:
        created.execute(plan)
    return created


def answer(text: str = "the answer") -> Draft:
    return Draft(text=text)


def calling(*names: str) -> Draft:
    return Draft(
        text="", tool_calls=tuple(ToolCall(id=f"c{i}", name=n) for i, n in enumerate(names))
    )


def result(name: str = "echo", output: str = "ok") -> ToolResult:
    return ToolResult(call_id="c0", tool_name=name, output=output)


def retry(critique: str = "too short") -> Verdict:
    return Verdict(VerdictAction.RETRY, critique)


def accept() -> Verdict:
    return Verdict.accept()


PLAN = Plan(task="count the rows", steps="1. count (tool: echo)")
