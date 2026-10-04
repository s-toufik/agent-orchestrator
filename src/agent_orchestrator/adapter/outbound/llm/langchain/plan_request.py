from typing import Any

from agent_orchestrator.domain.turn.draft import Draft

REQUEST_PLAN: str = "request_plan"

REQUEST_PLAN_TOOL: dict[str, Any] = {
    "name": REQUEST_PLAN,
    "description": "Ask for a plan when the answer needs to run one of your tools.",
    "parameters": {
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "What the answer needs."}},
        "required": ["reason"],
    },
}


def direct_draft(draft: Draft) -> Draft:
    if not draft.asks_for_tools:
        return draft
    requested = next((call for call in draft.tool_calls if call.name == REQUEST_PLAN), None)
    if requested is not None:
        return Draft.asking_for_plan(str(requested.arguments.get("reason", "")))
    return Draft.asking_for_plan(f"run {', '.join(call.name for call in draft.tool_calls)}")
