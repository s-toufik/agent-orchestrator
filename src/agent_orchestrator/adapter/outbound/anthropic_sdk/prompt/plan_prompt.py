from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.clock import utc_now
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.tool_catalog import tool_catalog
from agent_orchestrator.domain.model.tool_specification import ToolSpecification

_SYSTEM = """\
You are BlueAI, an assistant for the Blue homelab platform.
Right now your only job is to write a plan. Do not run anything and do not answer the task.

Task:
{task}
{revision}
Tools available when the plan runs:
{tools}

Write the plan in Markdown, exactly in this shape:
## Plan
1. <what this step does> (tool: <tool name, or none>)
2. ...
## Expected result
One sentence.

Use only the tools listed. Use the fewest steps that get the job done.
For heavy analysis (many rows, aggregation, joins, statistics), plan a python_executor step.

Current UTC time: {now}"""

_REVISION = """
The user asked to change this previous plan:
{previous}
"""

_APPROVAL = """\
{steps}

---
Reply **yes** to run this plan, or tell me what to change."""


def plan_system_prompt(
    task: str, tools: list[ToolSpecification], previous_steps: str | None = None
) -> str:
    return _SYSTEM.format(
        task=task,
        revision=_REVISION.format(previous=previous_steps) if previous_steps else "",
        tools=tool_catalog(tools),
        now=utc_now(),
    )


def plan_approval_message(steps: str) -> str:
    return _APPROVAL.format(steps=steps)
