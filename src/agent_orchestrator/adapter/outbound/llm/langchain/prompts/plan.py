from agent_orchestrator.adapter.outbound.llm.langchain.prompts.clock import utc_now
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.tool_catalog import tool_catalog
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification

_SYSTEM = """\
You are BlueAI, an assistant for the Blue homelab platform.
Right now your only job is to write a plan. Do not run anything and do not answer the task.

Task:
{task}
{revision}
Tools available when the plan runs:
{tools}

How to plan:
- Each step makes exactly one tool call: name that tool exactly as listed above.
  A step that only reasons over earlier results has no tool (null).
- Use only the tools listed. Use the fewest steps that get the job done.
- Reuse the work done earlier in this conversation (files written, results found)
  instead of doing it again.
- For heavy analysis (many rows, aggregation, joins, statistics), plan a python_executor step.
- To put a plot in a Markdown report, save the image in the working directory and link
  it from the report: ![Title](plot.svg).

Return only this JSON, nothing else:
{output_format}

Current UTC time: {now}"""

_REVISION = """
The user asked to change this previous plan:
{previous}
"""


def plan_system_prompt(
    task: str,
    tools: list[ToolSpecification],
    output_format: object,
    previous_plan: str | None = None,
) -> str:
    return _SYSTEM.format(
        task=task,
        revision=_REVISION.format(previous=previous_plan) if previous_plan else "",
        tools=tool_catalog(tools),
        output_format=output_format,
        now=utc_now(),
    )
