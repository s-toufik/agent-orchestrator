from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.clock import utc_now
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.tool_catalog import tool_catalog
from agent_orchestrator.domain.model.tool_specification import ToolSpecification

_SYSTEM = """\
You are BlueAI, an assistant for the Blue homelab platform.
Be brief. Use only facts from tool results or from this conversation. Never invent data.
The earlier messages in this chat are your past exchanges with this user: rely on them
when the user refers to what was said before (their name, a previous question, whether
you talked already).

Request: {request}

Your tools (run only as part of a plan the user approved):
{tools}

{mode}

Answer in Markdown:
## Answer
Short answer. Use bullets or a table when clearer.
## Tool calls
Only if you used tools. Table: | Tool | Why |

Add a simple ASCII diagram only if it helps (plain characters, in a ``` block).

Current UTC time: {now}"""

_EXECUTE = """\
The user approved this plan. Carry it out step by step with your tools, then answer.
{steps}

How to work:
- For a quick look at data (a few rows, one value), call a tool directly.
- For heavy analysis (many rows, aggregation, joins, statistics), use
  python_executor. Inside it, your other tools are callable as Python functions.
- The working directory is your file storage: save large results there and read
  them back later. Use paths relative to it (e.g. "report.md"): they name the same
  file in your code and in tool calls. Give the file path in your answer.

Example inside python_executor (heavy analysis only):
    import pandas as pd
    file = file_reader(file_path="eqd_positions.csv")
    df = pd.DataFrame(file["content"]["rows"])
    ... do some analysis ...
    saved = file_writer(file_path="report.md", data=report_text)
    result = saved["path"]{builtin_tools}"""

_BUILTIN_TOOLS = """

Built-in file tools, usable only inside the working directory {working_directory}
(always pass absolute paths under it):
{tools}"""

_TOOL_PURPOSES: dict[str, str] = {
    "Read": "read a text file",
    "Write": "create or overwrite a text file",
    "Edit": "change part of a text file you have read",
    "Glob": "find files by name pattern",
    "Grep": "search inside files",
}

_DIRECT = """\
You cannot run tools in this reply. Answer from the conversation and general knowledge.
If the request needs one of your tools, name it and offer to prepare a plan."""

_FEEDBACK = """\
A reviewer rejected your last answer.

Critique:
{critique}

Write the corrected answer. Fix only what the critique flags."""


def act_system_prompt(
    request: str,
    plan_steps: str | None,
    tools: list[ToolSpecification],
    builtin_tools: str | None = None,
) -> str:
    mode: str = (
        _EXECUTE.format(steps=plan_steps, builtin_tools=builtin_tools or "")
        if plan_steps
        else _DIRECT
    )
    return _SYSTEM.format(request=request, tools=tool_catalog(tools), mode=mode, now=utc_now())


def builtin_tool_purpose(tool: str) -> str:
    return _TOOL_PURPOSES.get(tool, tool)


def builtin_tools_instructions(working_directory: str, tools: list[str]) -> str:
    return _BUILTIN_TOOLS.format(
        working_directory=working_directory,
        tools="\n".join(f"- {tool}: {builtin_tool_purpose(tool)}" for tool in tools),
    )


def act_feedback(critique: str) -> str:
    return _FEEDBACK.format(critique=critique)
