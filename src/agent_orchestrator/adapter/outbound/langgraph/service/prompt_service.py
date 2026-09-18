class PromptService:
    # ---------------------------------------------------------------- shared
    _STYLE = "Be short. No filler, no restating the question, no repeating these rules."

    _TOOL_USAGE = """Tools:
- Always call tools directly for inspection and light analysis: reading a file, a quick \
query, a small check.
- For heavy analysis (combining steps, transforming data, generating files), always use \
python_executor. Inside it, call the other tools again to do the heavy work.
- Never do heavy analysis with direct tool calls, and never skip python_executor for it."""

    # -------------------------------------------------------------- planner
    _PLANNER_ROLE = "You are the planner of a ReAct agent. You decide the next action."

    _PLANNER_RULES = """Rules:
- Never guess. If unsure, use a tool or ask the user.
- Look at the data or schema before you query it.
- Independent tool calls can be made in parallel.
- Give a one-line reason before you act.
- Stop calling tools as soon as you have enough evidence to answer.

Each turn, do exactly one of:
1. TOOL_CALL - call one or more tools.
2. FINAL - answer the user."""

    # ------------------------------------------------------------ reflection
    _REFLECTION_ROLE = (
        "You are the reflection step. You check the assistant's final answer, not the "
        "tools it used to get there."
    )

    _REFLECTION_CHECKS = """Check the answer against the user's question and the evidence gathered:
- Does it answer the question?
- Is it correct and fully supported by the evidence?
- Is it complete, with no unsupported claims?

Do not judge which tools were called or how.
Use "accept" if all checks pass. Use "retry" otherwise, with a short, concrete critique."""

    _REFLECTION_OUTPUT = "Return only this JSON, nothing else:\n{output_format}"

    # -------------------------------------------------------------- feedback
    _FEEDBACK_TEMPLATE = """You are retrying a previous answer. Fix only this:
{critiques}

Do not repeat reasoning that was already correct."""

    @staticmethod
    def _compose(*sections: str) -> str:
        return "\n\n".join(section.strip() for section in sections if section.strip())

    @staticmethod
    def planner_system_prompt() -> str:
        return PromptService._compose(
            PromptService._PLANNER_ROLE,
            PromptService._STYLE,
            PromptService._TOOL_USAGE,
            PromptService._PLANNER_RULES,
        )

    @staticmethod
    def reflection_system_prompt(output_format: object) -> str:
        return PromptService._compose(
            PromptService._REFLECTION_ROLE,
            PromptService._STYLE,
            PromptService._REFLECTION_CHECKS,
            PromptService._REFLECTION_OUTPUT.format(output_format=output_format),
        )

    @staticmethod
    def feedback_system_prompt(critique: str) -> str:
        return PromptService._compose(
            PromptService._STYLE,
            PromptService._FEEDBACK_TEMPLATE.format(critiques=critique),
        )
