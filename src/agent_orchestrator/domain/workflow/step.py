from enum import StrEnum


class Step(StrEnum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    CLARIFY = "clarify"
    ACT = "act"
    RUN_TOOLS = "run_tools"
    REVIEW = "review"
    FEEDBACK = "feedback"
    FINISH = "finish"
    SUMMARIZE = "summarize"
    END = "end"
