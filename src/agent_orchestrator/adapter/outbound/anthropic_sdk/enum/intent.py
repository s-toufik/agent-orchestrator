from enum import StrEnum


class Intent(StrEnum):
    TASK = "task"
    CONTINUATION = "continuation"
    DIRECT = "direct"
    PLAN_APPROVAL = "plan_approval"
    PLAN_REVISION = "plan_revision"
    AMBIGUOUS = "ambiguous"
