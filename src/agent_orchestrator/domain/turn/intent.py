from enum import StrEnum


class Intent(StrEnum):
    TASK = "task"
    CONTINUATION = "continuation"
    DIRECT = "direct"
    PLAN_APPROVAL = "plan_approval"
    PLAN_REVISION = "plan_revision"
    AMBIGUOUS = "ambiguous"

    @property
    def needs_plan(self) -> bool:
        return self in (Intent.TASK, Intent.PLAN_REVISION)

    @property
    def keeps_pending_plan(self) -> bool:
        return self in (Intent.PLAN_REVISION, Intent.AMBIGUOUS)
