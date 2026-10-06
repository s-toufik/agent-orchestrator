from agent_orchestrator.domain.turn.plan import Plan, PlannedStep
from agent_orchestrator.domain.turn.plan_progress import PlanProgress
from tests.agent_orchestrator.domain.builders import failure, result

PLAN = Plan(
    "report",
    (
        PlannedStep("read the data", "file_reader"),
        PlannedStep("pick the scenarios"),
        PlannedStep("compute", "python_executor"),
        PlannedStep("write the report", "file_writer"),
    ),
)


def test_a_new_plan_starts_at_its_first_step() -> None:
    progress = PlanProgress.of(PLAN, [])

    assert (progress.done, progress.number) == (0, 1)
    assert progress.current == PLAN.steps[0]


def test_a_success_of_the_current_steps_tool_completes_it_and_skips_reasoning_steps() -> None:
    progress = PlanProgress.of(PLAN, [result("file_reader")])

    assert progress.current == PlannedStep("compute", "python_executor")
    assert progress.number == 3


def test_a_failure_or_another_tool_leaves_the_step_open() -> None:
    progress = PlanProgress.of(
        PLAN, [result("file_reader"), failure("python_executor"), result("file_writer")]
    )

    assert progress.current == PlannedStep("compute", "python_executor")


def test_every_step_done_completes_the_plan() -> None:
    progress = PlanProgress.of(
        PLAN, [result("file_reader"), result("python_executor"), result("file_writer")]
    )

    assert progress.is_complete and progress.current is None


def test_a_plan_without_tools_is_complete_at_once() -> None:
    assert PlanProgress.of(Plan("q", (PlannedStep("think"),)), []).is_complete


def test_the_progress_marks_done_current_and_pending_steps() -> None:
    assert PlanProgress.of(PLAN, [result("file_reader")]).render() == (
        "✓ 1. read the data (tool: file_reader)\n"
        "✓ 2. pick the scenarios (tool: none)\n"
        "→ 3. compute (tool: python_executor)\n"
        "· 4. write the report (tool: file_writer)"
    )
