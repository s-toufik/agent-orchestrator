from dataclasses import dataclass

from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.workflow.step import Step
from tests.evaluation.harness.expectation import Expect

UNDERSTAND, PLAN, CLARIFY, ACT = Step.UNDERSTAND, Step.PLAN, Step.CLARIFY, Step.ACT
RUN_TOOLS, REVIEW, FINISH, SUMMARIZE = Step.RUN_TOOLS, Step.REVIEW, Step.FINISH, Step.SUMMARIZE

PROPOSED = (UNDERSTAND, PLAN, FINISH, SUMMARIZE)


@dataclass(frozen=True, slots=True)
class Say:
    message: str
    expect: Expect
    auto_approve: bool = False


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    says: tuple[Say, ...]


CASES: tuple[Case, ...] = (
    Case(
        "a general question is answered directly, without tools or review",
        (
            Say(
                "In one sentence, what is the capital of France?",
                Expect(
                    intents=(Intent.DIRECT,),
                    outcome=Outcome.ANSWERED,
                    route=(UNDERSTAND, ACT, FINISH, SUMMARIZE),
                    tools=(),
                    answer_contains=("Paris",),
                ),
            ),
        ),
    ),
    Case(
        "a follow-up is understood against the conversation",
        (
            Say("In one sentence, what is Apache Kafka?", Expect(outcome=Outcome.ANSWERED)),
            Say(
                "Which company first created it?",
                Expect(
                    intents=(Intent.CONTINUATION, Intent.DIRECT),
                    outcome=Outcome.ANSWERED,
                    tools=(),
                    answer_contains=("LinkedIn",),
                ),
            ),
        ),
    ),
    Case(
        "an ambiguous message gets a clarifying question",
        (
            Say(
                "Show me the numbers.",
                Expect(
                    intents=(Intent.AMBIGUOUS,),
                    outcome=Outcome.CLARIFICATION,
                    route=(UNDERSTAND, CLARIFY, FINISH, SUMMARIZE),
                    tools=(),
                ),
            ),
        ),
    ),
    Case(
        "a task waits for approval, then runs to the end of its plan",
        (
            Say(
                "Use the python tool to compute 12 * 7.",
                Expect(
                    intents=(Intent.TASK,),
                    outcome=Outcome.AWAITING_APPROVAL,
                    route=PROPOSED,
                    tools=(),
                    plan_tools=frozenset({"python_executor"}),
                    plan_pending=True,
                ),
            ),
            Say(
                "yes",
                Expect(
                    intents=(Intent.PLAN_APPROVAL,),
                    outcome=Outcome.ANSWERED,
                    tools=("python_executor",),
                    answer_contains=("84",),
                    plan_pending=False,
                ),
            ),
        ),
    ),
    Case(
        "a requested change gives a new plan, still waiting for approval",
        (
            Say(
                "Write the text 'hello' to a file named eval_first.txt.",
                Expect(outcome=Outcome.AWAITING_APPROVAL, plan_tools=frozenset({"file_writer"})),
            ),
            Say(
                "Change the plan: name the file eval_second.txt instead.",
                Expect(
                    intents=(Intent.PLAN_REVISION,),
                    outcome=Outcome.AWAITING_APPROVAL,
                    route=PROPOSED,
                    tools=(),
                    plan_contains=("eval_second.txt",),
                    plan_pending=True,
                ),
            ),
        ),
    ),
    Case(
        "moving on to another question drops the pending plan",
        (
            Say(
                "Write the text 'hello' to a file named eval_dropped.txt.",
                Expect(outcome=Outcome.AWAITING_APPROVAL),
            ),
            Say(
                "Never mind that. In one sentence, what is a CPU?",
                Expect(
                    intents=(Intent.DIRECT,),
                    outcome=Outcome.ANSWERED,
                    tools=(),
                    plan_pending=False,
                ),
            ),
        ),
    ),
    Case(
        "with auto-approve, a task is planned and carried out in one turn",
        (
            Say(
                "Use the python tool to compute 15 * 4.",
                Expect(
                    outcome=Outcome.ANSWERED,
                    tools=("python_executor",),
                    answer_contains=("60",),
                    plan_pending=False,
                ),
                auto_approve=True,
            ),
        ),
    ),
    Case(
        "a plan of several steps runs every step, in order",
        (
            Say(
                "Write the text 'hello evaluation' to a file named eval_readback.txt, "
                "then read that file back and tell me what it contains.",
                Expect(
                    outcome=Outcome.ANSWERED,
                    tools=("file_writer", "file_reader"),
                    plan_tools=frozenset({"file_writer", "file_reader"}),
                    answer_contains=("hello evaluation",),
                ),
                auto_approve=True,
            ),
        ),
    ),
    Case(
        "the work of an earlier turn is remembered, not redone",
        (
            Say(
                "Write the text 'remember me' to a file named eval_memory.txt.",
                Expect(outcome=Outcome.ANSWERED, tools=("file_writer",)),
                auto_approve=True,
            ),
            Say(
                "Without running any tool: which file did you just write?",
                Expect(outcome=Outcome.ANSWERED, tools=(), answer_contains=("eval_memory.txt",)),
            ),
        ),
    ),
)
