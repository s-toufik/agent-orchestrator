from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests.evaluation.harness.quality import quality_score
from tests.evaluation.support import tools_called


async def test_agent_uses_the_python_tool_for_arithmetic(
    run_agent, judge_model, threshold, scorecard
) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.text,
        tools_called=tools_called(state),
        expected_tools=[ToolCall(name="python_executor")],
    )
    metric = ToolCorrectnessMetric(model=judge_model)

    scorecard(
        await quality_score("tool correctness: python for arithmetic", metric, test_case, threshold)
    )


async def test_agent_writes_then_reads_back_a_file(
    run_agent, judge_model, threshold, scorecard
) -> None:
    question = (
        "Write the text 'hello deepeval' to a file named notes.txt, "
        "then read that file back to confirm its contents."
    )
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.text,
        tools_called=tools_called(state),
        expected_tools=[ToolCall(name="file_writer"), ToolCall(name="file_reader")],
    )
    metric = ToolCorrectnessMetric(model=judge_model, should_consider_ordering=True)

    scorecard(
        await quality_score(
            "tool correctness: write then read a file, in order", metric, test_case, threshold
        )
    )
