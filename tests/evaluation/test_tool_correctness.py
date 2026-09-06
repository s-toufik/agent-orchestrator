import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests.evaluation.support import tools_called

pytestmark = pytest.mark.evaluation


async def test_agent_uses_the_python_tool_for_arithmetic(run_agent, judge_model) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.content,
        tools_called=tools_called(state),
        expected_tools=[ToolCall(name="python_executor")],
    )
    metric = ToolCorrectnessMetric(model=judge_model, threshold=1.0)

    assert_test(test_case, [metric])


async def test_agent_writes_then_reads_back_a_file(run_agent, judge_model) -> None:
    question = (
        "Write the text 'hello deepeval' to a file named notes.txt, "
        "then read that file back to confirm its contents."
    )
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.content,
        tools_called=tools_called(state),
        expected_tools=[ToolCall(name="file_writer"), ToolCall(name="file_reader")],
    )
    metric = ToolCorrectnessMetric(model=judge_model, threshold=1.0, should_consider_ordering=True)

    assert_test(test_case, [metric])
