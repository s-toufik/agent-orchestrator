import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

pytestmark = pytest.mark.evaluation


async def test_answer_does_not_contradict_a_known_fact(run_agent, judge_model) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, _ = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.content,
        context=["12 multiplied by 7 equals 84."],
    )
    metric = HallucinationMetric(model=judge_model, threshold=0.5)

    assert_test(test_case, [metric])
