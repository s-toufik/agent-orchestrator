import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.evaluation.support import retrieval_context

pytestmark = pytest.mark.evaluation


async def test_answer_is_faithful_to_the_tools_own_output(run_agent, judge_model) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.content,
        retrieval_context=retrieval_context(state),
    )
    metric = FaithfulnessMetric(model=judge_model, threshold=0.5)

    assert_test(test_case, [metric])
