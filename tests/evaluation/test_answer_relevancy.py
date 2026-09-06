import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

pytestmark = pytest.mark.evaluation


async def test_answer_stays_on_topic(run_agent, judge_model) -> None:
    question = "In one sentence, what is the capital of France?"
    message, _ = await run_agent(question)

    test_case = LLMTestCase(input=question, actual_output=message.content)
    metric = AnswerRelevancyMetric(model=judge_model, threshold=0.5)

    assert_test(test_case, [metric])
