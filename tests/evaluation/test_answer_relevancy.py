from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.evaluation.harness.quality import quality_score


async def test_answer_stays_on_topic(run_agent, judge_model, threshold, scorecard) -> None:
    question = "In one sentence, what is the capital of France?"
    message, _ = await run_agent(question)

    test_case = LLMTestCase(input=question, actual_output=message.text)
    metric = AnswerRelevancyMetric(model=judge_model)

    scorecard(
        await quality_score(
            "answer relevancy: a one-sentence question", metric, test_case, threshold
        )
    )
