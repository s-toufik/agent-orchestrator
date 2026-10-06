from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from tests.evaluation.harness.quality import quality_score


async def test_answer_does_not_contradict_a_known_fact(
    run_agent, judge_model, threshold, scorecard
) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, _ = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.text,
        context=["12 multiplied by 7 equals 84."],
    )
    metric = HallucinationMetric(model=judge_model)

    scorecard(
        await quality_score("freedom from hallucination: 12 * 7", metric, test_case, threshold)
    )
