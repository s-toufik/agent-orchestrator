from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.evaluation.harness.quality import quality_score
from tests.evaluation.support import retrieval_context


async def test_answer_is_faithful_to_the_tools_own_output(
    run_agent, judge_model, threshold, scorecard
) -> None:
    question = "Use the python tool to compute 12 * 7 and tell me the result."
    message, state = await run_agent(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=message.text,
        retrieval_context=retrieval_context(state),
    )
    metric = FaithfulnessMetric(model=judge_model)

    scorecard(
        await quality_score("faithfulness to the tool output: 12 * 7", metric, test_case, threshold)
    )
