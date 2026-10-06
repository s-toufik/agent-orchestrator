from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

from tests.evaluation.harness.score import Score, ScoreKind


async def quality_score(
    name: str,
    metric: BaseMetric,
    test_case: LLMTestCase,
    threshold: float,
) -> Score:
    value = await metric.a_measure(test_case)
    reason = metric.reason or "no reason given"
    return Score(
        name=name,
        kind=ScoreKind.QUALITY,
        value=value,
        threshold=threshold,
        failures=() if value >= threshold else (f"{metric.__name__}: {reason}",),
    )
