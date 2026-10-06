import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.evaluation.harness.route_coverage import RouteCoverage
from tests.evaluation.harness.score import Score


class EvaluationReport:
    def __init__(self, model: str, threshold: float, coverage: RouteCoverage) -> None:
        self.model = model
        self.threshold = threshold
        self._coverage = coverage
        self.scores: list[Score] = []
        self.unscored: list[tuple[str, str]] = []

    def add(self, score: Score) -> None:
        self.scores.append(score)

    def add_unscored(self, test: str, error: str) -> None:
        self.unscored.append((test, error))

    @property
    def is_empty(self) -> bool:
        return not self.scores and not self.unscored

    @property
    def mean(self) -> float:
        return sum(score.value for score in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def passed(self) -> int:
        return sum(score.passed for score in self.scores)

    def write(self, directory: Path, now: datetime) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{now:%Y%m%d-%H%M%S}_{re.sub(r'[^A-Za-z0-9.-]+', '-', self.model)}"
        markdown, data = directory / f"{stem}.md", directory / f"{stem}.json"
        markdown.write_text(self.markdown(now), encoding="utf-8")
        data.write_text(json.dumps(self.as_dict(now), indent=2) + "\n", encoding="utf-8")
        return [markdown, data]

    def summary(self) -> list[str]:
        return [
            f"model {self.model}, threshold {self.threshold:.2f}",
            f"mean score {self.mean:.2f}, {self.passed}/{len(self.scores)} at or above the threshold",
            *([f"{len(self.unscored)} not scored: they failed to run"] if self.unscored else []),
        ]

    def markdown(self, now: datetime) -> str:
        lines = [
            f"# Agent evaluation: {self.model}",
            "",
            f"{now:%Y-%m-%d %H:%M} · threshold **{self.threshold:.2f}** · "
            f"mean score **{self.mean:.2f}** · "
            f"**{self.passed}/{len(self.scores)}** at or above the threshold"
            + (f" · **{len(self.unscored)}** not scored" if self.unscored else ""),
            "",
            "| Score | Result | Kind | Test |",
            "|---:|---|---|---|",
            *(
                f"| {score.value:.2f} | {'pass' if score.passed else 'FAIL'} "
                f"| {score.kind} | {score.name} |"
                for score in sorted(self.scores, key=lambda score: score.value)
            ),
        ]
        if self.unscored:
            lines += [
                "",
                "## Not scored",
                "",
                "These failed to run, so they are left out of the mean.",
                "",
            ]
            lines += [f"- **{test}**: {error}" for test, error in self.unscored]
        failing = [score for score in self.scores if score.failures]
        if failing:
            lines += ["", "## What lowered the scores"]
            for score in failing:
                lines += ["", f"### {score.name} ({score.value:.2f})", ""]
                lines += [f"- {failure}" for failure in score.failures]
        lines += ["", "## Route coverage", "", "```text", *self._coverage.report(), "```", ""]
        return "\n".join(lines)

    def as_dict(self, now: datetime) -> dict[str, Any]:
        return {
            "model": self.model,
            "run_at": now.isoformat(timespec="seconds"),
            "threshold": self.threshold,
            "mean_score": round(self.mean, 4),
            "passed": self.passed,
            "total": len(self.scores),
            "scores": [
                {
                    "name": score.name,
                    "kind": str(score.kind),
                    "score": round(score.value, 4),
                    "passed": score.passed,
                    "failures": list(score.failures),
                }
                for score in self.scores
            ],
            "unscored": [{"name": test, "error": error} for test, error in self.unscored],
            "route_coverage": {
                "missing": [f"{source} -> {target}" for source, target in self._coverage.missing()]
            },
        }
