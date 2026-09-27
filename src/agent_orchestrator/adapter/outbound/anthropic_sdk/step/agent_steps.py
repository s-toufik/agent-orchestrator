from dataclasses import dataclass

from agent_orchestrator.adapter.outbound.anthropic_sdk.step.act_step import ActStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.clarify_step import ClarifyStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.context_step import ContextStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.feedback_step import FeedbackStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.finalize_step import FinalizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.ingest_step import IngestStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.plan_step import PlanStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.reflect_step import ReflectStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.summarize_step import SummarizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep


@dataclass(frozen=True, slots=True)
class AgentSteps:
    ingest: IngestStep
    context: ContextStep
    clarify: ClarifyStep
    plan: PlanStep
    act: ActStep
    tools: ToolsStep
    reflect: ReflectStep
    feedback: FeedbackStep
    finalize: FinalizeStep
    summarize: SummarizeStep
