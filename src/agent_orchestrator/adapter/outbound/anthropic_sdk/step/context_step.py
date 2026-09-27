from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.context_prompt import (
    context_request,
    context_system_prompt,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

# A pending plan survives only while the user is still discussing it.
_KEEPS_PENDING_PLAN: frozenset[Intent] = frozenset({Intent.PLAN_REVISION, Intent.AMBIGUOUS})


class ContextStep:
    def __init__(self, structured_output: StructuredOutput, history_exchanges: int = 3) -> None:
        self._structured_output = structured_output
        self._history_exchanges = history_exchanges

    async def run(
        self,
        profile: ModelProfile,
        agent_state: AgentState,
        user_message: str,
        channel: EventChannel,
    ) -> TurnState:
        await channel.emit(AgentMessageStream.status("Understanding your request"))
        context = await self._understand(profile, agent_state, user_message)
        return self._resolve(user_message, agent_state, context)

    async def _understand(
        self, profile: ModelProfile, agent_state: AgentState, user_message: str
    ) -> TurnContext:
        pending = agent_state.pending_plan
        context: TurnContext | None = await self._structured_output.invoke_structured(
            profile=profile,
            system=context_system_prompt(TurnContext.model_json_schema()),
            prompt=context_request(
                self._history(agent_state), pending.steps if pending else None, user_message
            ),
            schema=TurnContext,
        )
        # Falling back to a task is safe: it leads to a plan the user must approve.
        return context or TurnContext(intent=Intent.TASK, standalone_query=user_message)

    @staticmethod
    def _resolve(user_message: str, agent_state: AgentState, context: TurnContext) -> TurnState:
        pending = agent_state.pending_plan

        if context.intent is Intent.PLAN_APPROVAL and pending is None:
            context.intent = Intent.DIRECT
        if context.intent is Intent.AMBIGUOUS and not context.clarification_question:
            context.intent = Intent.TASK
        if context.intent not in _KEEPS_PENDING_PLAN:
            agent_state.pending_plan = None

        turn = TurnState(user_message, agent_state, context, mode=TurnMode.DIRECT)
        match context.intent:
            case Intent.PLAN_APPROVAL:
                turn.mode, turn.plan = TurnMode.EXECUTE, pending
                context.standalone_query = pending.task if pending else context.standalone_query
            case Intent.TASK | Intent.PLAN_REVISION:
                turn.mode = TurnMode.PLAN
            case Intent.AMBIGUOUS:
                turn.mode = TurnMode.CLARIFY
        return turn

    def _history(self, agent_state: AgentState) -> str:
        exchanges = agent_state.exchanges[-self._history_exchanges :]
        if not exchanges:
            return "(none)"
        return "\n".join(
            f"User: {exchange.user}\nAssistant: {exchange.assistant}" for exchange in exchanges
        )
