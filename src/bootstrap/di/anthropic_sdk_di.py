import base64
from functools import cached_property

from anthropic import AsyncAnthropic
from claude_agent_sdk.types import EffortLevel
from pycraftcore.application_configuration.model.connector import McpConnector, TelemetryConnector
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.token_auth import TokenAuth
from pymongo import AsyncMongoClient

from agent_orchestrator.adapter.outbound.anthropic_sdk.anthropic_sdk_agent import AnthropicSdkAgent
from agent_orchestrator.adapter.outbound.anthropic_sdk.options_factory import OptionsFactory
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_roles import ModelRoles
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.act_step import ActStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.agent_steps import AgentSteps
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.clarify_step import ClarifyStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.context_step import ContextStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.feedback_step import FeedbackStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.finalize_step import FinalizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.ingest_step import IngestStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.plan_step import PlanStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.reflect_step import ReflectStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.summarize_step import SummarizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.mongo_agent_state_store import (
    MongoAgentStateStore,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.mongo_session_store import (
    MongoSessionStore,
)
from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort
from agent_orchestrator.adapter.outbound.llm.lite_llm.lite_llm_config import lite_llm_config
from agent_orchestrator.adapter.outbound.llm.lite_llm.lite_llm_gateway import LiteLlmGateway
from agent_orchestrator.adapter.outbound.llm.mapper import NO_KEY, connector_api_key
from agent_orchestrator.adapter.outbound.llm.schema import ModelParameters
from agent_orchestrator.application.port.outbound.agent_port import AgentPort
from bootstrap.configuration.anthropic_sdk_settings import AnthropicSdkSettings
from bootstrap.di.agent_di import (
    LLM_CONNECTOR_NAME,
    MODEL_ALIASES,
    MONGODB_CONNECTOR_NAME,
    AgentDI,
    AgentRole,
)

STATES_COLLECTION: str = "agent_sdk_states"

_EFFORT: dict[ReasoningEffort, EffortLevel] = {
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
}
SESSIONS_COLLECTION: str = "agent_sdk_sessions"


class AnthropicSdkDI(AgentDI):
    _lite_llm_gateway: LiteLlmGateway | None = None
    _agent_mongo_client: AsyncMongoClient | None = None

    @cached_property
    def _anthropic_sdk_settings(self) -> AnthropicSdkSettings:
        return AnthropicSdkSettings.from_env()

    async def _anthropic_sdk_agent(self) -> AgentPort:
        settings = self._anthropic_sdk_settings
        base_url: str = await self._anthropic_base_url()
        states, sessions = await self._agent_sdk_stores()
        structured_output = StructuredOutput(
            AsyncAnthropic(base_url=base_url, api_key=settings.anthropic_auth_token), self._logging
        )
        settings.working_directory.mkdir(parents=True, exist_ok=True)
        sdk_settings = SdkSettings(
            base_url=base_url,
            auth_token=settings.anthropic_auth_token,
            working_directory=settings.working_directory,
            builtin_tools=settings.builtin_tools,
            mcp_servers=[
                _mcp_server(name, connector) for name, connector in self._mcp_connectors.items()
            ],
            otlp_endpoint=self._cli_otlp_endpoint(),
            environment=self._configuration.env.value,
        )
        # Discovered once at boot, like the LangGraph engine, so both describe the same tools.
        tools = ToolsStep(sdk_settings, (await self._tool_registry()).specifications())
        steps = AgentSteps(
            ingest=IngestStep(states),
            context=ContextStep(structured_output),
            clarify=ClarifyStep(),
            plan=PlanStep(structured_output, tools),
            act=ActStep(OptionsFactory(sdk_settings, sessions), tools),
            tools=tools,
            reflect=ReflectStep(structured_output),
            feedback=FeedbackStep(),
            finalize=FinalizeStep(),
            summarize=SummarizeStep(),
        )
        return AnthropicSdkAgent(
            models=self._model_profiles(),
            steps=steps,
            states=states,
            logger=self._logging,
            roles=self._model_roles(),
        )

    async def _stop_anthropic_sdk(self) -> None:
        if self._lite_llm_gateway is not None:
            await self._lite_llm_gateway.stop()
            self._lite_llm_gateway = None
        if self._agent_mongo_client is not None:
            await self._agent_mongo_client.close()
            self._agent_mongo_client = None

    async def _anthropic_base_url(self) -> str:
        settings = self._anthropic_sdk_settings
        if not settings.lite_llm_enabled:
            if not settings.anthropic_base_url:
                raise ValueError("ANTHROPIC_BASE_URL is required when LITELLM_ENABLED=false")
            return settings.anthropic_base_url

        upstream = self._configuration.connector.api(LLM_CONNECTOR_NAME)
        self._lite_llm_gateway = LiteLlmGateway(
            command=list(settings.lite_llm_command),
            config=lite_llm_config(
                self._gateway_models(),
                upstream.base_url,
                api_key=connector_api_key(upstream) or NO_KEY,
            ),
            logger=self._logging,
            host=settings.lite_llm_host,
            port=settings.lite_llm_port,
        )
        await self._lite_llm_gateway.start()
        return self._lite_llm_gateway.base_url

    async def _agent_sdk_stores(self) -> tuple[MongoAgentStateStore, MongoSessionStore]:
        connector = self._database_connector(MONGODB_CONNECTOR_NAME)
        self._agent_mongo_client = AsyncMongoClient(
            host=connector.host,
            port=int(connector.port),
            username=getattr(connector.auth, "username", None),
            password=getattr(connector.auth, "password", None),
            authSource="admin",
            serverSelectionTimeoutMS=connector.pool.get("timeout_ms", 5000),
        )
        await self._agent_mongo_client.admin.command("ping")
        database = self._agent_mongo_client[connector.default_name]
        ttl: int = connector.pool.get("ttl", 3600)

        states = MongoAgentStateStore(database[STATES_COLLECTION], ttl)
        # list/delete are optional SessionStore methods; ty mistakes them for abstract ones.
        sessions = MongoSessionStore(database[SESSIONS_COLLECTION], ttl)  # ty: ignore[call-non-callable]
        await states.ensure_indexes()
        await sessions.ensure_indexes()
        self._logging.info(f"Agent SDK sessions stored in MongoDB '{connector.default_name}'")
        return states, sessions

    def _cli_otlp_endpoint(self) -> str | None:
        # Same collector as the application's telemetry; empty host means telemetry is off.
        connector: TelemetryConnector = self._configuration.connector.telemetry("open_telemetry")
        if not (connector.host and connector.port):
            return None
        return f"http://{connector.host}:{connector.port}"

    def _model_profiles(self) -> dict[str, ModelProfile]:
        return {
            alias: _model_profile(self._model_settings(operation_name)[1])
            for alias, operation_name in MODEL_ALIASES.items()
        }

    def _model_roles(self) -> ModelRoles:
        # Summary is not a role here: the CLI compacts the conversation itself.
        profiles: dict[AgentRole, ModelProfile | None] = {}
        for role in (AgentRole.CONTEXT, AgentRole.PLAN, AgentRole.REFLECTION):
            settings = self._role_settings(role)
            profiles[role] = _model_profile(settings[1]) if settings else None
        return ModelRoles(
            context=profiles[AgentRole.CONTEXT],
            plan=profiles[AgentRole.PLAN],
            reflection=profiles[AgentRole.REFLECTION],
        )

    def _gateway_models(self) -> list[ModelParameters]:
        # Every selectable model, plus any model a role uses that is not selectable.
        models = {
            parameters.model_name: parameters
            for parameters in (self._model_settings(name)[1] for name in MODEL_ALIASES.values())
        }
        for role in (AgentRole.CONTEXT, AgentRole.PLAN, AgentRole.REFLECTION):
            settings = self._role_settings(role)
            if settings is not None:
                models.setdefault(settings[1].model_name, settings[1])
        return list(models.values())


def _model_profile(parameters: ModelParameters) -> ModelProfile:
    return ModelProfile(
        name=parameters.model_name,
        limits=AgentLimits(
            max_iterations=parameters.max_iterations,
            max_retries=parameters.max_reflection_retries,
        ),
        max_context_tokens=parameters.max_context_tokens,
        max_output_tokens=parameters.max_output_tokens,
        temperature=parameters.temperature,
        reasoning_effort=_EFFORT.get(parameters.reasoning_effort)
        if parameters.reasoning_effort
        else None,
    )


def _mcp_server(name: str, connector: McpConnector) -> McpServer:
    headers: dict[str, str] = {}
    auth = connector.auth
    if isinstance(auth, TokenAuth):
        headers[auth.key_name] = auth.key_value
    elif isinstance(auth, BasicAuth):
        token = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    transport = "sse" if connector.transport == "sse" else "http"
    return McpServer(name=name, url=connector.base_url, transport=transport, headers=headers)
