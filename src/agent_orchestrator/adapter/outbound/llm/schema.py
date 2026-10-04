from pydantic import BaseModel, SecretStr

from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort


class ModelConnector(BaseModel):
    base_url: str
    api_key: SecretStr | None


class ModelParameters(BaseModel):
    model_name: str
    temperature: float
    max_output_tokens: int
    max_context_tokens: int
    max_iterations: int
    use_streaming: bool
    max_reflection_retries: int = 2
    reasoning_effort: ReasoningEffort | None = None
