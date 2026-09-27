from typing import cast

from pycraftcore.application_configuration.model.connector import ApiConnector
from pycraftcore.application_configuration.model.operation import ApiOperation
from pycraftcore.authentication.model.token_auth import TokenAuth
from pydantic import SecretStr

from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort
from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters

# A local server needs no key, but OpenAI-compatible clients refuse an empty one.
NO_KEY: str = "No_Key"


class ModelSettingsMapper:
    def __init__(self, operation: ApiOperation) -> None:
        self._operation = operation

    def __call__(self) -> tuple[ModelConnector, ModelParameters]:
        connector: ApiConnector = self._operation.connector
        parameters = self._operation.parameters

        return (
            ModelConnector(
                base_url=connector.base_url,
                api_key=SecretStr(connector_api_key(connector) or NO_KEY),
            ),
            ModelParameters(
                model_name=str(parameters.get("model") or self._operation.name),
                max_output_tokens=cast(int, parameters.get("max_output_tokens", 8000)),
                max_context_tokens=cast(int, parameters.get("max_context_tokens", 8000)),
                temperature=cast(float, parameters.get("temperature", 0.0)),
                max_iterations=cast(int, parameters.get("max_iterations", 10)),
                use_streaming=cast(bool, parameters.get("use_streaming", False)),
                max_reflection_retries=cast(int, parameters.get("max_reflection_retries", 2)),
                reasoning_effort=_reasoning_effort(parameters.get("reasoning_effort")),
            ),
        )


def _reasoning_effort(value: object) -> ReasoningEffort | None:
    # null in llm.yml: the model does not reason, so nothing is sent.
    return ReasoningEffort(str(value)) if value else None


def connector_api_key(connector: ApiConnector) -> str | None:
    # A hosted provider (OpenRouter...) authenticates with the connector's token as a Bearer key.
    auth = connector.auth
    return auth.key_value if isinstance(auth, TokenAuth) and auth.key_value else None
