from typing import cast

from pycraftcore.application_configuration.model.connector import ApiConnector
from pycraftcore.application_configuration.model.operation import ApiOperation
from pydantic import SecretStr

from agent.adapter.outbound.llm.schema import ModelConnector, ModelParameters


class ModelSettingsMapper:
    def __init__(self, operation: ApiOperation) -> None:
        self._operation = operation

    def __call__(self) -> tuple[ModelConnector, ModelParameters]:
        connector: ApiConnector = self._operation.connector
        parameters = self._operation.parameters

        return (
            ModelConnector(
                base_url=connector.base_url,
                api_key=SecretStr("No_Key"),
            ),
            ModelParameters(
                model_name=self._operation.name,
                max_output_tokens=cast(int, parameters.get("max_output_tokens", 8000)),
                max_context_tokens=cast(int, parameters.get("max_context_tokens", 8000)),
                temperature=cast(float, parameters.get("temperature", 0.0)),
                max_iterations=cast(int, parameters.get("max_iterations", 10)),
                use_streaming=cast(bool, parameters.get("use_streaming", False)),
            ),
        )
