from pycraftcore.application_configuration.enum import ConnectorType, OperationType
from pycraftcore.application_configuration.model.connector import ApiConnector
from pycraftcore.application_configuration.model.operation import ApiOperation
from pycraftcore.authentication import NoAuth
from pycraftcore.authentication.model.auth_type import AuthType
from pycraftcore.authentication.model.token_auth import TokenAuth
from pycraftcore.http.enum import HttpMethod
from pydantic import SecretStr

from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort
from agent_orchestrator.adapter.outbound.llm.mapper import ModelSettingsMapper


def make_operation(parameters: dict, auth: NoAuth | TokenAuth | None = None) -> ApiOperation:
    connector = ApiConnector(
        name="conn",
        type=ConnectorType.api,
        auth=auth or NoAuth(type=AuthType.none),
        base_url="http://example.com",
        timeout=10,
        retry=1,
    )
    return ApiOperation(
        name="gpt-model",
        type=OperationType.api,
        connector=connector,
        endpoint="/chat",
        method=HttpMethod.POST,
        parameters=parameters,
    )


def test_maps_the_connector_base_url_and_a_placeholder_api_key() -> None:
    connector, _ = ModelSettingsMapper(make_operation({}))()

    assert connector.base_url == "http://example.com"
    assert isinstance(connector.api_key, SecretStr)
    assert connector.api_key.get_secret_value() == "No_Key"


def test_a_token_connector_passes_its_key_to_the_model_client() -> None:
    auth = TokenAuth(type=AuthType.token, key_name="apikey", key_value="sk-test")
    connector, _ = ModelSettingsMapper(make_operation({}, auth))()

    assert connector.api_key.get_secret_value() == "sk-test"


def test_defaults_every_parameter_when_absent() -> None:
    _, parameters = ModelSettingsMapper(make_operation({}))()

    assert parameters.model_name == "gpt-model"
    assert parameters.max_output_tokens == 8000
    assert parameters.max_context_tokens == 8000
    assert parameters.temperature == 0.0
    assert parameters.max_iterations == 10
    assert parameters.reasoning_effort is None


def test_uses_explicit_parameters_when_present() -> None:
    _, parameters = ModelSettingsMapper(
        make_operation(
            {
                "max_output_tokens": 500,
                "max_context_tokens": 2_000,
                "temperature": 0.7,
                "max_iterations": 3,
                "reasoning_effort": "medium",
            }
        )
    )()

    assert parameters.max_output_tokens == 500
    assert parameters.max_context_tokens == 2_000
    assert parameters.temperature == 0.7
    assert parameters.max_iterations == 3
    assert parameters.reasoning_effort is ReasoningEffort.MEDIUM


def test_a_null_reasoning_effort_means_the_model_does_not_reason() -> None:
    _, parameters = ModelSettingsMapper(make_operation({"reasoning_effort": None}))()

    assert parameters.reasoning_effort is None
