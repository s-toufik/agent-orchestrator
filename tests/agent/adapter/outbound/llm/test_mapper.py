from pycraftcore.application_configuration.enum import ConnectorType, OperationType
from pycraftcore.application_configuration.model.connector import ApiConnector
from pycraftcore.application_configuration.model.operation import ApiOperation
from pycraftcore.authentication import NoAuth
from pycraftcore.authentication.model.auth_type import AuthType
from pycraftcore.http.enum import HttpMethod
from pydantic import SecretStr

from agent.adapter.outbound.llm.mapper import ModelSettingsMapper


def make_operation(parameters: dict) -> ApiOperation:
    connector = ApiConnector(
        name="conn",
        type=ConnectorType.api,
        auth=NoAuth(type=AuthType.none),
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


def test_defaults_every_parameter_when_absent() -> None:
    _, parameters = ModelSettingsMapper(make_operation({}))()

    assert parameters.model_name == "gpt-model"
    assert parameters.max_output_tokens == 8000
    assert parameters.max_context_tokens == 8000
    assert parameters.temperature == 0.0
    assert parameters.max_iterations == 10
    assert parameters.use_streaming is False


def test_uses_explicit_parameters_when_present() -> None:
    _, parameters = ModelSettingsMapper(
        make_operation(
            {
                "max_output_tokens": 500,
                "max_context_tokens": 2_000,
                "temperature": 0.7,
                "max_iterations": 3,
                "use_streaming": True,
            }
        )
    )()

    assert parameters.max_output_tokens == 500
    assert parameters.max_context_tokens == 2_000
    assert parameters.temperature == 0.7
    assert parameters.max_iterations == 3
    assert parameters.use_streaming is True
