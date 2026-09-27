import socket
import sys
from pathlib import Path

import httpx2
import pytest

from agent_orchestrator.adapter.outbound.llm.enum.reasoning_effort import ReasoningEffort
from agent_orchestrator.adapter.outbound.llm.lite_llm.lite_llm_config import lite_llm_config
from agent_orchestrator.adapter.outbound.llm.lite_llm.lite_llm_gateway import LiteLlmGateway
from agent_orchestrator.adapter.outbound.llm.schema import ModelParameters

FAKE = str(Path(__file__).with_name("fake_litellm.py"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _parameters(name: str, temperature: float) -> ModelParameters:
    return ModelParameters(
        model_name=name,
        temperature=temperature,
        max_output_tokens=100,
        max_context_tokens=1_000,
        max_iterations=1,
        use_streaming=False,
    )


def test_every_model_is_routed_to_the_openai_compatible_server_with_its_temperature() -> None:
    config = lite_llm_config(
        [_parameters("qwen3-8b", 0.0), _parameters("qwen3-14b", 0.7)],
        "http://llama:8090/v1",
        api_key="k",
    )

    assert config["model_list"][0] == {
        "model_name": "qwen3-8b",
        "litellm_params": {
            "model": "hosted_vllm/qwen3-8b",
            "api_base": "http://llama:8090/v1",
            "api_key": "k",
            "temperature": 0.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    }
    assert config["model_list"][1]["litellm_params"]["temperature"] == 0.7
    assert [model["model_name"] for model in config["model_list"]] == ["qwen3-8b", "qwen3-14b"]


def test_a_model_with_a_reasoning_effort_is_served_with_thinking_on() -> None:
    thinking = _parameters("qwen3-14b", 0.0).model_copy(
        update={"reasoning_effort": ReasoningEffort.MEDIUM}
    )

    config = lite_llm_config([thinking], "http://llama:8090/v1", api_key="k")

    extra_body = config["model_list"][0]["litellm_params"]["extra_body"]
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


async def test_the_gateway_starts_serves_and_stops(logger) -> None:
    port = _free_port()
    gateway = LiteLlmGateway([sys.executable, FAKE], {"model_list": []}, logger, port=port)

    await gateway.start()
    async with httpx2.AsyncClient() as client:
        response = await client.get(f"{gateway.base_url}/health/liveliness")
    await gateway.stop()

    assert response.status_code == 200
    assert gateway.base_url == f"http://127.0.0.1:{port}"
    with pytest.raises(httpx2.TransportError):
        async with httpx2.AsyncClient() as client:
            await client.get(f"{gateway.base_url}/health/liveliness")


async def test_a_gateway_that_dies_on_startup_reports_its_output(logger) -> None:
    gateway = LiteLlmGateway(
        [sys.executable, FAKE, "--fail"], {}, logger, port=_free_port(), startup_timeout=10
    )

    with pytest.raises(RuntimeError, match="bad config"):
        await gateway.start()


async def test_stopping_a_gateway_that_never_started_is_safe(logger) -> None:
    await LiteLlmGateway(["litellm"], {}, logger).stop()
