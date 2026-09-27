from typing import Any

from agent_orchestrator.adapter.outbound.llm.schema import ModelParameters
from agent_orchestrator.adapter.outbound.llm.thinking import thinking_switch


def lite_llm_config(
    models: list[ModelParameters], upstream_base_url: str, api_key: str
) -> dict[str, Any]:
    return {
        "model_list": [
            {
                "model_name": model.model_name,
                "litellm_params": {
                    "model": f"hosted_vllm/{model.model_name}",
                    "api_base": upstream_base_url,
                    "api_key": api_key,
                    "temperature": model.temperature,
                    # The CLI's own effort/thinking options do not reach llama.cpp's switch.
                    "extra_body": thinking_switch(model.reasoning_effort),
                },
            }
            for model in models
        ],
        "litellm_settings": {"drop_params": True},
    }
