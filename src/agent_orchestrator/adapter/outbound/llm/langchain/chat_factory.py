from httpx import AsyncClient
from langchain_openai import ChatOpenAI

from agent_orchestrator.adapter.outbound.llm.schema import ModelConnector, ModelParameters
from agent_orchestrator.adapter.outbound.llm.thinking import thinking_switch


class LLMChat:
    def __init__(
        self,
        model_connector: ModelConnector,
        model_parameters: ModelParameters,
        async_client: AsyncClient | None = None,
    ) -> None:
        self._model_connector = model_connector
        self._model_parameters = model_parameters
        self._async_client = async_client

    def create_chat_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self._model_connector.base_url,
            api_key=self._model_connector.api_key,
            model=self._model_parameters.model_name,
            http_async_client=self._async_client,
            max_tokens=self._model_parameters.max_output_tokens,
            temperature=self._model_parameters.temperature,
            reasoning_effort=self._model_parameters.reasoning_effort,
            extra_body=thinking_switch(self._model_parameters.reasoning_effort),
            max_retries=0,
        )
