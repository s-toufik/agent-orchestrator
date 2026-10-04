from httpx import AsyncClient
from langchain_core.language_models import BaseChatModel

from agent_orchestrator.adapter.outbound.llm.langchain.chat_factory import LLMChat
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole, ModelCatalog


class ChatModels:
    def __init__(self, catalog: ModelCatalog, http_client: AsyncClient | None = None) -> None:
        self._catalog = catalog
        self._http_client = http_client
        self._clients: dict[tuple[str, str], BaseChatModel] = {}

    def for_role(self, role: AgentRole, model: str) -> BaseChatModel:
        connector, parameters = self._catalog.settings_for(role, model)
        key = (connector.base_url, parameters.model_dump_json())
        if key not in self._clients:
            unstreamed = parameters.model_copy(update={"use_streaming": False})
            self._clients[key] = LLMChat(
                connector, unstreamed, self._http_client
            ).create_chat_client()
        return self._clients[key]
