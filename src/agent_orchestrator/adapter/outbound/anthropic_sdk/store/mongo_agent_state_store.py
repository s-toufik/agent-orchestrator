from datetime import UTC, datetime
from typing import Any

from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState


class MongoAgentStateStore:
    def __init__(self, collection: Any, ttl_seconds: int) -> None:
        self._collection = collection
        self._ttl_seconds = ttl_seconds

    async def ensure_indexes(self) -> None:
        await self._collection.create_index("updated_at", expireAfterSeconds=self._ttl_seconds)

    async def load(self, session_id: str) -> AgentState | None:
        document: dict[str, Any] | None = await self._collection.find_one({"_id": session_id})
        if document is None:
            return None
        return AgentState.model_validate({**document, "session_id": document["_id"]})

    async def save(self, agent_state: AgentState) -> None:
        document = agent_state.model_dump(mode="json", exclude={"session_id"})
        await self._collection.replace_one(
            {"_id": agent_state.session_id},
            {**document, "updated_at": datetime.now(UTC)},
            upsert=True,
        )
