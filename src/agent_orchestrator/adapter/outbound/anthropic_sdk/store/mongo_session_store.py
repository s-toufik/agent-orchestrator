import json
import time
from datetime import UTC, datetime
from typing import Any

from claude_agent_sdk import SessionKey, SessionStore, SessionStoreEntry


class MongoSessionStore(SessionStore):
    def __init__(self, collection: Any, ttl_seconds: int) -> None:
        self._collection = collection
        self._ttl_seconds = ttl_seconds

    async def ensure_indexes(self) -> None:
        await self._collection.create_index([("key", 1), ("batch", 1), ("index", 1)])
        await self._collection.create_index("updated_at", expireAfterSeconds=self._ttl_seconds)

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        if not entries:
            return
        session_key, now, batch = _key(key), datetime.now(UTC), time.time_ns()
        await self._collection.insert_many(
            [
                # Entries are stored as JSON text: transcripts may hold "$" or "." field names.
                {
                    "key": session_key,
                    "batch": batch,
                    "index": i,
                    "entry": json.dumps(entry),
                    "updated_at": now,
                }
                for i, entry in enumerate(entries)
            ]
        )
        # A live session must not expire piece by piece.
        await self._collection.update_many({"key": session_key}, {"$set": {"updated_at": now}})

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        cursor = self._collection.find({"key": _key(key)}, {"_id": 0, "entry": 1}).sort(
            [("batch", 1), ("index", 1)]
        )
        entries: list[SessionStoreEntry] = [
            json.loads(document["entry"]) async for document in cursor
        ]
        return entries or None


def _key(key: SessionKey) -> str:
    parts = [key["project_key"], key["session_id"], key.get("subpath")]
    return "/".join(part for part in parts if part)
