from claude_agent_sdk import SessionKey

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.mongo_agent_state_store import (
    MongoAgentStateStore,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.mongo_session_store import (
    MongoSessionStore,
)
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.store.fake_collection import (
    FakeCollection,
)

KEY: SessionKey = {"project_key": "-tmp-agent", "session_id": "s-1"}


async def test_session_entries_come_back_in_append_order() -> None:
    store = MongoSessionStore(FakeCollection(), ttl_seconds=60)

    await store.append(KEY, [{"type": "user", "uuid": "1"}, {"type": "assistant", "uuid": "2"}])
    await store.append(KEY, [{"type": "user", "uuid": "3", "input": {"$schema": "x"}}])

    entries = await store.load(KEY)

    assert [entry["uuid"] for entry in entries or []] == ["1", "2", "3"]
    assert (entries or [])[2]["input"] == {"$schema": "x"}


async def test_subagent_transcripts_are_kept_apart() -> None:
    store = MongoSessionStore(FakeCollection(), ttl_seconds=60)

    await store.append({**KEY, "subpath": "agent-a"}, [{"type": "user", "uuid": "sub"}])

    assert await store.load(KEY) is None
    assert await store.load({**KEY, "subpath": "agent-a"}) == [{"type": "user", "uuid": "sub"}]


async def test_a_live_session_is_refreshed_as_a_whole() -> None:
    collection = FakeCollection()
    store = MongoSessionStore(collection, ttl_seconds=60)

    await store.append(KEY, [{"type": "user", "uuid": "1"}])
    first_write = collection.documents[0]["updated_at"]
    await store.append(KEY, [{"type": "user", "uuid": "2"}])

    assert collection.documents[0]["updated_at"] >= first_write
    assert len({document["updated_at"] for document in collection.documents}) == 1


async def test_session_indexes_expire_idle_sessions() -> None:
    collection = FakeCollection()

    await MongoSessionStore(collection, ttl_seconds=90).ensure_indexes()

    assert ("updated_at", {"expireAfterSeconds": 90}) in collection.indexes


async def test_agent_states_round_trip() -> None:
    collection = FakeCollection()
    store = MongoAgentStateStore(collection, ttl_seconds=60)
    conversation = AgentState(
        session_id="conv-1", sdk_session_id="s-1", pending_plan=Plan(task="t", steps="1. step")
    )
    conversation.record("hi", "hello")

    await store.save(conversation)
    await store.save(conversation)

    assert await store.load("conv-1") == conversation
    assert await store.load("unknown") is None
    assert len(collection.documents) == 1
    assert "updated_at" in collection.documents[0]
