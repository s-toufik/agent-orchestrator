from pathlib import Path

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.base_di import BaseDI

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def make_settings(role: str = "agent-orchestrator") -> ProcessSettings:
    return ProcessSettings(
        role=role,
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
    )


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_HOST", "/tmp")
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_NAME", "test-checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_HOST", "localhost")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PORT", "27017")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_NAME", "test-checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_USERNAME", "test")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PASSWORD", "test")
    monkeypatch.setenv("LLM_API_KEY", "test")


class RecordingClient:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def create_client(self):
        raise AssertionError("not expected to be called")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class RecordingRepository:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False

    async def connection(self):
        raise AssertionError("not expected to be called")

    async def connect(self) -> RecordingRepository:
        self.connected = True
        return self

    async def disconnect(self) -> None:
        self.disconnected = True

    async def execute(self, sql: str, parameters: tuple = ()) -> list[dict]:
        raise AssertionError("not expected to be called")


class NoLifecycleClient: ...


def test_register_client_returns_the_same_object_and_tracks_it() -> None:
    di = BaseDI(make_settings())
    client = RecordingClient()

    assert di._register_client(client) is client
    assert di._clients == [client]


def test_register_repository_returns_the_same_object_and_tracks_it() -> None:
    di = BaseDI(make_settings())
    repository = RecordingRepository()

    assert di._register_repository(repository) is repository
    assert di._repositories == [repository]


async def test_start_factories_starts_clients_and_connects_repositories() -> None:
    di = BaseDI(make_settings())
    client = RecordingClient()
    repository = RecordingRepository()
    di._register_client(client)
    di._register_repository(repository)
    di._register_client(NoLifecycleClient())  # ty: ignore[invalid-argument-type]  # must not raise

    await di._start_factories()

    assert client.started
    assert repository.connected


async def test_stop_factories_closes_clients_and_disconnects_repositories() -> None:
    di = BaseDI(make_settings())
    client = RecordingClient()
    repository = RecordingRepository()
    di._register_client(client)
    di._register_repository(repository)

    await di._stop_factories()

    assert client.closed
    assert repository.disconnected


def test_configuration_loads_the_real_tree(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    di = BaseDI(make_settings())

    assert di._configuration.connector.mcp("toolbox").transport == "streamable_http"


def test_telemetry_provider_service_name_is_derived_from_the_role(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    di = BaseDI(make_settings(role="worker"))

    provider = di._telemetry_provider

    assert provider is not None
    provider.shutdown()


async def test_shutdown_telemetry_is_a_no_op_when_never_constructed() -> None:
    di = BaseDI(make_settings())

    await di._shutdown_telemetry()


async def test_shutdown_telemetry_shuts_down_and_evicts_the_cached_provider(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    di = BaseDI(make_settings())
    _ = di._telemetry_provider
    assert "_telemetry_provider" in di.__dict__

    await di._shutdown_telemetry()

    assert "_telemetry_provider" not in di.__dict__
