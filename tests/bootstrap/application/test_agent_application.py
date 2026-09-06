from pathlib import Path

import pytest
import uvicorn as uvicorn_module
from starlette.testclient import TestClient

from bootstrap.application.agent_application import create_agent_application, main
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.container.agent_container import AgentContainer

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture(autouse=True)
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_HOST", str(tmp_path))
    monkeypatch.setenv("USER_DB_NAME", "users")
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")


def make_settings() -> ProcessSettings:
    return ProcessSettings(
        role="agent",
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
        host="0.0.0.0",
        port=8000,
    )


def test_full_lifespan_includes_the_booted_routers_and_reports_healthy(monkeypatch) -> None:
    # The real end-to-end boot (real toolbox, real sqlite, real graph
    # building) is already proven directly against AgentContainer in
    # tests/bootstrap/container/test_agent_container.py. This test is only
    # for _lifespan's own wiring -- does it call boot(), include whatever
    # routers boot() produced, and report healthy -- so boot() is stubbed
    # rather than driving a real network call through TestClient's own
    # blocking portal thread.
    from fastapi import APIRouter

    fake_router = APIRouter(prefix="/fake")

    @fake_router.get("/ping")
    async def ping() -> dict:
        return {"pong": True}

    async def fake_boot(self) -> None:
        self._routers.append(fake_router)
        self._routers.append(self._actuator_router())

    async def fake_stop(self) -> None:
        return None

    monkeypatch.setattr(AgentContainer, "boot", fake_boot)
    monkeypatch.setattr(AgentContainer, "stop", fake_stop)
    app = create_agent_application(settings=make_settings())

    with TestClient(app) as client:
        health = client.get("/actuator/health")
        fake = client.get("/fake/ping")

    assert health.status_code == 200
    assert health.json() == {"status": "UP"}
    assert fake.status_code == 200
    assert fake.json() == {"pong": True}


def test_a_boot_failure_is_logged_and_reraised_not_swallowed(monkeypatch) -> None:
    async def failing_boot(self) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentContainer, "boot", failing_boot)
    app = create_agent_application(settings=make_settings())

    with pytest.raises(RuntimeError, match="boom"):
        with TestClient(app):
            pass


def test_main_runs_uvicorn_with_the_configured_host_and_port(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_HOST", "127.0.0.1")
    monkeypatch.setenv("AGENT_PORT", "9999")
    calls: list[dict] = []

    def fake_run(target, **kwargs):
        calls.append({"target": target, **kwargs})

    monkeypatch.setattr(uvicorn_module, "run", fake_run)

    main()

    assert calls == [
        {
            "target": "bootstrap.application.agent_application:app",
            "host": "127.0.0.1",
            "port": 9999,
        }
    ]
