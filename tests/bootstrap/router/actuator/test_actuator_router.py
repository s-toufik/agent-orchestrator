from fastapi import FastAPI
from starlette.testclient import TestClient

from bootstrap.router.actuator.actuator_router import ActuatorRouter


def make_app(ready: bool | None = None) -> FastAPI:
    app = FastAPI()
    router = ActuatorRouter(
        app_name="agent",
        app_version="1.2.3",
        app_deployment_environment="debug",
        app_api_root_path="/",
        app_authors="dev@example.com",
    )
    app.include_router(router.router)
    if ready is not None:
        app.state.ready = ready
    return app


def test_health_is_always_up() -> None:
    client = TestClient(make_app())

    response = client.get("/actuator/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_liveness_is_always_up() -> None:
    client = TestClient(make_app())

    response = client.get("/actuator/health/liveness")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_readiness_is_down_before_the_app_marks_itself_ready() -> None:
    client = TestClient(make_app())

    response = client.get("/actuator/health/readiness")

    assert response.status_code == 503
    assert response.json() == {"status": "DOWN"}


def test_readiness_is_up_once_the_app_marks_itself_ready() -> None:
    client = TestClient(make_app(ready=True))

    response = client.get("/actuator/health/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_info_reports_the_constructor_fields() -> None:
    client = TestClient(make_app())

    response = client.get("/actuator/info")

    assert response.json() == {
        "name": "agent",
        "version": "1.2.3",
        "environment": "debug",
        "api_root_path": "/",
        "authors": "dev@example.com",
    }
