from fastapi import APIRouter
from starlette import status
from starlette.requests import Request
from starlette.responses import Response

from bootstrap.router.actuator.enum.status import HealthStatus
from bootstrap.router.actuator.schema.response import HealthSchema, InfoSchema


class ActuatorRouter:
    PREFIX = "/actuator"

    def __init__(
        self,
        app_name: str,
        app_version: str,
        app_deployment_environment: str,
        app_api_root_path: str,
        app_authors: str,
    ) -> None:
        self._app_name = app_name
        self._app_version = app_version
        self._app_deployment_environment = app_deployment_environment
        self._app_authors = app_authors
        self._app_root_path = app_api_root_path

        self._router = APIRouter(prefix=self.PREFIX, tags=["actuator"])
        self._router_register()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _router_register(self) -> None:
        self._router.add_api_route(
            "/health", self._health, methods=["GET"], response_model=HealthSchema
        )
        self._router.add_api_route(
            "/health/liveness", self._liveness, methods=["GET"], response_model=HealthSchema
        )
        self._router.add_api_route(
            "/health/readiness", self._readiness, methods=["GET"], response_model=HealthSchema
        )
        self._router.add_api_route("/info", self._info, methods=["GET"], response_model=InfoSchema)

    @staticmethod
    def _health() -> HealthSchema:
        return HealthSchema(status=HealthStatus.UP)

    @staticmethod
    def _liveness() -> HealthSchema:
        return HealthSchema(status=HealthStatus.UP)

    @staticmethod
    def _readiness(request: Request, response: Response) -> HealthSchema:
        if bool(getattr(request.app.state, "ready", False)):
            return HealthSchema(status=HealthStatus.UP)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthSchema(status=HealthStatus.DOWN)

    def _info(self) -> InfoSchema:
        return InfoSchema(
            name=self._app_name,
            version=self._app_version,
            environment=self._app_deployment_environment,
            api_root_path=self._app_root_path,
            authors=self._app_authors,
        )
