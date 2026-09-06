import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pycraftcore.http.middleware import RequestMiddleware
from pycraftcore.http.middleware.request_id_middleware import RequestIDMiddleware
from starlette.middleware.cors import CORSMiddleware

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.container.agent_container import AgentContainer

DEFAULT_PORT: int = 8000


@asynccontextmanager
async def _lifespan(application: FastAPI, container: AgentContainer) -> AsyncGenerator[None]:
    try:
        await container.boot()
    except Exception as exception:
        container.logging.critical(
            "Agent container failed to boot:\n" + "".join(traceback.format_exception(exception))
        )
        raise

    for router in container.routers:
        application.include_router(router)
        container.logging.info(f"Router {router.prefix} included")

    try:
        yield
    finally:
        await container.stop()


def create_agent_application(settings: ProcessSettings | None = None) -> FastAPI:
    process_settings = settings or ProcessSettings.for_role("agent", DEFAULT_PORT)
    container = AgentContainer(process_settings)

    application = FastAPI(
        title="agent",
        lifespan=lambda app: _lifespan(app, container),
    )

    application.add_middleware(RequestIDMiddleware)
    application.add_middleware(RequestMiddleware)
    # noinspection PyTypeChecker
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


app: FastAPI = create_agent_application()


def main() -> None:
    import uvicorn

    settings = ProcessSettings.for_role("agent", DEFAULT_PORT)
    uvicorn.run(
        "bootstrap.application.agent_application:app",
        host=settings.host,
        port=settings.port,
    )
