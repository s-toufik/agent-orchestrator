from pydantic import BaseModel

from bootstrap.router.actuator.enum.status import HealthStatus


class HealthSchema(BaseModel):
    status: HealthStatus


class InfoSchema(BaseModel):
    name: str
    version: str
    environment: str
    api_root_path: str
    authors: str
