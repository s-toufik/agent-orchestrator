from pydantic import BaseModel


class Plan(BaseModel):
    task: str
    steps: str
