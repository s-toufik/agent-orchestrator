from pydantic import BaseModel


class AgentLimits(BaseModel):
    max_iterations: int = 20
    max_retries: int = 2
