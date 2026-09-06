from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentRequest:
    message: str
    model_name: str
    request_id: str
