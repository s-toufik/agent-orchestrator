import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import dotenv


class AgentEngine(StrEnum):
    LANGGRAPH = "langgraph"
    ANTHROPIC_SDK = "anthropic_sdk"


@dataclass(frozen=True, slots=True)
class ProcessSettings:
    role: str
    environment: str
    configuration_directory: Path
    engine: AgentEngine = AgentEngine.LANGGRAPH
    max_concurrent_streams: int = 200

    @classmethod
    def for_role(cls, role: str) -> ProcessSettings:
        dotenv.load_dotenv()
        directory = os.getenv("CONFIGURATION_DIR", "./config")

        return cls(
            role=role,
            environment=os.getenv("APP_ENV", "debug"),
            configuration_directory=Path(directory),
            engine=AgentEngine(os.getenv("AGENT_ENGINE", AgentEngine.LANGGRAPH)),
            max_concurrent_streams=int(os.getenv("MAX_CONCURRENT_STREAMS", "200")),
        )
