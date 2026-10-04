import os
from dataclasses import dataclass
from pathlib import Path

import dotenv

from agent_orchestrator.adapter.outbound.llm.enum.answer_delivery import AnswerDelivery


@dataclass(frozen=True, slots=True)
class ProcessSettings:
    role: str
    environment: str
    configuration_directory: Path
    log_level: str = "INFO"
    max_concurrent_streams: int = 200
    answer_delivery: AnswerDelivery = AnswerDelivery.STREAM

    @classmethod
    def for_role(cls, role: str) -> ProcessSettings:
        dotenv.load_dotenv()
        directory = os.getenv("CONFIGURATION_DIR", "./config")

        return cls(
            role=role,
            environment=os.getenv("APP_ENV", "debug"),
            configuration_directory=Path(directory),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_concurrent_streams=int(os.getenv("MAX_CONCURRENT_STREAMS", "200")),
            answer_delivery=AnswerDelivery(os.getenv("AGENT_ANSWER_DELIVERY", "stream")),
        )
