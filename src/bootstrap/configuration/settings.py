import os
from dataclasses import dataclass
from pathlib import Path

import dotenv


@dataclass(frozen=True, slots=True)
class ProcessSettings:
    role: str
    environment: str
    configuration_directory: Path
    host: str
    port: int

    @classmethod
    def for_role(cls, role: str, default_port: int) -> ProcessSettings:
        dotenv.load_dotenv()
        prefix = role.upper()
        directory = os.getenv("CONFIGURATION_DIR", "./config")

        return cls(
            role=role,
            environment=os.getenv("APP_ENV", "debug"),
            configuration_directory=Path(directory),
            host=os.getenv(f"{prefix}_HOST", "0.0.0.0"),
            port=int(os.getenv(f"{prefix}_PORT", str(default_port))),
        )
