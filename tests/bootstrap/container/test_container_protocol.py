from pathlib import Path

from bootstrap.configuration.settings import ProcessSettings
from bootstrap.container.agent_container import AgentContainer
from bootstrap.container.container import Container

REAL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def test_agent_container_satisfies_the_container_protocol() -> None:
    settings = ProcessSettings(
        role="agent-orchestrator",
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
        host="0.0.0.0",
        port=8000,
    )

    assert isinstance(AgentContainer(settings), Container)
