"""Keeps the agent's hexagon honest now that it's a standalone service.

`agent` talks to its toolbox only over MCP, as an external, independently
deployed service -- it must never reach into a toolbox package directly,
and its domain layer must stay framework-free.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"


def _imported_roots(package: str) -> set[str]:
    roots: set[str] = set()
    for path in (SOURCE / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_agent_does_not_import_a_toolbox_package_directly() -> None:
    roots = _imported_roots("agent")
    assert not roots & {"toolbox", "agent_toolbox"}


def test_agent_does_not_import_bootstrap() -> None:
    assert "bootstrap" not in _imported_roots("agent")


def test_domain_layer_stays_free_of_frameworks() -> None:
    forbidden = {"fastapi", "starlette", "mcp", "langchain", "langgraph", "pydantic"}
    for path in (SOURCE / "agent" / "domain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name.split(".", 1)[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".", 1)[0]
            assert module not in forbidden, f"{path} imports {module}"
