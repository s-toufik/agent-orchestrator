import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SOURCE / "agent_orchestrator"

FRAMEWORKS = {
    "fastapi",
    "starlette",
    "mcp",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "pymongo",
    "pydantic",
    "httpx",
    "httpx2",
}


def _imports(directory: Path) -> dict[Path, set[str]]:
    found: dict[Path, set[str]] = {}
    for path in directory.rglob("*.py"):
        modules: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
        found[path] = modules
    return found


def _violations(directory: Path, forbidden: set[str]) -> list[str]:
    assert directory.is_dir(), f"{directory} does not exist"
    return [
        f"{path.relative_to(SOURCE)} imports {module}"
        for path, modules in _imports(directory).items()
        for module in modules
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]


def test_the_domain_is_free_of_frameworks_and_outer_layers() -> None:
    forbidden = FRAMEWORKS | {
        "agent_orchestrator.application",
        "agent_orchestrator.adapter",
        "bootstrap",
    }
    assert _violations(PACKAGE / "domain", forbidden) == []


def test_the_application_depends_only_on_the_domain() -> None:
    forbidden = FRAMEWORKS | {"agent_orchestrator.adapter", "bootstrap"}
    assert _violations(PACKAGE / "application", forbidden) == []


def test_the_hexagon_never_imports_the_composition_root() -> None:
    assert _violations(PACKAGE, {"bootstrap"}) == []


def test_outbound_adapters_never_import_inbound_adapters() -> None:
    forbidden = {"agent_orchestrator.adapter.inbound"}
    assert _violations(PACKAGE / "adapter" / "outbound", forbidden) == []


def test_the_agent_never_imports_a_toolbox_package_directly() -> None:
    assert _violations(PACKAGE, {"toolbox", "agent_toolbox"}) == []
