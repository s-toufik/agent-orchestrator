import os
import shlex
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LITE_LLM_COMMAND: str = "uvx --from litellm[proxy]==1.102.1 litellm"
DEFAULT_WORKING_DIRECTORY: str = "working_directory"
SUPPORTED_BUILTIN_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep")


@dataclass(frozen=True, slots=True)
class AnthropicSdkSettings:
    working_directory: Path
    builtin_tools: tuple[str, ...]
    lite_llm_enabled: bool
    lite_llm_command: tuple[str, ...]
    lite_llm_host: str
    lite_llm_port: int
    anthropic_base_url: str
    anthropic_auth_token: str

    @classmethod
    def from_env(cls) -> AnthropicSdkSettings:
        return cls(
            working_directory=_working_directory(os.getenv("WORKING_DIRECTORY", "")),
            builtin_tools=_builtin_tools(os.getenv("AGENT_SDK_TOOLS", "")),
            lite_llm_enabled=os.getenv("LITELLM_ENABLED", "true").lower() == "true",
            lite_llm_command=tuple(
                shlex.split(os.getenv("LITELLM_COMMAND", DEFAULT_LITE_LLM_COMMAND))
            ),
            lite_llm_host=os.getenv("LITELLM_HOST", "127.0.0.1"),
            lite_llm_port=int(os.getenv("LITELLM_PORT", "4000")),
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            anthropic_auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN", "none"),
        )


def _working_directory(value: str) -> Path:
    # Unset: a folder under the current directory, never the current directory itself
    # (that would put the application's own files in reach of the file tools).
    return Path(value or Path.cwd() / DEFAULT_WORKING_DIRECTORY).resolve()


def _builtin_tools(value: str) -> tuple[str, ...]:
    tools: tuple[str, ...] = tuple(dict.fromkeys(t.strip() for t in value.split(",") if t.strip()))
    unknown: list[str] = [tool for tool in tools if tool not in SUPPORTED_BUILTIN_TOOLS]
    if unknown:
        raise ValueError(
            f"AGENT_SDK_TOOLS: unsupported built-in tool(s) {', '.join(unknown)}; "
            f"supported: {', '.join(SUPPORTED_BUILTIN_TOOLS)}"
        )
    if "Edit" in tools and "Read" not in tools:
        raise ValueError("AGENT_SDK_TOOLS: Edit needs Read (the CLI only edits files it has read)")
    return tools
