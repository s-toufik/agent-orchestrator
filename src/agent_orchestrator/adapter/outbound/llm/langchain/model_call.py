from collections.abc import Sequence
from typing import Any

from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)


async def invoke(model: Runnable, messages: Sequence[BaseMessage]) -> Any:
    try:
        return await model.ainvoke(list(messages))
    except ModelError as exception:
        if exception.is_retryable:
            raise AgentUnavailableException(str(exception)) from exception
        raise


async def invoke_structured[T: BaseModel](
    model: BaseChatModel, schema: type[T], messages: Sequence[BaseMessage]
) -> T | None:
    raw: Any = await invoke(model.with_structured_output(schema, include_raw=True), messages)
    parsed: Any = raw.get("parsed") if isinstance(raw, dict) else None
    return parsed if isinstance(parsed, schema) else None
