from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


async def invoke_structured[T: BaseModel](
    llm: BaseChatModel, schema: type[T], messages: Sequence[BaseMessage]
) -> T | None:
    raw: Any = await llm.with_structured_output(schema, include_raw=True).ainvoke(list(messages))
    parsed: Any = raw.get("parsed") if isinstance(raw, dict) else None
    return parsed if isinstance(parsed, schema) else None
