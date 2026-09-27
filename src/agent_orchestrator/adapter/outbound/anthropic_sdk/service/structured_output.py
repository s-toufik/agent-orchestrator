import re

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pycraftcore.logger.port import Logger
from pydantic import BaseModel, ValidationError

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.llm.thinking import thinking_switch

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class StructuredOutput:
    def __init__(self, client: AsyncAnthropic, logger: Logger, max_tokens: int = 1_024) -> None:
        self._client = client
        self._logger = logger
        self._max_tokens = max_tokens

    async def invoke(
        self,
        profile: ModelProfile,
        system: str,
        messages: list[MessageParam],
        max_tokens: int | None = None,
    ) -> str:
        # Streamed: the client refuses a non-streaming call whose max_tokens could take over
        # 10 minutes (~21k tokens), and a slow local model may need that long.
        async with self._client.messages.stream(
            model=profile.name,
            max_tokens=max_tokens or self._max_tokens,
            system=system,
            messages=messages,
            # Not part of the Anthropic API: forwarded as-is to the OpenAI-compatible model.
            extra_body=_sampling(profile),
        ) as stream:
            response = await stream.get_final_message()
        text: str = "".join(block.text for block in response.content if block.type == "text")
        return _THINKING.sub("", text).strip()

    async def invoke_structured[T: BaseModel](
        self, profile: ModelProfile, system: str, prompt: str, schema: type[T]
    ) -> T | None:
        text = await self.invoke(profile, system, [{"role": "user", "content": prompt}])
        return self._parse(text, schema)

    def _parse[T: BaseModel](self, body: str, schema: type[T]) -> T | None:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end < start:
            self._logger.warning(f"No JSON object in {schema.__name__} output: {body[:200]!r}")
            return None
        try:
            return schema.model_validate_json(body[start : end + 1])
        except ValidationError as exception:
            self._logger.warning(f"Invalid {schema.__name__} output: {exception}")
            return None


def _sampling(profile: ModelProfile) -> dict[str, object]:
    sampling: dict[str, object] = {
        "temperature": profile.temperature,
        **thinking_switch(profile.reasoning_effort),
    }
    if profile.reasoning_effort:
        sampling["reasoning_effort"] = profile.reasoning_effort
    return sampling
