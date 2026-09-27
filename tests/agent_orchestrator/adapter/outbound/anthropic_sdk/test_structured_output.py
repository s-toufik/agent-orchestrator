import json
from typing import Any, cast

import httpx2
from anthropic import AsyncAnthropic

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import profile


class Block:
    def __init__(self, text: str, kind: str = "text") -> None:
        self.type = kind
        self.text = text


class FakeStream:
    def __init__(self, blocks: list[Block]) -> None:
        self._blocks = blocks

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get_final_message(self) -> Any:
        return type("Response", (), {"content": self._blocks})()


class FakeMessages:
    def __init__(self, blocks: list[Block]) -> None:
        self._blocks = blocks
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self._blocks)


def _completion(blocks: list[Block], logger) -> tuple[StructuredOutput, FakeMessages]:
    messages = FakeMessages(blocks)
    client = type("Client", (), {"messages": messages})()
    return StructuredOutput(cast(AsyncAnthropic, client), logger), messages


async def test_json_is_parsed_past_thinking_and_prose(logger) -> None:
    completion, messages = _completion(
        [Block('<think>{"no": 1}</think>Sure: {"action": "retry", "critique": "units"}')], logger
    )

    verdict = await completion.invoke_structured(profile(), "system", "prompt", ReflectionDecision)

    assert verdict == ReflectionDecision(action="retry", critique="units")
    assert messages.calls[0]["model"] == "qwen3-8b"
    assert messages.calls[0]["system"] == "system"
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert messages.calls[0]["extra_body"] == {
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def test_the_model_temperature_and_reasoning_are_forwarded(logger) -> None:
    completion, messages = _completion([Block('{"action": "accept"}')], logger)

    await completion.invoke_structured(
        profile(reasoning_effort="medium"), "s", "p", ReflectionDecision
    )

    assert messages.calls[0]["extra_body"] == {
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "medium",
    }


async def test_non_text_blocks_are_ignored(logger) -> None:
    completion, _ = _completion(
        [Block("ignored", kind="thinking"), Block('{"action": "accept"}')], logger
    )

    assert await completion.invoke_structured(
        profile(), "s", "p", ReflectionDecision
    ) == ReflectionDecision(action="accept")


async def test_missing_or_invalid_json_returns_none_and_warns(logger) -> None:
    for text in ("no json here", '{"action": "maybe"}'):
        completion, _ = _completion([Block(text)], logger)

        assert await completion.invoke_structured(profile(), "s", "p", ReflectionDecision) is None

    assert len(logger.messages("warning")) == 2


def _sse(*events: dict[str, Any]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


async def test_a_long_output_budget_goes_through_the_real_client(logger) -> None:
    requests: list[dict[str, Any]] = []

    def gateway(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        body = _sse(
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "qwen3-8b",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "1. Echo"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = AsyncAnthropic(
        base_url="http://gateway",
        api_key="none",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(gateway)),
    )

    text = await StructuredOutput(client, logger).invoke(
        profile(), "system", [{"role": "user", "content": "plan"}], max_tokens=30_000
    )

    assert text == "1. Echo"
    assert requests[0]["max_tokens"] == 30_000
    assert requests[0]["stream"] is True
    assert requests[0]["temperature"] == 0.0
