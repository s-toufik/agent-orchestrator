import json

import pytest

from agent.adapter.inbound.web.schema.agent_message_stream_schema import AgentMessageStreamSchema
from agent.domain.enum.agent_message_status import MessageStreamType


@pytest.mark.parametrize("stream_type", list(MessageStreamType))
def test_serialize_for_every_stream_type(stream_type) -> None:
    schema = AgentMessageStreamSchema(type=stream_type, content="payload")

    text = schema.serialize().decode("utf-8")
    event_line, data_line, blank = text.split("\n", 2)

    assert event_line == f"event: {stream_type.value}"
    assert blank == "\n"
    assert json.loads(data_line.removeprefix("data: ")) == {
        "type": stream_type.value,
        "content": "payload",
    }


def test_multiline_content_does_not_break_the_frame() -> None:
    schema = AgentMessageStreamSchema(type=MessageStreamType.TOKEN, content="line one\nline two")

    text = schema.serialize().decode("utf-8")

    assert text.endswith("\n\n")
    _, data_line, _ = text.split("\n", 2)
    assert json.loads(data_line.removeprefix("data: "))["content"] == "line one\nline two"
