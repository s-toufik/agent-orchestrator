import json

from agent.adapter.inbound.web.schema.agent_message_schema import AgentMessageSchema
from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.enum.agent_message_type import AgentMessageType


def test_serialize_defaults() -> None:
    schema = AgentMessageSchema(session_id="s1", content="hello")

    text = schema.serialize().decode("utf-8")

    event_line, data_line, blank = text.split("\n", 2)
    assert event_line == "event: final"
    assert blank == "\n"
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload == {
        "session_id": "s1",
        "content": "hello",
        "metadata": {},
        "error": None,
        "message_status": "final",
        "message_type": "text",
    }


def test_serialize_with_error_and_custom_status_type() -> None:
    schema = AgentMessageSchema(
        session_id="s2",
        content="",
        metadata={"k": "v"},
        error="boom",
        message_status=MessageStreamType.ERROR,
        message_type=AgentMessageType.IMAGE,
    )

    text = schema.serialize().decode("utf-8")
    event_line, data_line, _ = text.split("\n", 2)

    assert event_line == "event: error"
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload["error"] == "boom"
    assert payload["message_type"] == "image"
    assert payload["metadata"] == {"k": "v"}


def test_multiline_content_does_not_break_the_frame() -> None:
    traceback_text = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad'
    schema = AgentMessageSchema(session_id="s3", content="", error=traceback_text)

    text = schema.serialize().decode("utf-8")

    assert text.endswith("\n\n")
    _, data_line, _ = text.split("\n", 2)
    assert json.loads(data_line.removeprefix("data: "))["error"] == traceback_text
