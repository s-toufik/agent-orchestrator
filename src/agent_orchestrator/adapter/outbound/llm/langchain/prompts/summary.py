_SYSTEM = """\
You keep a running summary of a conversation between a user and an assistant.
Merge the existing summary with the new messages.
Keep facts, figures, file paths, decisions and open questions. Drop small talk.
At most 200 words. Plain text only."""

_REQUEST = """\
Existing summary:
{summary}

New messages:
{messages}"""


def summary_system_prompt() -> str:
    return _SYSTEM


def summary_request(summary: str, messages: str) -> str:
    return _REQUEST.format(summary=summary or "(none)", messages=messages)
