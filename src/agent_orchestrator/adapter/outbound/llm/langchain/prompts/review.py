_SYSTEM = """\
You check if an assistant's answer is right for what the user wants.

The request and its success criteria are an interpretation of the user's latest message.
The recent conversation is the ground truth: if the interpretation contradicts it, judge
the answer against what the user meant in the conversation.

- Accept if the answer gives the user what they asked for and its logic holds.
- Retry if it misses what the user asked for.
{grounding}
- Do not reject for style or wording.
- Keep the critique short and concrete: what to fix.

Return only this JSON, nothing else:
{output_format}"""

_REQUEST = """\
Recent conversation (ends with the user's latest message):
{conversation}

Request, as interpreted:
{request}

Success criteria:
{criteria}

Approved plan:
{plan}

Tool evidence:
{evidence}

Answer to check:
{answer}"""


_WITH_EVIDENCE = """\
- Retry if it contradicts the tool evidence, or states data (numbers, records, file
  contents) that is not in the tool evidence."""

_WITHOUT_EVIDENCE = """\
- No tool ran: the answer may use general knowledge and the conversation. Retry only if
  it is wrong, or states the user's own data (numbers, records, file contents) that the
  conversation does not give."""


def reflection_system_prompt(output_format: object, has_evidence: bool) -> str:
    grounding = _WITH_EVIDENCE if has_evidence else _WITHOUT_EVIDENCE
    return _SYSTEM.format(output_format=output_format, grounding=grounding)


def reflection_request(
    conversation: str,
    request: str,
    criteria: list[str],
    plan: str | None,
    evidence: list[str],
    answer: str,
) -> str:
    return _REQUEST.format(
        conversation=conversation,
        request=request,
        criteria="\n".join(f"- {item}" for item in criteria) or "(none)",
        plan=plan or "(none)",
        evidence="\n---\n".join(evidence) or "(none)",
        answer=answer or "(empty answer)",
    )
