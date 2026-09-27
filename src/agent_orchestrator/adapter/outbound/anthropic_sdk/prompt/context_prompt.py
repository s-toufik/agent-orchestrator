import json

_SYSTEM = """\
You read a conversation and work out what the user wants from their latest message.
You never answer the user.

Read the latest message as the next move in the conversation, never in isolation.
A message that is short, partial or unclear on its own usually depends on the earlier
turns: it may extend, deepen, reframe, narrow, correct or react to the previous exchange.
Treat it as a new subject only when it clearly moves away from what was being discussed.

Pick the intent from what the assistant must do next:
- task: the user needs something that requires data, tools or actions.
- continuation: the user wants the assistant to rework the previous exchange (go further,
  go deeper, change its scope, its angle or its form) without new data.
- direct: a new, self-contained request that the assistant can answer from general
  knowledge, with no data and no tools.
- plan_approval: the user accepts the pending plan as it is.
- plan_revision: the user wants the pending plan changed.
- ambiguous: even with the whole conversation, you cannot tell what the user wants.

Fill the fields:
- standalone_query: the complete request the user is making now, written so that someone
  who never saw the conversation could act on it. When the message depends on earlier
  turns, merge it with them: keep the earlier subject and apply what the latest message
  adds, restricts or corrects. Keep the user's language.
- success_criteria: 1 to 3 checks on the substance of the answer the user expects.
- clarification_question: one short question, only when the intent is ambiguous.

plan_approval and plan_revision are only possible when a pending plan is shown.

Return only one JSON object matching this schema, nothing else:
{schema}"""

_REQUEST = """\
Conversation so far:
{history}

Pending plan awaiting approval:
{pending_plan}

Latest user message:
{message}"""


def context_system_prompt(output_format: dict) -> str:
    return _SYSTEM.format(schema=json.dumps(output_format))


def context_request(history: str, pending_plan: str | None, message: str) -> str:
    return _REQUEST.format(history=history, pending_plan=pending_plan or "(none)", message=message)
