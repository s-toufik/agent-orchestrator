def thinking_switch(reasoning_effort: str | None) -> dict[str, object]:
    return {"chat_template_kwargs": {"enable_thinking": reasoning_effort is not None}}
