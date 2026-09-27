def thinking_switch(reasoning_effort: str | None) -> dict[str, object]:
    # llama.cpp ignores reasoning_effort for Qwen: thinking is switched by the chat template,
    # per request. Qwen has no levels, so any effort means on. Models without a thinking
    # mode ignore the flag.
    return {"chat_template_kwargs": {"enable_thinking": reasoning_effort is not None}}
