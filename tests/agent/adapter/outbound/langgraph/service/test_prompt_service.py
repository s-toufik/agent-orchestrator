from agent.adapter.outbound.langgraph.service.prompt_service import PromptService


def test_planner_system_prompt_is_static() -> None:
    service = PromptService()

    assert "planner" in service.planner_system_prompt().lower()


def test_reflection_system_prompt_embeds_the_output_format() -> None:
    service = PromptService()

    prompt = service.reflection_system_prompt('{"type": "object"}')

    assert '{"type": "object"}' in prompt


def test_feedback_system_prompt_embeds_the_critique() -> None:
    service = PromptService()

    prompt = service.feedback_system_prompt("missing the total")

    assert "missing the total" in prompt
