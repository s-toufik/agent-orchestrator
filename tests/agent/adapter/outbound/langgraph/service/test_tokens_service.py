from langchain_core.messages import HumanMessage

from agent.adapter.outbound.langgraph.service.tokens_service import count_message_tokens


def test_empty_content_costs_only_the_per_message_overhead() -> None:
    assert count_message_tokens([HumanMessage(content="")]) == 4


def test_counts_grow_with_word_and_punctuation_count() -> None:
    short = count_message_tokens([HumanMessage(content="hello")])
    longer = count_message_tokens([HumanMessage(content="hello, world! how are you?")])

    assert longer > short


def test_sums_across_multiple_messages() -> None:
    single = count_message_tokens([HumanMessage(content="hello")])
    doubled = count_message_tokens([HumanMessage(content="hello"), HumanMessage(content="hello")])

    assert doubled == single * 2


def test_non_string_content_is_stringified() -> None:
    message = HumanMessage(content=[{"type": "text", "text": "hi"}])

    assert count_message_tokens([message]) > 4
