from agent.adapter.outbound.langgraph.build_agent import build_agent
from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state
from agent.adapter.outbound.llm.schema import ModelParameters
from agent.adapter.outbound.tool.tool_registry import ToolRegistry
from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome
from agent.domain.model.tool_specification import ToolSpecification


class FakeAIMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class SequencedBoundLLM:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)

    async def ainvoke(self, messages):
        return self._responses.pop(0)


class SequencedLLM:
    def __init__(self, responses: list) -> None:
        self._bound = SequencedBoundLLM(responses)

    def bind_tools(self, tools):
        return self._bound


class FakeStructuredRunnable:
    def __init__(self, result: dict) -> None:
        self._result = result

    async def ainvoke(self, messages):
        return self._result


class ReflectionLLM:
    def __init__(self, action: ReflectionAction) -> None:
        self._decision = ReflectionDecision(action=action, critique="fine")

    def with_structured_output(self, schema, include_raw=True):
        return FakeStructuredRunnable({"parsed": self._decision})


class EchoTool:
    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(name="echo", description="Echoes.", parameters={"type": "object"})

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome:
        return ToolOutcome(
            invocation_id=invocation.id,
            tool_name="echo",
            output=f"echo:{invocation.arguments.get('text')}",
        )


def _model_parameters(
    model_name: str = "m",
    temperature: float = 0.0,
    max_output_tokens: int = 100,
    max_context_tokens: int = 4_000,
    max_iterations: int = 6,
    use_streaming: bool = False,
) -> ModelParameters:
    return ModelParameters(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_context_tokens=max_context_tokens,
        max_iterations=max_iterations,
        use_streaming=use_streaming,
    )


async def test_a_tool_call_then_a_final_answer_flows_through_the_whole_graph(logger) -> None:
    planner = SequencedLLM(
        [
            FakeAIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "hi"}}]),
            FakeAIMessage(content="final answer"),
        ]
    )
    reflection = ReflectionLLM(ReflectionAction.ACCEPT)
    tool_registry = ToolRegistry([EchoTool()])

    graph, _ = build_agent(
        planner_llm=planner,
        reflection_llm=reflection,
        tool_registry=tool_registry,
        model_parameters=_model_parameters(),
        logger=logger,
    )

    initial = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.USER, content="say hi")])
    )
    result = await graph.ainvoke(pack_state(initial), config={"configurable": {"thread_id": "t1"}})
    final_state = unpack_state(result)

    assert final_state.final_answer == "final answer"
    assert final_state.last_node == "final"
    tool_messages = [m for m in final_state.conversation.messages if m.role is Role.TOOL]
    assert tool_messages[0].content == "echo:hi"


async def test_a_retry_requested_by_reflection_loops_back_through_feedback(logger) -> None:
    planner = SequencedLLM(
        [
            FakeAIMessage(content="first try"),
            FakeAIMessage(content="second try"),
        ]
    )

    class TwoStepReflection:
        def __init__(self) -> None:
            self._decisions = [
                ReflectionDecision(action=ReflectionAction.RETRY, critique="too short"),
                ReflectionDecision(action=ReflectionAction.ACCEPT, critique="fine"),
            ]

        def with_structured_output(self, schema, include_raw=True):
            decision = self._decisions.pop(0)
            return FakeStructuredRunnable({"parsed": decision})

    graph, _ = build_agent(
        planner_llm=planner,
        reflection_llm=TwoStepReflection(),
        tool_registry=ToolRegistry([]),
        model_parameters=_model_parameters(),
        logger=logger,
    )

    initial = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.USER, content="hi")])
    )
    result = await graph.ainvoke(pack_state(initial), config={"configurable": {"thread_id": "t2"}})
    final_state = unpack_state(result)

    assert final_state.final_answer == "second try"


async def test_hitting_max_iterations_forces_a_final_answer_without_reflection(logger) -> None:
    # Planner always asks for a tool it never gets to use meaningfully -- the
    # router's max_iterations guard must cut the loop short regardless.
    planner = SequencedLLM(
        [
            FakeAIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": str(i)}}])
            for i in range(10)
        ]
    )
    graph, _ = build_agent(
        planner_llm=planner,
        reflection_llm=ReflectionLLM(ReflectionAction.ACCEPT),
        tool_registry=ToolRegistry([EchoTool()]),
        model_parameters=_model_parameters(max_iterations=2),
        logger=logger,
    )

    initial = AgentState(max_iterations=2)
    result = await graph.ainvoke(pack_state(initial), config={"configurable": {"thread_id": "t3"}})
    final_state = unpack_state(result)

    assert final_state.last_node == "final"
    assert final_state.iteration == 2
