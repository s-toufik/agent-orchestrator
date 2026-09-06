from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

_SCHEMA_INSTRUCTION = (
    "\n\nRespond with ONLY a single valid JSON object matching this schema. "
    "No markdown fences, no commentary.\n{schema}"
)


class LocalJudgeModel(DeepEvalBaseLLM):
    """Wraps the project's own local LLM as a DeepEval judge.

    DeepEval metrics default to calling out to an external provider (OpenAI)
    to score test cases. This wraps whichever `ChatOpenAI` client the agent
    itself already talks to (LM Studio, served from `connector.llm.base_url`)
    so evaluation never leaves the machine.
    """

    def __init__(self, chat: ChatOpenAI) -> None:
        self._chat = chat
        super().__init__(model=chat.model_name)

    def load_model(self) -> ChatOpenAI:  # ty: ignore[invalid-method-override]
        return self._chat

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str:
        return self._as_text(self._chat.invoke(self._with_schema(prompt, schema)))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str:
        return self._as_text(await self._chat.ainvoke(self._with_schema(prompt, schema)))

    def get_model_name(self) -> str:
        return self._chat.model_name

    @staticmethod
    def _as_text(message: BaseMessage) -> str:
        content = message.content
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _with_schema(prompt: str, schema: type[BaseModel] | None) -> str:
        if schema is None:
            return prompt
        return prompt + _SCHEMA_INSTRUCTION.format(schema=schema.model_json_schema())
