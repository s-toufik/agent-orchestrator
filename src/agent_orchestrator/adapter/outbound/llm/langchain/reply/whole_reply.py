from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from agent_orchestrator.adapter.outbound.llm.langchain.model_call import invoke


class WholeReply:
    async def read(self, model: Runnable, messages: Sequence[BaseMessage]) -> BaseMessage:
        return await invoke(model, messages)
