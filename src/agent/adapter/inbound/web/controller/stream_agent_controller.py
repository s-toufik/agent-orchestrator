import asyncio
import traceback
from asyncio import Task
from collections.abc import AsyncIterator, Callable

from pycraftcore.http.context.request_context import request_context, request_id_context
from pycraftcore.logger.port import Logger
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse

from agent.adapter.inbound.web.schema.agent_message_schema import AgentMessageSchema
from agent.adapter.inbound.web.schema.agent_message_stream_schema import (
    AgentMessageStreamSchema,
)
from agent.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent.application.port.inbound.stream_agent_port import StreamAgentPort
from agent.application.port.outbound.sse_queue_port import SSEQueuePort
from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.model.agent_message_stream import AgentMessageStream
from agent.domain.model.agent_request import AgentRequest


class StreamAgentController:
    def __init__(
        self,
        use_case: StreamAgentPort,
        stream_events: Callable[[], SSEQueuePort],
        logger: Logger,
        max_concurrent_streams: int = 200,
    ) -> None:
        self._use_case = use_case
        self._stream_events = stream_events
        self._logger = logger
        self._admission = asyncio.Semaphore(max_concurrent_streams)

    async def execute(self, request: AgentRequestSchema) -> StreamingResponse:
        request_id: str = request.request_id or request_id_context.get() or "N/A"

        if self._admission.locked():
            self._logger.warning(f"[{request_id}] rejected: server at capacity")
            raise HTTPException(status_code=503, detail="Server is at capacity, please retry.")

        await self._admission.acquire()

        try:
            starlette_request: Request | None = request_context.get() or None
            self._logger.info(f"[{request_id}] stream request accepted")

            domain_request: AgentRequest = request.to_domain()
            events: SSEQueuePort = self._stream_events()

            use_case_task: Task[None] = asyncio.create_task(
                self._use_case.execute(domain_request, events)
            )
        except Exception:
            self._admission.release()
            raise

        return StreamingResponse(
            self._event_generator(request_id, request, events, use_case_task, starlette_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _event_generator(
        self,
        request_id: str,
        request: AgentRequestSchema,
        events: SSEQueuePort,
        use_case_task: Task[None],
        starlette_request: Request | None,
    ) -> AsyncIterator[bytes]:
        try:
            while True:
                if starlette_request is not None and await starlette_request.is_disconnected():
                    raise asyncio.CancelledError

                event: AgentMessageStream = await events.queue.get()

                if event.type is MessageStreamType.FINAL:
                    yield AgentMessageSchema(
                        session_id=request.request_id,
                        content=event.content,
                        metadata=event.metadata,
                    ).serialize()
                else:
                    yield AgentMessageStreamSchema(
                        type=event.type, content=event.content
                    ).serialize()

                if event.type is MessageStreamType.COMPLETE:
                    break

        except asyncio.CancelledError:
            self._logger.warning(f"[{request_id}] stream cancelled by the client")
            raise
        except Exception as exception:
            traceback_str: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"[{request_id}] unhandled streaming error:\n{traceback_str}")
            yield AgentMessageSchema(
                session_id=request.request_id, content="", error=traceback_str
            ).serialize()
            raise
        finally:
            await self._cancel(use_case_task)
            self._admission.release()

    @staticmethod
    async def _cancel(task: Task[None]) -> None:
        if task.done():
            return
        task.cancel()
        try:
            await task
        except BaseException:
            pass
