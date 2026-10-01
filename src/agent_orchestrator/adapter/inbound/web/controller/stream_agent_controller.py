import asyncio
import traceback
from asyncio import Task
from collections.abc import AsyncIterator, Callable

from pycraftcore.http.context.request_context import request_context, request_id_context
from pycraftcore.logger.port import Logger
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse

from agent_orchestrator.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent_orchestrator.adapter.inbound.web.sse_presenter import SsePresenter
from agent_orchestrator.application.port.inbound.handle_message_port import HandleMessagePort
from agent_orchestrator.application.port.outbound.turn_event_stream import TurnEventStream


class StreamAgentController:
    def __init__(
        self,
        use_case: HandleMessagePort,
        stream_events: Callable[[], TurnEventStream],
        logger: Logger,
        max_concurrent_streams: int = 200,
        presenter: SsePresenter | None = None,
    ) -> None:
        self._use_case = use_case
        self._stream_events = stream_events
        self._logger = logger
        self._admission = asyncio.Semaphore(max_concurrent_streams)
        self._presenter = presenter or SsePresenter()

    async def execute(self, request: AgentRequestSchema) -> StreamingResponse:
        if request.request_id:
            request_id_context.set(request.request_id)

        if self._admission.locked():
            self._logger.warning("rejected: server at capacity")
            raise HTTPException(status_code=503, detail="Server is at capacity, please retry.")

        await self._admission.acquire()

        try:
            starlette_request: Request | None = request_context.get() or None
            self._logger.info("stream request accepted")
            events: TurnEventStream = self._stream_events()
            use_case_task: Task[None] = asyncio.create_task(
                self._use_case.handle(request.to_domain(), events)
            )
        except Exception:
            self._admission.release()
            raise

        return StreamingResponse(
            self._event_generator(request, events, use_case_task, starlette_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _event_generator(
        self,
        request: AgentRequestSchema,
        events: TurnEventStream,
        use_case_task: Task[None],
        starlette_request: Request | None,
    ) -> AsyncIterator[bytes]:
        try:
            stream = aiter(events)
            while True:
                if starlette_request is not None and await starlette_request.is_disconnected():
                    raise asyncio.CancelledError
                event = await anext(stream, None)
                if event is None:
                    break
                for chunk in self._presenter.present(event, request.request_id):
                    yield chunk
            yield self._presenter.complete()

        except asyncio.CancelledError:
            self._logger.warning("stream cancelled by the client")
            raise
        except Exception as exception:
            trace: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"unhandled streaming error:\n{trace}")
            yield self._presenter.error(trace)
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
