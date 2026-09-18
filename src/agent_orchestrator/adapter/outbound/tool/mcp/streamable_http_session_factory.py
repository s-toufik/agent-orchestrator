import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx2
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication import AuthTyping
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.token_auth import TokenAuth
from pycraftcore.http.context.request_context import request_id_context
from pycraftcore.logger.port import Logger

from agent_orchestrator.domain.exception.tool_unavailable_exception import ToolUnavailableException


class StreamableHttpSessionFactory:
    def __init__(
        self,
        connector: McpConnector,
        logger: Logger,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        max_delay: float = 5.0,
    ) -> None:
        self._connector = connector
        self._logger = logger
        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._max_delay = max_delay

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with AsyncExitStack() as stack:
            yield await self._connect_with_retry(stack)

    async def _connect_with_retry(self, stack: AsyncExitStack) -> ClientSession:
        delay = self._initial_delay
        last_exception: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            attempt_stack = AsyncExitStack()
            try:
                session: ClientSession = await self._connect(attempt_stack)
                await stack.enter_async_context(attempt_stack)
                self._logger.info(
                    f"MCP session established on {self._connector.base_url} "
                    f"(attempt {attempt}/{self._max_attempts})"
                )
                return session
            except Exception as exception:
                await attempt_stack.aclose()
                last_exception = exception
                self._logger.warning(
                    f"MCP connection to {self._connector.base_url} failed "
                    f"(attempt {attempt}/{self._max_attempts}): {exception}"
                )
                if attempt < self._max_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._max_delay)

        raise ToolUnavailableException(
            f"Could not reach the MCP server at {self._connector.base_url} "
            f"after {self._max_attempts} attempts"
        ) from last_exception

    async def _connect(self, stack: AsyncExitStack) -> ClientSession:
        read, write = await self._open_transport(stack)
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _open_transport(self, stack: AsyncExitStack):
        headers: dict[str, str] = self._headers()

        if self._connector.transport == "sse":
            return await stack.enter_async_context(
                sse_client(
                    self._connector.base_url,
                    headers=headers,
                    timeout=self._connector.timeout,
                )
            )
        http_client = httpx2.AsyncClient(
            headers=headers,
            timeout=httpx2.Timeout(self._connector.timeout),
            follow_redirects=True,
            trust_env=False,
        )
        await stack.enter_async_context(http_client)
        return await stack.enter_async_context(
            streamable_http_client(self._connector.base_url, http_client=http_client)
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}

        request_id: str | None = request_id_context.get()
        if request_id:
            headers["X-Request-ID"] = request_id

        auth: AuthTyping = self._connector.auth
        if isinstance(auth, TokenAuth):
            headers[auth.key_name] = auth.key_value
        elif isinstance(auth, BasicAuth):
            token = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        return headers
