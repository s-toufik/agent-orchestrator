import asyncio
import base64
from contextlib import AsyncExitStack

import httpx2
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication import AuthTyping
from pycraftcore.authentication.model.basic_auth import BasicAuth
from pycraftcore.authentication.model.token_auth import TokenAuth
from pycraftcore.logger.port import Logger

from agent.domain.exception.tool_unavailable_exception import ToolUnavailableException


class StreamableHttpSessionFactory:
    def __init__(
        self,
        connector: McpConnector,
        logger: Logger,
        max_attempts: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 15.0,
    ) -> None:
        self._connector = connector
        self._logger = logger
        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            await self._connect_with_retry()

    async def session(self) -> ClientSession:
        async with self._lock:
            if self._session is None:
                await self._connect_with_retry()
            assert self._session is not None
            return self._session

    async def invalidate(self) -> None:
        async with self._lock:
            await self._teardown()

    async def close(self) -> None:
        async with self._lock:
            await self._teardown()

    async def _connect_with_retry(self) -> None:
        delay = self._initial_delay
        last_exception: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._connect()
                self._logger.info(
                    f"MCP session established on {self._connector.base_url} "
                    f"(attempt {attempt}/{self._max_attempts})"
                )
                return
            except Exception as exception:
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

    async def _connect(self) -> None:
        stack = AsyncExitStack()
        try:
            read, write = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    async def _open_transport(self, stack: AsyncExitStack):
        headers = self._headers()

        if self._connector.transport == "sse":
            return await stack.enter_async_context(
                sse_client(
                    self._connector.base_url,
                    headers=headers,
                    timeout=self._connector.timeout,
                )
            )

        http_client = create_mcp_http_client(
            headers=headers, timeout=httpx2.Timeout(self._connector.timeout)
        )
        await stack.enter_async_context(http_client)
        return await stack.enter_async_context(
            streamable_http_client(self._connector.base_url, http_client=http_client)
        )

    async def _teardown(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is None:
            return
        try:
            await stack.aclose()
        except Exception as exception:
            self._logger.warning(f"MCP session teardown raised: {exception}")

    def _headers(self) -> dict[str, str]:
        auth: AuthTyping = self._connector.auth
        if isinstance(auth, TokenAuth):
            return {auth.key_name: auth.key_value}
        if isinstance(auth, BasicAuth):
            token = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            return {"Authorization": f"Basic {token}"}
        return {}
