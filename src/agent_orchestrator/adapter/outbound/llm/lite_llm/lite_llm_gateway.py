import asyncio
import json
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx2
from pycraftcore.logger.port import Logger


class LiteLlmGateway:
    def __init__(
        self,
        command: list[str],
        config: dict[str, Any],
        logger: Logger,
        host: str = "127.0.0.1",
        port: int = 4000,
        startup_timeout: float = 180.0,
    ) -> None:
        self._command = command
        self._config = config
        self._logger = logger
        self._host = host
        self._port = port
        self._startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._output: deque[str] = deque(maxlen=30)
        self._pump: asyncio.Task[None] | None = None
        self._directory: TemporaryDirectory[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def start(self) -> None:
        self._directory = TemporaryDirectory(prefix="litellm-")
        config_path = Path(self._directory.name) / "config.yaml"
        config_path.write_text(json.dumps(self._config))

        self._logger.info(f"Starting LiteLLM gateway on {self.base_url}")
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            "--config",
            str(config_path),
            "--host",
            self._host,
            "--port",
            str(self._port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._pump = asyncio.create_task(self._read_output(self._process))
        try:
            await asyncio.wait_for(self._wait_until_ready(), self._startup_timeout)
        except BaseException:
            await self.stop()
            raise
        self._logger.info("LiteLLM gateway ready")

    async def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self._pump is not None:
            await asyncio.gather(self._pump, return_exceptions=True)
            self._pump = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    async def _wait_until_ready(self) -> None:
        async with httpx2.AsyncClient(timeout=2.0) as client:
            while True:
                if self._process is None or self._process.returncode is not None:
                    raise RuntimeError(
                        "LiteLLM gateway exited during startup:\n" + "\n".join(self._output)
                    )
                try:
                    response = await client.get(f"{self.base_url}/health/liveliness")
                    if response.status_code == 200:
                        return
                except httpx2.TransportError:
                    pass
                await asyncio.sleep(0.5)

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        async for raw in process.stdout:
            line = raw.decode(errors="replace").rstrip()
            self._output.append(line)
            self._logger.debug(f"[litellm] {line}")
