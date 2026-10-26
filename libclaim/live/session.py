from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Scheduling(StrEnum):
    SILENT = "SILENT"
    WHEN_IDLE = "WHEN_IDLE"
    INTERRUPT = "INTERRUPT"


class RateLimited(Exception):
    pass


class SessionClosed(Exception):
    pass


@dataclass(frozen=True)
class Audio:
    data: bytes


@dataclass(frozen=True)
class Transcript:
    speaker: str
    text: str


@dataclass(frozen=True)
class Interrupted:
    pass


@dataclass(frozen=True)
class TurnComplete:
    pass


@dataclass(frozen=True)
class GoAway:
    pass


@dataclass(frozen=True)
class Handle:
    value: str


@dataclass(frozen=True)
class Usage:
    input: dict[str, int] = field(default_factory=dict)
    output: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


Event = Audio | Transcript | Interrupted | TurnComplete | GoAway | Handle | Usage | ToolCall


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    non_blocking: bool = False


class Tools(Protocol):
    monitor_name: str

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class LiveSession(Protocol):
    async def send_audio(self, data: bytes) -> None: ...

    async def send_video(self, data: bytes) -> None: ...

    async def send_text(self, text: str) -> None: ...

    async def send_tool_response(
        self,
        call_id: str,
        name: str,
        response: dict[str, Any],
        will_continue: bool,
        scheduling: Scheduling,
    ) -> None: ...

    def receive(self) -> AsyncIterator[Event]: ...


SessionFactory = Callable[[str | None], AbstractAsyncContextManager[LiveSession]]
