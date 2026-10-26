from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types
from websockets.exceptions import ConnectionClosed

from libclaim.live import session as events
from libclaim.live.bridge import LiveConfig

PROMPT_PATH = Path(__file__).with_name("prompt.md")
RATE_LIMITED_CODE = 429
MODALITY_BY_MEDIA = {"AUDIO": "audio", "VIDEO": "video", "IMAGE": "image", "TEXT": "text"}


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_config(
    config: LiveConfig,
    specs: Sequence[events.ToolSpec],
    handle: str | None,
    voice: str = "",
    prompt: str | None = None,
) -> types.LiveConnectConfig:
    declarations = [
        types.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            parameters_json_schema=spec.parameters,
            behavior=types.Behavior.NON_BLOCKING if spec.non_blocking else None,
        )
        for spec in specs
    ]
    compression = (
        types.ContextWindowCompressionConfig(
            trigger_tokens=config.compression_trigger_tokens,
            sliding_window=types.SlidingWindow(target_tokens=config.compression_target_tokens),
        )
        if config.compression_enabled
        else None
    )
    speech = (
        types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))
        if voice
        else None
    )
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=prompt if prompt is not None else load_prompt(),
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        session_resumption=types.SessionResumptionConfig(handle=handle),
        context_window_compression=compression,
        media_resolution=types.MediaResolution(config.media_resolution),
        speech_config=speech,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


def usage_totals(details: Sequence[types.ModalityTokenCount] | None) -> dict[str, int]:
    totals: dict[str, int] = {}
    for detail in details or []:
        modality = MODALITY_BY_MEDIA.get(detail.modality.name, "text") if detail.modality else "text"
        totals[modality] = totals.get(modality, 0) + (detail.token_count or 0)
    return totals


def events_from_message(message: types.LiveServerMessage) -> list[events.Event]:
    found: list[events.Event] = []
    if message.usage_metadata:
        found.append(
            events.Usage(
                input=usage_totals(message.usage_metadata.prompt_tokens_details),
                output=usage_totals(message.usage_metadata.response_tokens_details),
            )
        )
    update = message.session_resumption_update
    if update and update.resumable and update.new_handle:
        found.append(events.Handle(update.new_handle))
    if message.tool_call:
        found.extend(events.ToolCall(call.id or "", call.name or "", dict(call.args or {})) for call in message.tool_call.function_calls or [])
    content = message.server_content
    if content:
        if content.interrupted:
            found.append(events.Interrupted())
        if content.model_turn:
            found.extend(
                events.Audio(part.inline_data.data)
                for part in content.model_turn.parts or []
                if part.inline_data and part.inline_data.data
            )
        if content.input_transcription and content.input_transcription.text:
            found.append(events.Transcript("user", content.input_transcription.text))
        if content.output_transcription and content.output_transcription.text:
            found.append(events.Transcript("model", content.output_transcription.text))
        if content.turn_complete:
            found.append(events.TurnComplete())
    if message.go_away:
        found.append(events.GoAway())
    return found


def translate(error: Exception) -> Exception:
    if isinstance(error, errors.APIError) and error.code == RATE_LIMITED_CODE:
        return events.RateLimited()
    return events.SessionClosed()


class GenaiSession:
    def __init__(self, session: Any, config: LiveConfig):
        self.session = session
        self.config = config

    async def send_audio(self, data: bytes) -> None:
        await self.session.send_realtime_input(audio=types.Blob(data=data, mime_type=self.config.audio_mime))

    async def send_video(self, data: bytes) -> None:
        await self.session.send_realtime_input(video=types.Blob(data=data, mime_type=self.config.frame_mime))

    async def send_text(self, text: str) -> None:
        await self.session.send_realtime_input(text=text)

    async def send_tool_response(
        self,
        call_id: str,
        name: str,
        response: dict[str, Any],
        will_continue: bool,
        scheduling: events.Scheduling,
    ) -> None:
        await self.session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response=response,
                    will_continue=will_continue,
                    scheduling=types.FunctionResponseScheduling(scheduling.value),
                )
            ]
        )

    async def receive(self) -> AsyncIterator[events.Event]:
        try:
            while True:
                async for message in self.session.receive():
                    for event in events_from_message(message):
                        yield event
        except (errors.APIError, ConnectionClosed) as error:
            raise translate(error) from error


class GenaiFactory:
    def __init__(
        self,
        client: genai.Client,
        model: str,
        config: LiveConfig,
        specs: Sequence[events.ToolSpec],
        voice: str = "",
    ):
        self.client = client
        self.model = model
        self.config = config
        self.specs = specs
        self.voice = voice
        self.prompt = load_prompt()

    @asynccontextmanager
    async def connect(self, handle: str | None) -> AsyncIterator[events.LiveSession]:
        live_config = build_config(self.config, self.specs, handle, self.voice, self.prompt)
        try:
            async with self.client.aio.live.connect(model=self.model, config=live_config) as session:
                yield GenaiSession(session, self.config)
        except (errors.APIError, ConnectionClosed) as error:
            raise translate(error) from error

    def __call__(self, handle: str | None):
        return self.connect(handle)
