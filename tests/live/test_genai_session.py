import pytest
from google.genai import errors, types
from websockets.exceptions import ConnectionClosed

from libclaim.live import session as events
from libclaim.live.bridge import load_live_config
from libclaim.live.genai_session import GenaiSession, build_config, events_from_message, load_prompt, translate, usage_totals


def test_config_carries_compression_resumption_and_low_resolution():
    config = load_live_config()
    spec = events.ToolSpec("capture_monitor", "Start the monitor.", non_blocking=True)
    built = build_config(config, [spec], "h1")
    assert built.context_window_compression.trigger_tokens == 12000
    assert built.context_window_compression.sliding_window.target_tokens == 6000
    assert built.session_resumption.handle == "h1"
    assert built.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert built.tools[0].function_declarations[0].behavior == types.Behavior.NON_BLOCKING
    assert built.system_instruction == load_prompt()


def test_compression_can_be_disabled():
    config = load_live_config().model_copy(update={"compression_enabled": False})
    assert build_config(config, [], None).context_window_compression is None


def test_prompt_carries_the_rules():
    prompt = load_prompt()
    for phrase in ("country and currency", "2 seconds", "wall meets the ceiling", "original or a print", "Never state a title"):
        assert phrase in prompt


def test_prompt_directs_locale_and_statement_tool_use():
    prompt = load_prompt()
    for phrase in ("call set_locale", "Never guess", "call record_user_statement"):
        assert phrase in prompt


def test_message_is_normalised():
    message = types.LiveServerMessage(
        usage_metadata=types.UsageMetadata(
            prompt_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.VIDEO, token_count=264)],
            response_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=50)],
        ),
        session_resumption_update=types.LiveServerSessionResumptionUpdate(new_handle="h9", resumable=True),
        server_content=types.LiveServerContent(
            interrupted=True,
            model_turn=types.Content(parts=[types.Part(inline_data=types.Blob(data=b"pcm", mime_type="audio/pcm"))]),
            output_transcription=types.Transcription(text="hello"),
            turn_complete=True,
        ),
        go_away=types.LiveServerGoAway(time_left="5s"),
    )
    assert events_from_message(message) == [
        events.Usage(input={"video": 264}, output={"audio": 50}),
        events.Handle("h9"),
        events.Interrupted(),
        events.Audio(b"pcm"),
        events.Transcript("model", "hello"),
        events.TurnComplete(),
        events.GoAway(),
    ]


def test_unresumable_update_is_ignored():
    message = types.LiveServerMessage(
        session_resumption_update=types.LiveServerSessionResumptionUpdate(new_handle="h", resumable=False)
    )
    assert events_from_message(message) == []


def test_tool_call_is_normalised():
    message = types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(function_calls=[types.FunctionCall(id="c1", name="capture_monitor", args={"unit": "A"})])
    )
    assert events_from_message(message) == [events.ToolCall("c1", "capture_monitor", {"unit": "A"})]


def test_429_maps_to_rate_limited():
    assert isinstance(translate(errors.APIError(429, {"error": {}})), events.RateLimited)
    assert isinstance(translate(errors.APIError(500, {"error": {}})), events.SessionClosed)


class ClosedSocket:
    async def receive(self):
        return
        yield

    async def send_realtime_input(self, **kwargs):
        raise ConnectionClosed(None, None)


async def test_receive_that_ends_without_messages_raises_instead_of_spinning():
    session = GenaiSession(ClosedSocket(), load_live_config())
    with pytest.raises(events.SessionClosed):
        async for _ in session.receive():
            pass


async def test_send_on_closed_socket_is_translated():
    session = GenaiSession(ClosedSocket(), load_live_config())
    with pytest.raises(events.SessionClosed):
        await session.send_audio(b"x")


def test_unknown_modality_is_not_priced_as_text():
    details = [types.ModalityTokenCount(modality=types.MediaModality.DOCUMENT, token_count=7)]
    assert usage_totals(details) == {"document": 7}
