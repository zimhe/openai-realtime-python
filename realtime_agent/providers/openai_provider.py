import base64
import logging
import os
from builtins import anext
from typing import Any, AsyncGenerator

from realtime_agent.logger import setup_logger
from realtime_agent.realtime.connection import DEFAULT_VIRTUAL_MODEL, RealtimeApiConnection
from realtime_agent.realtime.struct import (
    ErrorMessage,
    FunctionCallOutputItemParam,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    ItemCreate,
    ItemInputAudioTranscriptionCompleted,
    ItemInputAudioTranscriptionDelta,
    ResponseAudioDelta,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,
    ResponseCancel,
    ResponseCreate,
    ResponseCreated,
    ResponseDone,
    ResponseFunctionCallArgumentsDone,
    SessionUpdate,
    SessionUpdated,
)
from realtime_agent.tools import ToolContext

from .base import (
    DEFAULT_OPENAI_VOICE,
    ProviderAudioChunk,
    ProviderCapabilities,
    ProviderError,
    ProviderEvent,
    ProviderResponseFinished,
    ProviderResponseStarted,
    ProviderSessionStarted,
    ProviderSpeechStarted,
    ProviderSpeechStopped,
    ProviderToolCall,
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    RealtimeProviderConnection,
)

logger = setup_logger(name=__name__, log_level=logging.INFO)


class OpenAIRealtimeProvider(RealtimeProviderConnection):
    provider_name = "openai"
    capabilities = ProviderCapabilities(
        supports_manual_responses=True,
        supports_tools=True,
        default_voice=DEFAULT_OPENAI_VOICE,
    )

    def __init__(self, *, verbose: bool = False):
        self.connection = RealtimeApiConnection(
            base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
            api_key=os.getenv("OPENAI_API_KEY"),
            verbose=verbose,
        )
        self._response_active = False

    async def __aenter__(self) -> "OpenAIRealtimeProvider":
        await self.connection.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return await self.connection.__aexit__(exc_type, exc_value, traceback)

    async def start_session(
        self,
        *,
        system_message: str | None,
        voice: str | None,
        tools: ToolContext | None,
        turn_detection: Any | None,
        dialog_id: str | None,
    ) -> ProviderSessionStarted:
        session_payload = {
            "type": "realtime",
            "model": os.environ.get("OPENAI_MODEL", DEFAULT_VIRTUAL_MODEL),
            "instructions": system_message,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                    },
                    "turn_detection": turn_detection,
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "voice": voice or self.capabilities.default_voice,
                },
            },
        }

        if tools:
            session_payload["tools"] = tools.model_description()
            session_payload["tool_choice"] = "auto"

        await self.connection.send_request(
            SessionUpdate(
                session=session_payload
            )
        )

        while True:
            message = await anext(self.connection.listen())
            if isinstance(message, SessionUpdated):
                return ProviderSessionStarted(
                    session_id=message.session.id,
                    model=message.session.model,
                )
            if isinstance(message, ErrorMessage):
                raise RuntimeError(message.error.message)

    async def send_audio_data(self, audio_data: bytes) -> None:
        await self.connection.send_audio_data(audio_data)

    async def send_text(self, text: str) -> None:
        await self.connection.send_text(text)

    async def request_response(self) -> None:
        await self.connection.send_request(ResponseCreate())

    async def send_tool_output(self, call_id: str, output: str) -> None:
        await self.connection.send_request(
            ItemCreate(
                item=FunctionCallOutputItemParam(
                    call_id=call_id,
                    output=output,
                )
            )
        )
        await self.request_response()

    async def interrupt_response(self) -> None:
        if not self._response_active:
            logger.info("Skipping response.cancel because no active response is tracked")
            return
        await self.connection.send_request(ResponseCancel())

    async def listen(self) -> AsyncGenerator[ProviderEvent, None]:
        async for message in self.connection.listen():
            if isinstance(message, ResponseAudioDelta):
                yield ProviderAudioChunk(
                    audio=base64.b64decode(message.delta),
                    item_id=message.item_id,
                )
            elif isinstance(message, ResponseAudioTranscriptDelta):
                yield ProviderTranscriptDelta(
                    role="assistant",
                    text=message.delta,
                    item_id=message.item_id,
                )
            elif isinstance(message, ResponseAudioTranscriptDone):
                yield ProviderTranscriptDone(
                    role="assistant",
                    text=message.transcript,
                    item_id=message.item_id,
                )
            elif isinstance(message, ItemInputAudioTranscriptionDelta):
                yield ProviderTranscriptDelta(
                    role="user",
                    text=message.delta,
                    item_id=message.item_id,
                )
            elif isinstance(message, ItemInputAudioTranscriptionCompleted):
                yield ProviderTranscriptDone(
                    role="user",
                    text=message.transcript,
                    item_id=message.item_id,
                )
            elif isinstance(message, InputAudioBufferSpeechStarted):
                yield ProviderSpeechStarted(item_id=message.item_id)
            elif isinstance(message, InputAudioBufferSpeechStopped):
                yield ProviderSpeechStopped(item_id=message.item_id)
            elif isinstance(message, ResponseCreated):
                self._response_active = True
                yield ProviderResponseStarted(response_id=message.response.id)
            elif isinstance(message, ResponseDone):
                self._response_active = False
                yield ProviderResponseFinished(response_id=message.response.id)
            elif isinstance(message, ResponseFunctionCallArgumentsDone):
                yield ProviderToolCall(
                    name=message.name,
                    arguments=message.arguments,
                    call_id=message.call_id,
                )
            elif isinstance(message, ErrorMessage):
                if message.error.code == "response_cancel_not_active":
                    logger.info("Ignoring benign cancel error from server: %s", message.error.message)
                    self._response_active = False
                    continue
                yield ProviderError(
                    message=message.error.message,
                    code=message.error.code,
                )

    async def close(self) -> None:
        await self.connection.close()