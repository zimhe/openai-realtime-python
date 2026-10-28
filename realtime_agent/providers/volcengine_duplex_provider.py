"""Seeduplex 3.0 adapter for the Realtime provider interface.

The protocol is JSON text over WebSocket. Agora supplies PCM16 mono at 24 kHz;
Seeduplex expects 16 kHz input and returns PCM16 at 24 kHz.
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator

import aiohttp

from realtime_agent.logger import setup_logger
from realtime_agent.tools import ToolContext
from realtime_agent.utils import PCMChunker, resample_pcm16

from .base import (
    DEFAULT_VOLCENGINE_VOICE,
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

DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
MODEL = "1.2.6.1"
INPUT_RATE = 16000
OUTPUT_RATE = 24000
INPUT_CHUNK_BYTES = 640  # 20 ms, mono PCM16


class VolcengineDuplexRealtimeProvider(RealtimeProviderConnection):
    provider_name = "volcengine_duplex"
    capabilities = ProviderCapabilities(
        supports_manual_responses=False,
        supports_tools=True,
        default_voice=DEFAULT_VOLCENGINE_VOICE,
    )

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.api_key = os.getenv("VOLCENGINE_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("VOLCENGINE_API_KEY is required for volcengine_duplex")
        self.url = os.getenv("VOLCENGINE_DUPLEX_WS_URL", DEFAULT_URL)
        self.session: aiohttp.ClientSession | None = None
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.session_id: str | None = None
        self._chunker = PCMChunker(INPUT_CHUNK_BYTES)
        self._response_active = False
        self._pending_tool_ids: set[str] = set()
        self._tool_outputs: dict[str, str] = {}
        self._next_audio_send = 0.0

    async def __aenter__(self) -> "VolcengineDuplexRealtimeProvider":
        self.session = aiohttp.ClientSession()
        try:
            self.websocket = await self.session.ws_connect(
                self.url,
                headers={"X-Api-Key": self.api_key},
            )
        except BaseException:
            await self.session.close()
            self.session = None
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        await self.close()
        return False

    async def _send(self, event_type: str, **fields: Any) -> None:
        if self.websocket is None or self.websocket.closed:
            raise ConnectionError("Seeduplex WebSocket is not connected")
        event = {"type": event_type, "event_id": uuid.uuid4().hex, **fields}
        await self.websocket.send_str(json.dumps(event, ensure_ascii=False))

    async def _receive(self) -> dict[str, Any]:
        if self.websocket is None:
            raise ConnectionError("Seeduplex WebSocket is not connected")
        message = await self.websocket.receive()
        if message.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(message.data)
            if not isinstance(event, dict):
                raise ValueError("Seeduplex event must be a JSON object")
            return event
        if message.type == aiohttp.WSMsgType.ERROR:
            raise ConnectionError("Seeduplex WebSocket failed") from self.websocket.exception()
        raise ConnectionError(f"Seeduplex WebSocket closed: {message.type}")

    async def start_session(
        self,
        *,
        system_message: str | None,
        voice: str | None,
        tools: ToolContext | None,
        turn_detection: Any | None,
        dialog_id: str | None,
    ) -> ProviderSessionStarted:
        # dialog_id is an Agora channel name, not a Seeduplex session ID.
        # Seeduplex creates its own session ID; do not send the channel name as one.
        del turn_detection, dialog_id
        session_payload: dict[str, Any] = {
            "model": MODEL,
            "instructions": system_message or "",
            "audio": {
                "input": {"format": {"type": "pcm", "rate": INPUT_RATE}},
                "output": {
                    "format": {"type": "pcm_s16le", "rate": OUTPUT_RATE},
                    "voice": voice or self.capabilities.default_voice,
                },
            },
        }
        if tools:
            session_payload["tools"] = tools.model_description()
        extension: dict[str, Any] = {"asr": {}, "tts": {}, "dialog": {}}
        search_key = os.getenv("VOLCENGINE_WEBSEARCH_API_KEY", "").strip()
        if search_key:
            extension["dialog"]["extra"] = {
                "enable_volc_websearch": True,
                "volc_websearch_type": os.getenv("VOLCENGINE_WEBSEARCH_TYPE", "web_global_api"),
                "volc_websearch_api_key": search_key,
            }
        await self._send(
            "session.create",
            session=session_payload,
            extension=extension,
        )
        while True:
            event = await self._receive()
            event_type = event.get("type")
            if event_type == "session.created":
                session = event.get("session") or {}
                self.session_id = str(session.get("id") or "")
                return ProviderSessionStarted(session_id=self.session_id, model=MODEL)
            if event_type == "error":
                raise RuntimeError(self._error_message(event))

    async def send_audio_data(self, audio_data: bytes) -> None:
        pcm16 = resample_pcm16(
            audio_data, source_rate=OUTPUT_RATE, target_rate=INPUT_RATE
        )
        for chunk in self._chunker.push(pcm16):
            now = time.monotonic()
            if self._next_audio_send > now:
                await asyncio.sleep(self._next_audio_send - now)
            await self._send(
                "input_audio_buffer.append",
                audio=base64.b64encode(chunk).decode("ascii"),
            )
            self._next_audio_send = max(self._next_audio_send, time.monotonic()) + 0.020

    async def mute_input(self) -> None:
        self._chunker.clear()
        self._next_audio_send = 0.0
        await self._send("input_audio_mute.commit")

    async def unmute_input(self) -> None:
        await self._send("input_audio_unmute.commit")

    async def commit_input(self) -> None:
        await self._send("input_audio_buffer.commit")

    async def update_instructions(self, instructions: str) -> None:
        await self._send("session.update", session={"instructions": instructions})

    async def send_text(self, text: str) -> None:
        # The documented speech_text_buffer.commit is a greeting/TTS event,
        # not a general text question. Do not misroute chat messages to it.
        logger.warning("Seeduplex does not document a general text-query event; ignoring text input")

    async def request_response(self) -> None:
        # Unlike OpenAI Realtime, this API does not document response.create.
        return None

    async def send_tool_output(self, call_id: str, output: str) -> None:
        self._tool_outputs[call_id] = output
        if not self._pending_tool_ids or not self._pending_tool_ids.issubset(self._tool_outputs):
            return
        items = [
            {
                "call_id": pending_id,
                "role": "tool",
                "content": [{"type": "input_text", "text": self._tool_outputs[pending_id]}],
            }
            for pending_id in sorted(self._pending_tool_ids)
        ]
        await self._send("conversation.item.create", items=items)
        self._pending_tool_ids.clear()
        self._tool_outputs.clear()

    async def interrupt_response(self) -> None:
        if self._response_active:
            await self._send("response.cancel")

    @staticmethod
    def _error_message(event: dict[str, Any]) -> str:
        error = event.get("error") or {}
        if isinstance(error, dict):
            return str(error.get("message") or event.get("message") or error)
        return str(error or event.get("message") or "Unknown Seeduplex error")

    async def listen(self) -> AsyncGenerator[ProviderEvent, None]:
        while True:
            try:
                event = await self._receive()
            except ConnectionError as exc:
                yield ProviderError(message=str(exc), code="duplex_disconnected")
                return
            event_type = event.get("type")
            item_id = event.get("item_id")
            if event_type == "conversation.item.input_audio_transcription.started":
                yield ProviderSpeechStarted(item_id=item_id)
            elif event_type == "conversation.item.input_audio_transcription.delta":
                yield ProviderTranscriptDelta(
                    role="user", text=str(event.get("delta") or ""), item_id=item_id
                )
            elif event_type == "conversation.item.input_audio_transcription.completed":
                yield ProviderSpeechStopped(item_id=item_id)
                yield ProviderTranscriptDone(
                    role="user",
                    text=str(event.get("transcript") or event.get("text") or ""),
                    item_id=item_id,
                )
            elif event_type == "response.output_text.delta":
                yield ProviderTranscriptDelta(
                    role="assistant", text=str(event.get("delta") or ""), item_id=item_id
                )
            elif event_type == "response.output_text.done":
                yield ProviderTranscriptDone(
                    role="assistant",
                    text=str(event.get("text") or event.get("transcript") or ""),
                    item_id=item_id,
                )
            elif event_type == "response.output_audio.started":
                self._response_active = True
                yield ProviderResponseStarted(response_id=event.get("response_id"))
            elif event_type == "response.output_audio.delta":
                delta = event.get("delta") or ""
                if delta:
                    try:
                        yield ProviderAudioChunk(
                            audio=base64.b64decode(delta, validate=True), item_id=item_id
                        )
                    except ValueError:
                        yield ProviderError(message="Invalid Seeduplex audio delta", code="audio_decode")
            elif event_type == "response.function_call_arguments.done":
                calls = event.get("items") or []
                self._pending_tool_ids = {
                    str(item["call_id"]) for item in calls if item.get("call_id")
                }
                self._tool_outputs.clear()
                for item in calls:
                    if item.get("call_id") and item.get("name"):
                        yield ProviderToolCall(
                            name=str(item["name"]),
                            arguments=str(item.get("arguments") or "{}"),
                            call_id=str(item["call_id"]),
                        )
            elif event_type in ("response.done", "response.canceled"):
                self._response_active = False
                yield ProviderResponseFinished(response_id=event.get("response_id"))
            elif event_type == "error":
                error = event.get("error") or {}
                code = error.get("code") if isinstance(error, dict) else None
                yield ProviderError(message=self._error_message(event), code=code)

    async def close(self) -> None:
        try:
            if self.websocket is not None and not self.websocket.closed:
                await self._send("session.close")
                await self.websocket.close()
        finally:
            self.websocket = None
            if self.session is not None:
                await self.session.close()
                self.session = None
