import gzip
import json
import logging
import os
import re
import struct
import uuid

from dataclasses import dataclass
from typing import Any, AsyncGenerator

import aiohttp

from realtime_agent.logger import setup_logger
from realtime_agent.tools import ToolContext
from realtime_agent.utils import PCMChunker, resample_pcm16

from .base import (
    DEFAULT_VOLCENGINE_VOICE,
    OPENAI_REALTIME_VOICES,
    ProviderAudioChunk,
    ProviderCapabilities,
    ProviderError,
    ProviderEvent,
    ProviderResponseFinished,
    ProviderResponseStarted,
    ProviderSessionStarted,
    ProviderSpeechStarted,
    ProviderSpeechStopped,
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    RealtimeProviderConnection,
)

logger = setup_logger(name=__name__, log_level=logging.INFO)

VOLCENGINE_HEADER_VERSION = 0x11
VOLCENGINE_FLAG_EVENT = 0x04

VOLCENGINE_MSG_TYPE_FULL_REQUEST = 0x01
VOLCENGINE_MSG_TYPE_AUDIO_REQUEST = 0x02
VOLCENGINE_MSG_TYPE_FULL_RESPONSE = 0x09
VOLCENGINE_MSG_TYPE_AUDIO_RESPONSE = 0x0B
VOLCENGINE_MSG_TYPE_ERROR = 0x0F

VOLCENGINE_SERIALIZATION_RAW = 0x00
VOLCENGINE_SERIALIZATION_JSON = 0x01
VOLCENGINE_COMPRESSION_NONE = 0x00
VOLCENGINE_COMPRESSION_GZIP = 0x01

VOLCENGINE_EVENT_START_CONNECTION = 1
VOLCENGINE_EVENT_FINISH_CONNECTION = 2
VOLCENGINE_EVENT_START_SESSION = 100
VOLCENGINE_EVENT_FINISH_SESSION = 102
VOLCENGINE_EVENT_TASK_REQUEST = 200
VOLCENGINE_EVENT_CHAT_TTS_TEXT = 500
VOLCENGINE_EVENT_CHAT_TEXT_QUERY = 501
VOLCENGINE_EVENT_CONVERSATION_CREATE = 510
VOLCENGINE_EVENT_CONVERSATION_UPDATE = 511
VOLCENGINE_EVENT_CLIENT_INTERRUPT = 515

VOLCENGINE_EVENT_CONNECTION_STARTED = 50
VOLCENGINE_EVENT_CONNECTION_FAILED = 51
VOLCENGINE_EVENT_CONNECTION_FINISHED = 52
VOLCENGINE_EVENT_SESSION_STARTED = 150
VOLCENGINE_EVENT_SESSION_FINISHED = 152
VOLCENGINE_EVENT_SESSION_FAILED = 153
VOLCENGINE_EVENT_TTS_SENTENCE_START = 350
VOLCENGINE_EVENT_TTS_RESPONSE = 352
VOLCENGINE_EVENT_TTS_ENDED = 359
VOLCENGINE_EVENT_ASR_INFO = 450
VOLCENGINE_EVENT_ASR_RESPONSE = 451
VOLCENGINE_EVENT_ASR_ENDED = 459
VOLCENGINE_EVENT_CHAT_RESPONSE = 550
VOLCENGINE_EVENT_CHAT_ENDED = 559
VOLCENGINE_EVENT_COMMON_ERROR = 599

VOLCENGINE_UPSTREAM_SAMPLE_RATE = 16000
VOLCENGINE_DOWNSTREAM_SAMPLE_RATE = 24000
VOLCENGINE_PCM_20MS_BYTES = 640
VOLCENGINE_RESOURCE_ID = "volc.speech.dialog"
VOLCENGINE_APP_KEY = "PlgvMymc7f3tQnJ6"
VOLCENGINE_REALTIME_WS_URL = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
VOLCENGINE_DEFAULT_DIALOG_MODEL = "1.2.1.1"


@dataclass(frozen=True)
class VolcengineFrame:
    event_id: int
    message_type: int
    payload: bytes
    session_id: str | None = None
    code: int | None = None


class VolcengineRealtimeProvider(RealtimeProviderConnection):
    provider_name = "volcengine"
    capabilities = ProviderCapabilities(
        supports_manual_responses=False,
        supports_tools=False,
        default_voice=DEFAULT_VOLCENGINE_VOICE,
    )

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.ws_url = VOLCENGINE_REALTIME_WS_URL

        self.app_id = os.getenv("VOLCENGINE_APP_ID")
        self.access_key = os.getenv("VOLCENGINE_ACCESS_KEY")
        if not self.app_id or not self.access_key:
            raise ValueError("VOLCENGINE_APP_ID and VOLCENGINE_ACCESS_KEY must be set for the volcengine provider")

        self.app_key = VOLCENGINE_APP_KEY
        self.resource_id = VOLCENGINE_RESOURCE_ID
        self.model = VOLCENGINE_DEFAULT_DIALOG_MODEL
        self.use_gzip = os.getenv("VOLCENGINE_USE_GZIP", "true").strip().lower() != "false"

        self.session = aiohttp.ClientSession()
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.session_id = str(uuid.uuid4())
        self.dialog_id = os.getenv("VOLCENGINE_DIALOG_ID", "").strip() or None
        self._audio_chunker = PCMChunker(VOLCENGINE_PCM_20MS_BYTES)
        self._current_question_id: str | None = None
        self._assistant_text_by_reply: dict[str, str] = {}
        self._reply_id_by_question_id: dict[str, str] = {}
        self._connection_started = False

    async def __aenter__(self) -> "VolcengineRealtimeProvider":
        headers = {
            "X-Api-App-ID": self.app_id,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-App-Key": self.app_key,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self.websocket = await self.session.ws_connect(self.ws_url, headers=headers)
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        await self.close()
        return False

    async def start_session(
        self,
        *,
        system_message: str | None,
        voice: str | None,
        tools: ToolContext | None,
        turn_detection: Any | None,
        dialog_id: str | None,
    ) -> ProviderSessionStarted:
        if tools:
            logger.warning("Volcengine realtime provider does not support OpenAI-style tool calling; tool declarations are ignored")

        if dialog_id:
            self.dialog_id = dialog_id

        selected_voice = (voice or "").strip() or self.capabilities.default_voice
        if selected_voice in OPENAI_REALTIME_VOICES:
            logger.warning("Replacing incompatible OpenAI voice '%s' with Volcengine default voice '%s'", selected_voice, self.capabilities.default_voice)
            selected_voice = self.capabilities.default_voice

        if not self._connection_started:
            await self._send_json_event(VOLCENGINE_EVENT_START_CONNECTION, {})
            await self._wait_for_control_event(
                success_event=VOLCENGINE_EVENT_CONNECTION_STARTED,
                failure_events={VOLCENGINE_EVENT_CONNECTION_FAILED, VOLCENGINE_EVENT_COMMON_ERROR},
            )
            self._connection_started = True

        payload = self._build_session_payload(
            system_message=system_message,
            voice=selected_voice,
        )
        await self._send_json_event(
            VOLCENGINE_EVENT_START_SESSION,
            payload,
            session_id=self.session_id,
        )

        frame = await self._wait_for_control_event(
            success_event=VOLCENGINE_EVENT_SESSION_STARTED,
            failure_events={VOLCENGINE_EVENT_SESSION_FAILED, VOLCENGINE_EVENT_COMMON_ERROR},
        )

        dialog_id = self._decode_json_payload(frame).get("dialog_id") or self.dialog_id
        self.dialog_id = dialog_id
        return ProviderSessionStarted(session_id=self.session_id, model=dialog_id)

    async def send_audio_data(self, audio_data: bytes) -> None:
        if self.websocket is None:
            raise RuntimeError("Volcengine websocket is not connected")

        resampled = resample_pcm16(
            audio_data,
            source_rate=VOLCENGINE_DOWNSTREAM_SAMPLE_RATE,
            target_rate=VOLCENGINE_UPSTREAM_SAMPLE_RATE,
        )
        for chunk in self._audio_chunker.push(resampled):
            await self._send_raw_event(
                VOLCENGINE_EVENT_TASK_REQUEST,
                chunk,
                session_id=self.session_id,
            )

    async def send_text(self, text: str) -> None:
        await self._send_json_event(
            VOLCENGINE_EVENT_CHAT_TEXT_QUERY,
            {"content": text},
            session_id=self.session_id,
        )

    async def send_tts_text(self, text: str) -> None:
        await self._send_json_event(
            VOLCENGINE_EVENT_CHAT_TTS_TEXT,
            {
                "start": True,
                "content": text,
                "end": True,
            },
            session_id=self.session_id,
        )

    async def upsert_conversation_turn(
        self,
        *,
        question_id: str | None,
        user_text: str,
        assistant_text: str,
    ) -> None:
        reply_id = self._reply_id_by_question_id.get(question_id or "") if question_id else None
        if reply_id:
            await self._send_json_event(
                VOLCENGINE_EVENT_CONVERSATION_UPDATE,
                {"items": [{"item_id": reply_id, "text": assistant_text}]},
                session_id=self.session_id,
            )
            logger.info("Updated Volcengine conversation reply %s for question %s", reply_id, question_id)
            return

        if not user_text or not assistant_text:
            logger.warning("Skipping Volcengine conversation upsert due to missing text")
            return

        await self._send_json_event(
            VOLCENGINE_EVENT_CONVERSATION_CREATE,
            {
                "items": [
                    {"role": "user", "text": user_text},
                    {"role": "assistant", "text": assistant_text},
                ]
            },
            session_id=self.session_id,
        )
        logger.info("Appended Volcengine conversation turn for question %s", question_id)

    async def request_response(self) -> None:
        return None

    async def send_tool_output(self, call_id: str, output: str) -> None:
        raise NotImplementedError("Volcengine realtime provider does not support tool outputs")

    async def interrupt_response(self) -> None:
        await self._send_json_event(
            VOLCENGINE_EVENT_CLIENT_INTERRUPT,
            {},
            session_id=self.session_id,
        )

    async def listen(self) -> AsyncGenerator[ProviderEvent, None]:
        while True:
            frame = await self._receive_frame()
            event = self._map_frame_to_event(frame)
            if event is not None:
                yield event

    async def close(self) -> None:
        try:
            if self.websocket is not None and not self.websocket.closed:
                if self._connection_started:
                    await self._send_json_event(
                        VOLCENGINE_EVENT_FINISH_SESSION,
                        {},
                        session_id=self.session_id,
                    )
                    await self._send_json_event(VOLCENGINE_EVENT_FINISH_CONNECTION, {})
                await self.websocket.close()
        finally:
            self.websocket = None
            await self.session.close()

    async def _wait_for_control_event(self, *, success_event: int, failure_events: set[int]) -> VolcengineFrame:
        while True:
            frame = await self._receive_frame()
            if frame.event_id == success_event:
                return frame
            if frame.event_id in failure_events:
                payload = self._decode_json_payload(frame)
                error_message = payload.get("message") or payload.get("error") or "Volcengine realtime request failed"
                error_code = str(payload.get("status_code") or frame.code or "") or None
                raise RuntimeError(f"{error_message} ({error_code})" if error_code else error_message)

    def _build_session_payload(self, *, system_message: str | None, voice: str) -> dict[str, Any]:
        dialog_extra: dict[str, Any] = {
            "input_mod": "keep_alive",
        }
        if self.model:
            dialog_extra["model"] = self.model

        dialog_payload = {
            "bot_name": os.getenv("VOLCENGINE_BOT_NAME", "agent"),
            "dialog_id": self.dialog_id or "",
            "extra": dialog_extra,
            "speaking_style": "专业冷静沉稳耐心细致"
        }
        if system_message:
            prompt_field = self._get_system_prompt_field()
            dialog_payload[prompt_field] = system_message
            logger.info("Applying Volcengine system prompt via dialog.%s", prompt_field)

        return {
            "dialog": dialog_payload,
            "asr": {
                "audio_info": {
                    "format": "pcm",
                    "sample_rate": VOLCENGINE_UPSTREAM_SAMPLE_RATE,
                    "channel": 1,
                }
            },
            "tts": {
                "speaker": voice,
                "audio_config": {
                    "channel": 1,
                    "format": "pcm_s16le",
                    "sample_rate": VOLCENGINE_DOWNSTREAM_SAMPLE_RATE,
                },
            },
        }

    def _get_system_prompt_field(self) -> str:
        prompt_field_override = os.getenv("VOLCENGINE_PROMPT_FIELD", "").strip().lower()
        if prompt_field_override in {"system_role", "character_manifest"}:
            return prompt_field_override

        model_name = (self.model or "").strip().lower()
        if model_name == "2.2.0.0":
            return "character_manifest"
        if model_name == "1.2.1.1":
            return "system_role"
        if re.search(r"(^|[^a-z0-9])sc(?:2(?:\.0)?)?([^a-z0-9]|$)", model_name):
            return "character_manifest"

        return "system_role"

    async def _send_json_event(self, event_id: int, payload: dict[str, Any], session_id: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self._send_frame(
            message_type=VOLCENGINE_MSG_TYPE_FULL_REQUEST,
            serialization=VOLCENGINE_SERIALIZATION_JSON,
            compression=VOLCENGINE_COMPRESSION_GZIP if self.use_gzip else VOLCENGINE_COMPRESSION_NONE,
            event_id=event_id,
            payload=body,
            session_id=session_id,
        )

    async def _send_raw_event(self, event_id: int, payload: bytes, session_id: str | None = None) -> None:
        await self._send_frame(
            message_type=VOLCENGINE_MSG_TYPE_AUDIO_REQUEST,
            serialization=VOLCENGINE_SERIALIZATION_RAW,
            compression=VOLCENGINE_COMPRESSION_GZIP if self.use_gzip else VOLCENGINE_COMPRESSION_NONE,
            event_id=event_id,
            payload=payload,
            session_id=session_id,
        )

    async def _send_frame(
        self,
        *,
        message_type: int,
        serialization: int,
        compression: int,
        event_id: int,
        payload: bytes,
        session_id: str | None = None,
    ) -> None:
        if self.websocket is None:
            raise RuntimeError("Volcengine websocket is not connected")

        encoded_payload = gzip.compress(payload) if compression == VOLCENGINE_COMPRESSION_GZIP else payload

        header = bytes(
            [
                VOLCENGINE_HEADER_VERSION,
                (message_type << 4) | VOLCENGINE_FLAG_EVENT,
                (serialization << 4) | compression,
                0x00,
            ]
        )
        parts = [header, struct.pack(">I", event_id)]
        if session_id is not None:
            session_bytes = session_id.encode("utf-8")
            parts.extend([struct.pack(">I", len(session_bytes)), session_bytes])
        parts.extend([struct.pack(">I", len(encoded_payload)), encoded_payload])
        frame = b"".join(parts)
        if self.verbose:
            logger.info("-> volcengine event=%s bytes=%s", event_id, len(frame))
        await self.websocket.send_bytes(frame)

    async def _receive_frame(self) -> VolcengineFrame:
        if self.websocket is None:
            raise RuntimeError("Volcengine websocket is not connected")

        message = await self.websocket.receive()
        if message.type == aiohttp.WSMsgType.CLOSE:
            raise RuntimeError("Volcengine websocket closed")
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(str(self.websocket.exception()))
        if message.type != aiohttp.WSMsgType.BINARY:
            raise RuntimeError(f"Unexpected websocket message type: {message.type}")

        data = message.data
        if len(data) < 8:
            raise RuntimeError("Invalid Volcengine websocket frame")

        message_type = data[1] >> 4
        compression = data[2] & 0x0F
        index = 4

        if message_type == VOLCENGINE_MSG_TYPE_ERROR:
            code = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload_size = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload = data[index : index + payload_size]
            if compression == VOLCENGINE_COMPRESSION_GZIP:
                payload = gzip.decompress(payload)
            return VolcengineFrame(
                event_id=VOLCENGINE_EVENT_COMMON_ERROR,
                message_type=message_type,
                payload=payload,
                code=code,
            )

        event_id = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        session_id: str | None = None
        if event_id not in {
            VOLCENGINE_EVENT_CONNECTION_STARTED,
            VOLCENGINE_EVENT_CONNECTION_FAILED,
            VOLCENGINE_EVENT_CONNECTION_FINISHED,
        }:
            session_size = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            session_id = data[index : index + session_size].decode("utf-8")
            index += session_size

        payload_size = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        payload = data[index : index + payload_size]
        if compression == VOLCENGINE_COMPRESSION_GZIP:
            payload = gzip.decompress(payload)
        return VolcengineFrame(
            event_id=event_id,
            message_type=message_type,
            payload=payload,
            session_id=session_id,
        )

    def _map_frame_to_event(self, frame: VolcengineFrame) -> ProviderEvent | None:
        if frame.event_id in {
            VOLCENGINE_EVENT_CONNECTION_STARTED,
            VOLCENGINE_EVENT_CONNECTION_FINISHED,
            VOLCENGINE_EVENT_SESSION_FINISHED,
        }:
            return None

        if frame.event_id in {
            VOLCENGINE_EVENT_CONNECTION_FAILED,
            VOLCENGINE_EVENT_SESSION_FAILED,
            VOLCENGINE_EVENT_COMMON_ERROR,
        }:
            payload = self._decode_json_payload(frame)
            return ProviderError(
                message=payload.get("message") or payload.get("error") or "Volcengine realtime error",
                code=str(payload.get("status_code") or frame.code or "") or None,
            )

        if frame.event_id == VOLCENGINE_EVENT_SESSION_STARTED:
            payload = self._decode_json_payload(frame)
            self.dialog_id = payload.get("dialog_id") or self.dialog_id
            return ProviderSessionStarted(
                session_id=frame.session_id or self.session_id,
                model=self.dialog_id,
            )

        if frame.event_id == VOLCENGINE_EVENT_ASR_INFO:
            payload = self._decode_json_payload(frame)
            self._current_question_id = payload.get("question_id") or self._current_question_id
            return ProviderSpeechStarted(item_id=self._current_question_id)

        if frame.event_id == VOLCENGINE_EVENT_ASR_RESPONSE:
            payload = self._decode_json_payload(frame)
            results = payload.get("results") or []
            if not results:
                return None

            text = "".join(str(item.get("text", "")) for item in results)
            item_id = payload.get("question_id") or self._current_question_id or frame.session_id
            is_interim = any(bool(item.get("is_interim", False)) for item in results)
            if is_interim:
                return ProviderTranscriptDelta(role="user", text=text, item_id=item_id)
            return ProviderTranscriptDone(role="user", text=text, item_id=item_id)

        if frame.event_id == VOLCENGINE_EVENT_ASR_ENDED:
            return ProviderSpeechStopped(item_id=self._current_question_id)

        if frame.event_id == VOLCENGINE_EVENT_TTS_SENTENCE_START:
            payload = self._decode_json_payload(frame)
            question_id = payload.get("question_id") or self._current_question_id
            reply_id = payload.get("reply_id")
            if question_id and reply_id:
                self._reply_id_by_question_id[question_id] = reply_id
            return ProviderResponseStarted(response_id=reply_id)

        if frame.event_id == VOLCENGINE_EVENT_TTS_RESPONSE:
            return ProviderAudioChunk(audio=frame.payload)

        if frame.event_id == VOLCENGINE_EVENT_CHAT_RESPONSE:
            payload = self._decode_json_payload(frame)
            question_id = payload.get("question_id") or self._current_question_id
            reply_id = payload.get("reply_id") or frame.session_id
            if question_id and reply_id:
                self._reply_id_by_question_id[question_id] = reply_id
            text = str(payload.get("content", ""))
            self._assistant_text_by_reply[reply_id] = self._assistant_text_by_reply.get(reply_id, "") + text
            return ProviderTranscriptDelta(role="assistant", text=text, item_id=reply_id)

        if frame.event_id == VOLCENGINE_EVENT_CHAT_ENDED:
            payload = self._decode_json_payload(frame)
            reply_id = payload.get("reply_id") or frame.session_id
            text = self._assistant_text_by_reply.pop(reply_id, "")
            return ProviderTranscriptDone(role="assistant", text=text, item_id=reply_id)

        if frame.event_id == VOLCENGINE_EVENT_TTS_ENDED:
            payload = self._decode_json_payload(frame)
            return ProviderResponseFinished(response_id=payload.get("reply_id"))

        return None

    def _decode_json_payload(self, frame: VolcengineFrame) -> dict[str, Any]:
        if not frame.payload:
            return {}
        return json.loads(frame.payload.decode("utf-8"))