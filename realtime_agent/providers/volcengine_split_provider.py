import asyncio
import gzip
import json
import logging
import os
import struct
import uuid
import re

from dataclasses import dataclass
from typing import Any, AsyncGenerator

import aiohttp
from openai import OpenAI

from realtime_agent.logger import setup_logger
from realtime_agent.tools import LocalToolCallExecuted, ToolContext
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
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    RealtimeProviderConnection,
)

logger = setup_logger(name=__name__, log_level=logging.INFO)

DEFAULT_SPLIT_TTS_VOICE = "zh_female_vv_uranus_bigtts"

ASR_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
ASR_RESOURCE_ID_DEFAULT = "volc.seedasr.sauc.duration"
ASR_UPSTREAM_SAMPLE_RATE = 16000
ASR_CHUNK_BYTES = 640

TTS_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
TTS_RESOURCE_ID_DEFAULT = "seed-tts-2.0"
TTS_SAMPLE_RATE = 24000

VOLCENGINE_HEADER_VERSION = 0x11

ASR_MSG_TYPE_FULL_REQUEST = 0x01
ASR_MSG_TYPE_AUDIO_REQUEST = 0x02
ASR_MSG_TYPE_FULL_RESPONSE = 0x09
ASR_MSG_TYPE_ERROR = 0x0F

SERIALIZATION_RAW = 0x00
SERIALIZATION_JSON = 0x01
COMPRESSION_NONE = 0x00
COMPRESSION_GZIP = 0x01

TTS_FLAG_EVENT = 0x04
TTS_MSG_TYPE_FULL_REQUEST = 0x01
TTS_MSG_TYPE_AUDIO_RESPONSE = 0x0B
TTS_MSG_TYPE_FULL_RESPONSE = 0x09
TTS_MSG_TYPE_ERROR = 0x0F

TTS_EVENT_START_CONNECTION = 1
TTS_EVENT_FINISH_CONNECTION = 2
TTS_EVENT_START_SESSION = 100
TTS_EVENT_CANCEL_SESSION = 101
TTS_EVENT_FINISH_SESSION = 102
TTS_EVENT_TASK_REQUEST = 200

TTS_EVENT_CONNECTION_STARTED = 50
TTS_EVENT_CONNECTION_FAILED = 51
TTS_EVENT_CONNECTION_FINISHED = 52
TTS_EVENT_SESSION_STARTED = 150
TTS_EVENT_SESSION_CANCELED = 151
TTS_EVENT_SESSION_FINISHED = 152
TTS_EVENT_SESSION_FAILED = 153
TTS_EVENT_SENTENCE_START = 350
TTS_EVENT_SENTENCE_END = 351
TTS_EVENT_RESPONSE = 352

NATIVE_TOOL_OVERLAP = {
    "get_web_search_results",
    "interpert_image",
}


@dataclass(frozen=True)
class AsrFrame:
    message_type: int
    flags: int
    payload: bytes
    sequence: int | None = None
    code: int | None = None


@dataclass(frozen=True)
class TtsFrame:
    event_id: int
    message_type: int
    payload: bytes
    session_id: str | None = None
    code: int | None = None


class VolcengineSplitRealtimeProvider(RealtimeProviderConnection):
    provider_name = "volcengine_split"
    capabilities = ProviderCapabilities(
        supports_manual_responses=True,
        supports_tools=True,
        default_voice=DEFAULT_SPLIT_TTS_VOICE,
    )

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.session = aiohttp.ClientSession()
        self.events: asyncio.Queue[ProviderEvent] = asyncio.Queue()

        self.api_key = os.getenv("VOLCENGINE_API_KEY", "").strip() or None
        self.app_id = os.getenv("VOLCENGINE_APP_ID", "").strip() or None
        self.access_key = os.getenv("VOLCENGINE_ACCESS_KEY", "").strip() or None

        if not self.api_key and not (self.app_id and self.access_key):
            raise ValueError(
                "VOLCENGINE_API_KEY or VOLCENGINE_APP_ID/VOLCENGINE_ACCESS_KEY must be set "
                "for the volcengine_split provider"
            )

        ark_api_key = os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_ARK_API_KEY")
        if not ark_api_key:
            raise ValueError("ARK_API_KEY or VOLCENGINE_ARK_API_KEY must be set for volcengine_split")

        self.ark_client = OpenAI(
            api_key=ark_api_key,
            base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        )
        self.supports_responses_api = hasattr(self.ark_client, "responses")
        self.llm_model = os.getenv(
            "VOLCENGINE_SPLIT_MODEL",
            os.getenv("VOLCENGINE_FUNCTION_CALL_MODEL", "doubao-seed-2-0-mini-260428"),
        )
        self.service_tier = os.getenv("VOLCENGINE_SPLIT_SERVICE_TIER", "").strip() or None
        self.web_search_enabled = os.getenv("VOLCENGINE_SPLIT_ENABLE_WEB_SEARCH", "true").strip().lower() != "false"
        self.max_tool_calls = max(1, int(os.getenv("VOLCENGINE_SPLIT_MAX_TOOL_CALLS", "6")))
        self.parallel_tool_calls = os.getenv("VOLCENGINE_PARALLEL_TOOL_CALLS", "false").strip().lower() == "true"

        if not self.supports_responses_api:
            logger.warning(
                "OpenAI SDK on this runtime does not expose client.responses; "
                "falling back to chat.completions for the split Volcengine provider"
            )

        self.asr_ws: aiohttp.ClientWebSocketResponse | None = None
        self.asr_request_id = str(uuid.uuid4())
        self.asr_resource_id = os.getenv("VOLCENGINE_ASR_RESOURCE_ID", ASR_RESOURCE_ID_DEFAULT)
        self._audio_chunker = PCMChunker(ASR_CHUNK_BYTES)
        self._asr_receive_task: asyncio.Task[None] | None = None
        self._asr_current_item_id: str | None = None
        self._asr_current_interim_text = ""
        self._asr_current_text = ""

        self.tts_ws: aiohttp.ClientWebSocketResponse | None = None
        self.tts_resource_id = os.getenv(
            "VOLCENGINE_SPLIT_TTS_RESOURCE_ID",
            os.getenv("VOLCENGINE_TTS_RESOURCE_ID", TTS_RESOURCE_ID_DEFAULT),
        )
        self.tts_voice = os.getenv("VOLCENGINE_SPLIT_TTS_VOICE", DEFAULT_SPLIT_TTS_VOICE).strip() or DEFAULT_SPLIT_TTS_VOICE
        self._tts_connection_started = False
        self._tts_session_id: str | None = None
        self._tts_done_event: asyncio.Event | None = None
        self._tts_cancel_requested = False

        self.system_message: str | None = None
        self.dialog_id: str | None = None
        self.tools: ToolContext | None = None
        self.session_id = str(uuid.uuid4())

        self._pending_user_text: str | None = None
        self._pending_user_item_id: str | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._tts_worker_task: asyncio.Task[None] | None = None
        self._last_response_id: str | None = None
        self._conversation_messages: list[dict[str, Any]] = []

    async def __aenter__(self) -> "VolcengineSplitRealtimeProvider":
        self.asr_ws = await self.session.ws_connect(
            ASR_WS_URL,
            headers=self._build_asr_headers(self.asr_request_id),
        )
        self._asr_receive_task = asyncio.create_task(self._asr_receive_loop())
        await self._start_tts_connection()
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
        del turn_detection

        self.system_message = system_message
        self.tools = tools
        if dialog_id:
            self.dialog_id = dialog_id

        self._conversation_messages = []
        if self.system_message:
            self._conversation_messages.append({"role": "system", "content": self.system_message})

        selected_voice = (voice or "").strip() or self.capabilities.default_voice
        self.tts_voice = self._normalize_tts_voice(selected_voice)

        await self._start_asr_session()
        return ProviderSessionStarted(session_id=self.session_id, model=self.llm_model)

    async def send_audio_data(self, audio_data: bytes) -> None:
        if self.asr_ws is None:
            raise RuntimeError("Volcengine split ASR websocket is not connected")

        resampled = resample_pcm16(
            audio_data,
            source_rate=TTS_SAMPLE_RATE,
            target_rate=ASR_UPSTREAM_SAMPLE_RATE,
        )
        for chunk in self._audio_chunker.push(resampled):
            await self._send_asr_audio(chunk)

    async def send_text(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return

        self._pending_user_text = normalized
        self._pending_user_item_id = uuid.uuid4().hex

    async def request_response(self) -> None:
        if not self._pending_user_text:
            logger.info("Skipping split response request because no pending user text is buffered")
            return

        if self._response_task is not None and not self._response_task.done():
            logger.info("Skipping split response request because a response is already running")
            return

        user_text = self._pending_user_text
        item_id = self._pending_user_item_id or uuid.uuid4().hex
        self._pending_user_text = None
        self._pending_user_item_id = None

        self._response_task = asyncio.create_task(self._run_response(user_text, item_id))

    async def send_tool_output(self, call_id: str, output: str) -> None:
        del call_id, output
        raise NotImplementedError("volcengine_split handles tool execution internally")

    async def interrupt_response(self) -> None:
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()

        self._tts_cancel_requested = True
        if self.tts_ws is not None and self._tts_session_id is not None:
            try:
                await self._send_tts_json_event(
                    TTS_EVENT_CANCEL_SESSION,
                    {"event": TTS_EVENT_CANCEL_SESSION},
                    session_id=self._tts_session_id,
                )
            except Exception as exc:
                logger.info("Ignoring TTS cancel failure: %s", exc)

    async def listen(self) -> AsyncGenerator[ProviderEvent, None]:
        while True:
            yield await self.events.get()

    async def close(self) -> None:
        try:
            if self._response_task is not None and not self._response_task.done():
                self._response_task.cancel()
            if self._tts_worker_task is not None and not self._tts_worker_task.done():
                self._tts_worker_task.cancel()
            if self._asr_receive_task is not None and not self._asr_receive_task.done():
                self._asr_receive_task.cancel()
            await self._reset_tts_connection()
            if self.asr_ws is not None and not self.asr_ws.closed:
                try:
                    await self._send_asr_audio(b"", is_final=True)
                except Exception:
                    pass
                await self.asr_ws.close()
        finally:
            self.tts_ws = None
            self.asr_ws = None
            await self.session.close()

    async def _run_response(self, user_text: str, item_id: str) -> None:
        response_started = False
        response_event_id = uuid.uuid4().hex
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        try:
            await self.events.put(ProviderResponseStarted(response_id=response_event_id))
            response_started = True
            self._tts_worker_task = asyncio.create_task(self._stream_tts_worker(tts_queue))
            final_text, response_id = await self._generate_response_text(user_text, response_event_id, tts_queue)
            self._last_response_id = response_id or self._last_response_id
            await tts_queue.put(None)
            if self._tts_worker_task is not None:
                await self._tts_worker_task
            if not final_text:
                return

            await self.events.put(
                ProviderTranscriptDone(role="assistant", text=final_text, item_id=response_event_id)
            )
        except asyncio.CancelledError:
            logger.info("Cancelled split response task for user item %s", item_id)
            try:
                await tts_queue.put(None)
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.warning("Split response generation failed: %s", exc)
            await self.events.put(ProviderError(message=str(exc), code="split_response_error"))
        finally:
            if self._tts_worker_task is not None and not self._tts_worker_task.done():
                self._tts_worker_task.cancel()
            self._tts_worker_task = None
            if response_started:
                await self.events.put(ProviderResponseFinished(response_id=response_event_id))

    async def _generate_response_text(
        self,
        user_text: str,
        response_event_id: str,
        tts_queue: asyncio.Queue[str | None],
    ) -> tuple[str | None, str | None]:
        self._conversation_messages.append({"role": "user", "content": user_text})

        final_text: str | None = None
        response_id: str | None = None
        loop = asyncio.get_running_loop()

        for _ in range(self.max_tool_calls):
            stream_result = await asyncio.to_thread(
                self._stream_chat_completion_turn,
                list(self._conversation_messages),
                loop,
                response_event_id,
                tts_queue,
            )
            response_id = stream_result["response_id"] or response_id
            assistant_text = stream_result["text"]
            tool_calls = stream_result["tool_calls"]

            if not tool_calls:
                final_text = assistant_text
                self._conversation_messages.append({"role": "assistant", "content": assistant_text or ""})
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or "",
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
            self._conversation_messages.append(assistant_message)

            if self.tools is None:
                logger.warning("Model requested function calls but no tool context is configured")
                final_text = assistant_text
                break

            for tool_call in tool_calls:
                result = await self.tools.execute_tool(tool_call["name"], tool_call["arguments"])
                if isinstance(result, LocalToolCallExecuted):
                    output = result.json_encoded_output
                elif result is None:
                    output = json.dumps(
                        {"status": "error", "message": f"Tool '{tool_call['name']}' is not registered."},
                        ensure_ascii=False,
                    )
                else:
                    output = json.dumps(
                        {
                            "status": "error",
                            "message": f"Tool '{tool_call['name']}' is not available for local execution in the split voice chain.",
                        },
                        ensure_ascii=False,
                    )

                self._conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": output,
                    }
                )

        return final_text, response_id

    def _build_response_tools(self) -> list[dict[str, Any]]:
        declared_tools: list[dict[str, Any]] = []
        if self.web_search_enabled:
            declared_tools.append({"type": "web_search"})

        if self.tools is None:
            return declared_tools

        for tool in self.tools.model_description():
            if tool.get("name") in NATIVE_TOOL_OVERLAP:
                continue
            if tool.get("type") != "function":
                continue
            declared_tools.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return declared_tools

    def _build_chat_completion_tools(self) -> list[dict[str, Any]]:
        if self.tools is None:
            return []

        declared_tools: list[dict[str, Any]] = []
        for tool in self.tools.model_description():
            if tool.get("type") != "function":
                continue
            declared_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return declared_tools

    def _stream_chat_completion_turn(
        self,
        messages: list[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        response_event_id: str,
        tts_queue: asyncio.Queue[str | None],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "tools": self._build_chat_completion_tools(),
            "parallel_tool_calls": self.parallel_tool_calls,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier

        stream = self.ark_client.chat.completions.create(**kwargs)
        response_id: str | None = None
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        sentence_buffer = ""

        for chunk in stream:
            if response_id is None and getattr(chunk, "id", None):
                response_id = str(chunk.id)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                text = str(content)
                content_parts.append(text)
                sentence_buffer += text
                asyncio.run_coroutine_threadsafe(
                    self.events.put(
                        ProviderTranscriptDelta(
                            role="assistant",
                            text=text,
                            item_id=response_event_id,
                        )
                    ),
                    loop,
                ).result()
                sentence_buffer = self._flush_sentence_chunks(sentence_buffer, tts_queue, loop)

            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_call, "index", 0) or 0)
                entry = tool_calls_by_index.setdefault(
                    index,
                    {
                        "id": getattr(tool_call, "id", None) or uuid.uuid4().hex,
                        "name": "",
                        "arguments": "",
                    },
                )

                function = getattr(tool_call, "function", None)
                if function is not None:
                    function_name = getattr(function, "name", None)
                    function_arguments = getattr(function, "arguments", None)
                    if function_name:
                        entry["name"] = str(function_name)
                    if function_arguments:
                        entry["arguments"] += str(function_arguments)

        if sentence_buffer.strip():
            asyncio.run_coroutine_threadsafe(tts_queue.put(sentence_buffer.strip()), loop).result()

        ordered_tool_calls = [tool_calls_by_index[key] for key in sorted(tool_calls_by_index.keys())]
        return {
            "response_id": response_id,
            "text": "".join(content_parts).strip() or None,
            "tool_calls": ordered_tool_calls,
        }

    def _flush_sentence_chunks(
        self,
        buffer: str,
        tts_queue: asyncio.Queue[str | None],
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        sentence_end_pattern = re.compile(r"([。！？!?\n])")
        start_index = 0

        for match in sentence_end_pattern.finditer(buffer):
            end_index = match.end()
            chunk = buffer[start_index:end_index].strip()
            if chunk:
                asyncio.run_coroutine_threadsafe(tts_queue.put(chunk), loop).result()
            start_index = end_index

        return buffer[start_index:]

    async def _stream_tts_worker(self, tts_queue: asyncio.Queue[str | None]) -> None:
        while True:
            chunk = await tts_queue.get()
            if chunk is None:
                return
            await self._synthesize_text(chunk)

    def _extract_output_text(self, response: Any) -> str | None:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip() or None

        for item in self._response_output_items(response):
            if item.get("type") != "message":
                continue
            content = item.get("content") or []
            text_parts = [
                str(content_item.get("text", ""))
                for content_item in content
                if content_item.get("type") in {"output_text", "text"}
            ]
            text = "".join(text_parts).strip()
            if text:
                return text
        return None

    def _extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for item in self._response_output_items(response):
            if item.get("type") != "function_call":
                continue
            tool_calls.append(item)
        return tool_calls

    def _response_output_items(self, response: Any) -> list[dict[str, Any]]:
        raw_items = getattr(response, "output", None) or []
        items: list[dict[str, Any]] = []
        for item in raw_items:
            if hasattr(item, "model_dump"):
                items.append(item.model_dump())
            elif isinstance(item, dict):
                items.append(item)
        return items

    async def _start_asr_session(self) -> None:
        if self.asr_ws is None:
            raise RuntimeError("Volcengine split ASR websocket is not connected")

        payload = {
            "user": {
                "uid": self.dialog_id or self.session_id,
            },
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": ASR_UPSTREAM_SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_nonstream": True,
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "result_type": "single",
                "end_window_size": int(os.getenv("VOLCENGINE_ASR_END_WINDOW_MS", "800")),
            },
        }
        await self._send_asr_full_request(payload)

    async def _asr_receive_loop(self) -> None:
        try:
            while True:
                frame = await self._receive_asr_frame()
                await self._handle_asr_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Split ASR receive loop stopped: %s", exc)
            await self.events.put(ProviderError(message=str(exc), code="split_asr_error"))

    async def _handle_asr_frame(self, frame: AsrFrame) -> None:
        if frame.message_type == ASR_MSG_TYPE_ERROR:
            payload = self._decode_json_bytes(frame.payload)
            await self.events.put(
                ProviderError(
                    message=payload.get("error") or "Volcengine split ASR error",
                    code=str(frame.code) if frame.code is not None else None,
                )
            )
            return

        payload = self._decode_json_bytes(frame.payload)
        result = payload.get("result") or {}
        utterances = result.get("utterances") or []
        if not utterances:
            return

        for utterance in utterances:
            text = str(utterance.get("text", "")).strip()
            if not text:
                continue
            is_definite = bool(utterance.get("definite", False))

            if self._asr_current_item_id is None:
                self._asr_current_item_id = uuid.uuid4().hex

            if not is_definite:
                if not self._asr_current_interim_text:
                    await self.events.put(ProviderSpeechStarted(item_id=self._asr_current_item_id))
                self._asr_current_interim_text = text
                await self.events.put(
                    ProviderTranscriptDelta(
                        role="user",
                        text=text,
                        item_id=self._asr_current_item_id,
                    )
                )
                continue

            item_id = self._asr_current_item_id or uuid.uuid4().hex
            if self._asr_current_interim_text and self._asr_current_interim_text != text:
                await self.events.put(
                    ProviderTranscriptDelta(role="user", text=text, item_id=item_id)
                )

            self._pending_user_text = text
            self._pending_user_item_id = item_id
            await self.events.put(ProviderSpeechStopped(item_id=item_id))
            await self.events.put(
                ProviderTranscriptDone(role="user", text=text, item_id=item_id)
            )
            self._asr_current_item_id = None
            self._asr_current_interim_text = ""

    async def _send_asr_full_request(self, payload: dict[str, Any]) -> None:
        if self.asr_ws is None:
            raise RuntimeError("Volcengine split ASR websocket is not connected")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        compressed = gzip.compress(body)
        header = bytes(
            [
                VOLCENGINE_HEADER_VERSION,
                ASR_MSG_TYPE_FULL_REQUEST << 4,
                (SERIALIZATION_JSON << 4) | COMPRESSION_GZIP,
                0x00,
            ]
        )
        frame = b"".join([header, struct.pack(">I", len(compressed)), compressed])
        await self.asr_ws.send_bytes(frame)

    async def _send_asr_audio(self, audio_chunk: bytes, *, is_final: bool = False) -> None:
        if self.asr_ws is None:
            raise RuntimeError("Volcengine split ASR websocket is not connected")

        encoded = gzip.compress(audio_chunk) if audio_chunk else b""
        flags = 0x02 if is_final else 0x00
        header = bytes(
            [
                VOLCENGINE_HEADER_VERSION,
                (ASR_MSG_TYPE_AUDIO_REQUEST << 4) | flags,
                (SERIALIZATION_RAW << 4) | (COMPRESSION_GZIP if audio_chunk else COMPRESSION_NONE),
                0x00,
            ]
        )
        frame = b"".join([header, struct.pack(">I", len(encoded)), encoded])
        await self.asr_ws.send_bytes(frame)

    async def _receive_asr_frame(self) -> AsrFrame:
        if self.asr_ws is None:
            raise RuntimeError("Volcengine split ASR websocket is not connected")

        message = await self.asr_ws.receive()
        if message.type == aiohttp.WSMsgType.CLOSE:
            raise RuntimeError("Volcengine split ASR websocket closed")
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(str(self.asr_ws.exception()))
        if message.type != aiohttp.WSMsgType.BINARY:
            raise RuntimeError(f"Unexpected ASR websocket message type: {message.type}")

        data = message.data
        message_type = data[1] >> 4
        flags = data[1] & 0x0F
        compression = data[2] & 0x0F
        index = 4

        if message_type == ASR_MSG_TYPE_ERROR:
            code = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload_size = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload = data[index : index + payload_size]
            if compression == COMPRESSION_GZIP:
                payload = gzip.decompress(payload)
            return AsrFrame(message_type=message_type, flags=flags, payload=payload, code=code)

        sequence: int | None = None
        if flags in {0x01, 0x03}:
            sequence = struct.unpack(">i", data[index : index + 4])[0]
            index += 4

        payload_size = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        payload = data[index : index + payload_size]
        if compression == COMPRESSION_GZIP:
            payload = gzip.decompress(payload)
        return AsrFrame(message_type=message_type, flags=flags, payload=payload, sequence=sequence)

    async def _start_tts_connection(self) -> None:
        if self.tts_ws is not None and not self.tts_ws.closed:
            return

        self.tts_ws = await self.session.ws_connect(
            TTS_WS_URL,
            headers=self._build_tts_headers(),
        )
        await self._send_tts_json_event(TTS_EVENT_START_CONNECTION, {})
        await self._wait_for_tts_control_event(
            success_event=TTS_EVENT_CONNECTION_STARTED,
            failure_events={TTS_EVENT_CONNECTION_FAILED},
            session_id=None,
        )
        self._tts_connection_started = True

    async def _synthesize_text(self, text: str) -> None:
        for attempt in range(2):
            await self._ensure_tts_connection()

            try:
                self._tts_cancel_requested = False
                self._tts_session_id = str(uuid.uuid4())
                self._tts_done_event = asyncio.Event()

                start_payload = {
                    "user": {"uid": self.dialog_id or self.session_id},
                    "event": TTS_EVENT_START_SESSION,
                    "namespace": "BidirectionalTTS",
                    "req_params": {
                        "speaker": self.tts_voice,
                        "audio_params": {
                            "format": "pcm",
                            "sample_rate": TTS_SAMPLE_RATE,
                        },
                    },
                }
                await self._send_tts_json_event(TTS_EVENT_START_SESSION, start_payload, session_id=self._tts_session_id)
                await self._wait_for_tts_control_event(
                    success_event=TTS_EVENT_SESSION_STARTED,
                    failure_events={TTS_EVENT_SESSION_FAILED},
                    session_id=self._tts_session_id,
                )

                await self._send_tts_json_event(
                    TTS_EVENT_TASK_REQUEST,
                    {"req_params": {"text": text}},
                    session_id=self._tts_session_id,
                )
                await self._finish_tts_session(self._tts_session_id)

                while True:
                    frame = await self._receive_tts_frame()
                    if frame.event_id == TTS_EVENT_SENTENCE_START:
                        continue
                    if frame.event_id == TTS_EVENT_RESPONSE:
                        if not self._tts_cancel_requested:
                            await self.events.put(ProviderAudioChunk(audio=frame.payload))
                        continue
                    if frame.event_id in {TTS_EVENT_SESSION_FINISHED, TTS_EVENT_SESSION_CANCELED}:
                        self._tts_done_event.set()
                        self._tts_session_id = None
                        return
                    if frame.event_id == TTS_EVENT_SESSION_FAILED:
                        payload = self._decode_json_bytes(frame.payload)
                        raise RuntimeError(payload.get("message") or payload.get("error") or "Volcengine split TTS session failed")
            except Exception as exc:
                is_retryable = self._is_retryable_tts_connection_error(exc)
                await self._reset_tts_connection(abort_only=is_retryable)
                if attempt == 0 and is_retryable:
                    logger.info("Retrying split TTS after transient connection failure: %s", exc)
                    continue
                raise

    async def _finish_tts_session(self, session_id: str) -> None:
        await self._send_tts_json_event(
            TTS_EVENT_FINISH_SESSION,
            {"event": TTS_EVENT_FINISH_SESSION},
            session_id=session_id,
        )

    async def _wait_for_tts_control_event(
        self,
        *,
        success_event: int,
        failure_events: set[int],
        session_id: str | None,
    ) -> TtsFrame:
        while True:
            frame = await self._receive_tts_frame()
            if session_id is not None and frame.session_id not in {None, session_id}:
                continue
            if frame.event_id == success_event:
                return frame
            if frame.event_id in failure_events:
                payload = self._decode_json_bytes(frame.payload)
                error_message = payload.get("message") or payload.get("error") or "Volcengine split TTS request failed"
                raise RuntimeError(error_message)

    async def _send_tts_json_event(self, event_id: int, payload: dict[str, Any], session_id: str | None = None) -> None:
        if self.tts_ws is None or self.tts_ws.closed or getattr(self.tts_ws, "_closing", False):
            raise RuntimeError("Volcengine split TTS websocket is not connected")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = bytes(
            [
                VOLCENGINE_HEADER_VERSION,
                (TTS_MSG_TYPE_FULL_REQUEST << 4) | TTS_FLAG_EVENT,
                (SERIALIZATION_JSON << 4) | COMPRESSION_NONE,
                0x00,
            ]
        )
        parts = [header, struct.pack(">I", event_id)]
        if session_id is not None:
            encoded_session = session_id.encode("utf-8")
            parts.extend([struct.pack(">I", len(encoded_session)), encoded_session])
        parts.extend([struct.pack(">I", len(body)), body])
        await self.tts_ws.send_bytes(b"".join(parts))

    async def _receive_tts_frame(self) -> TtsFrame:
        if self.tts_ws is None:
            raise RuntimeError("Volcengine split TTS websocket is not connected")

        message = await self.tts_ws.receive()
        if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
            raise RuntimeError("Volcengine split TTS websocket closed")
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(str(self.tts_ws.exception()))
        if message.type != aiohttp.WSMsgType.BINARY:
            raise RuntimeError(f"Unexpected TTS websocket message type: {message.type}")

        data = message.data
        message_type = data[1] >> 4
        index = 4

        if message_type == TTS_MSG_TYPE_ERROR:
            code = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload_size = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            payload = data[index : index + payload_size]
            return TtsFrame(event_id=TTS_EVENT_SESSION_FAILED, message_type=message_type, payload=payload, code=code)

        event_id = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        session_id: str | None = None
        if event_id not in {TTS_EVENT_CONNECTION_STARTED, TTS_EVENT_CONNECTION_FAILED, TTS_EVENT_CONNECTION_FINISHED}:
            session_size = struct.unpack(">I", data[index : index + 4])[0]
            index += 4
            session_id = data[index : index + session_size].decode("utf-8")
            index += session_size

        payload_size = struct.unpack(">I", data[index : index + 4])[0]
        index += 4
        payload = data[index : index + payload_size]
        return TtsFrame(event_id=event_id, message_type=message_type, payload=payload, session_id=session_id)

    def _build_asr_headers(self, request_id: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-Api-Resource-Id": self.asr_resource_id,
            "X-Api-Connect-Id": request_id,
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
            return headers

        assert self.app_id is not None
        assert self.access_key is not None
        headers["X-Api-App-Key"] = self.app_id
        headers["X-Api-Access-Key"] = self.access_key
        return headers

    def _build_tts_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-Api-Resource-Id": self.tts_resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
            return headers

        assert self.app_id is not None
        assert self.access_key is not None
        headers["X-Api-App-Id"] = self.app_id
        headers["X-Api-Access-Key"] = self.access_key
        return headers

    def _normalize_tts_voice(self, requested_voice: str) -> str:
        fallback_voice = os.getenv("VOLCENGINE_SPLIT_TTS_VOICE", DEFAULT_SPLIT_TTS_VOICE).strip() or DEFAULT_SPLIT_TTS_VOICE
        if not requested_voice:
            return fallback_voice

        if self.tts_resource_id.startswith("seed-tts-2.0") and requested_voice == DEFAULT_VOLCENGINE_VOICE:
            logger.warning(
                "Replacing integrated Volcengine voice '%s' with standalone TTS-compatible voice '%s'",
                requested_voice,
                fallback_voice,
            )
            return fallback_voice

        return requested_voice

    async def _ensure_tts_connection(self) -> None:
        if self.tts_ws is None:
            await self._start_tts_connection()
            return

        if self.tts_ws.closed or getattr(self.tts_ws, "_closing", False):
            await self._reset_tts_connection(abort_only=True)
            await self._start_tts_connection()

    async def _reset_tts_connection(self, *, abort_only: bool = False) -> None:
        if self.tts_ws is not None and not self.tts_ws.closed:
            if not abort_only and self._tts_session_id is not None:
                try:
                    await self._finish_tts_session(self._tts_session_id)
                except Exception:
                    pass
            if not abort_only and self._tts_connection_started:
                try:
                    await self._send_tts_json_event(TTS_EVENT_FINISH_CONNECTION, {}, session_id=None)
                except Exception:
                    pass
            try:
                await self.tts_ws.close()
            except Exception:
                pass

        self.tts_ws = None
        self._tts_connection_started = False
        self._tts_session_id = None
        self._tts_done_event = None

    def _decode_json_bytes(self, payload: bytes) -> dict[str, Any]:
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    def _is_retryable_tts_connection_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in {
                "closing transport",
                "websocket closed",
                "not connected",
                "connection reset",
                "broken pipe",
            }
        )
