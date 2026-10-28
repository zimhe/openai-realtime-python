"""Per-Agora-UID speech detection and streaming transcription.

The caller owns the participant identity. An ASR session receives audio from
one UID only, so no acoustic speaker clustering is needed for attribution.
"""

import asyncio
import gzip
import json
import math
import os
import struct
import uuid
from array import array
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from realtime_agent.utils import PCMChunker, resample_pcm16


ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
ASR_RATE = 16000
RTC_RATE = 24000
ASR_CHUNK_BYTES = 640
ASR_RESOURCE_ID = "volc.seedasr.sauc.duration"


class EnergySpeechDetector:
    """A cheap first-stage VAD; ASR still decides the actual spoken words."""

    def __init__(
        self,
        *,
        min_rms: float = 250.0,
        onset_ms: float = 40.0,
        release_ms: float = 900.0,
    ):
        self.min_rms = min_rms
        self.onset_ms = onset_ms
        self.release_ms = release_ms
        self.noise_floor = 100.0
        self.candidate_ms = 0.0
        self.silence_ms = 0.0
        self.active = False

    def feed(self, pcm16: bytes) -> tuple[bool, bool]:
        if not pcm16:
            return False, False
        samples = array("h")
        samples.frombytes(pcm16)
        if not samples:
            return False, False
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        frame_ms = len(samples) * 1000.0 / RTC_RATE
        speaking = rms >= max(self.min_rms, self.noise_floor * 3.0)
        started = stopped = False

        if speaking:
            self.candidate_ms += frame_ms
            self.silence_ms = 0.0
            if not self.active and self.candidate_ms >= self.onset_ms:
                self.active = True
                started = True
        else:
            self.candidate_ms = 0.0
            if not self.active:
                self.noise_floor = self.noise_floor * 0.97 + rms * 0.03
            else:
                self.silence_ms += frame_ms
                if self.silence_ms >= self.release_ms:
                    self.active = False
                    self.silence_ms = 0.0
                    stopped = True
        return started, stopped


class SpeakerAsrSession:
    """One short-lived Volcengine SAUC connection for one speaker's utterance."""

    def __init__(
        self,
        uid: int,
        on_text: Callable[[int, str, bool], Awaitable[None]],
    ):
        self.uid = uid
        self.on_text = on_text
        self.auth_mode = os.getenv("VOLCENGINE_ASR_AUTH_MODE", "api_key").strip().lower()
        self.api_key = (os.getenv("VOLCENGINE_ASR_API_KEY") or
                        os.getenv("VOLCENGINE_API_KEY", "")).strip()
        self.app_id = os.getenv("VOLCENGINE_APP_ID", "").strip()
        self.access_key = os.getenv("VOLCENGINE_ACCESS_KEY", "").strip()
        if self.auth_mode == "legacy":
            self.api_key = ""
        if not self.api_key and not (self.app_id and self.access_key):
            raise ValueError("Volcengine ASR requires VOLCENGINE_API_KEY or legacy app credentials")
        self.resource_id = os.getenv("VOLCENGINE_ASR_RESOURCE_ID", ASR_RESOURCE_ID)
        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.reader: asyncio.Task[None] | None = None
        self.chunker = PCMChunker(ASR_CHUNK_BYTES)
        self.last_interim = ""
        self.last_final = ""
        self.final_received = asyncio.Event()

    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        headers = {
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        else:
            headers["X-Api-App-Key"] = self.app_id
            headers["X-Api-Access-Key"] = self.access_key
        try:
            self.ws = await self.session.ws_connect(ASR_URL, headers=headers)
            request = {
                "user": {"uid": str(self.uid)},
                "audio": {
                    "format": "pcm",
                    "codec": "raw",
                    "rate": ASR_RATE,
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
                    "end_window_size": 800,
                },
            }
            await self._send_frame(1, json.dumps(request).encode("utf-8"), json_body=True)
            self.reader = asyncio.create_task(self._read_loop())
        except aiohttp.WSServerHandshakeError as exc:
            log_id = exc.headers.get("X-Tt-Logid", "") if exc.headers else ""
            await self.close()
            raise RuntimeError(
                f"ASR handshake HTTP {exc.status}; resource_id={self.resource_id}; "
                f"auth={'api_key' if self.api_key else 'legacy_app_credentials'}; "
                f"log_id={log_id or 'unavailable'}"
            ) from exc
        except BaseException:
            await self.close()
            raise

    async def send_pcm24(self, pcm16_24k: bytes) -> None:
        data = resample_pcm16(pcm16_24k, source_rate=RTC_RATE, target_rate=ASR_RATE)
        for chunk in self.chunker.push(data):
            await self._send_frame(2, chunk)

    async def finish(self) -> None:
        try:
            if self.ws is not None and not self.ws.closed:
                remainder = bytes(self.chunker.buffer)
                self.chunker.clear()
                if remainder:
                    await self._send_frame(2, remainder)
                await self._send_frame(2, b"", final=True)
            if self.reader is not None:
                try:
                    await asyncio.wait_for(self.final_received.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    if self.reader.done() and self.reader.exception():
                        raise self.reader.exception()
                    if self.last_interim:
                        await self.on_text(self.uid, self.last_interim, True)
        finally:
            await self.close()

    async def _send_frame(
        self,
        kind: int,
        payload: bytes,
        *,
        json_body: bool = False,
        final: bool = False,
    ) -> None:
        if self.ws is None or self.ws.closed:
            raise ConnectionError("ASR WebSocket is closed")
        encoded = gzip.compress(payload) if payload else b""
        flags = 0x02 if final else 0x00
        serialization = 0x10 if json_body else 0x00
        compression = 0x01 if payload else 0x00
        header = bytes([0x11, (kind << 4) | flags, serialization | compression, 0])
        await self.ws.send_bytes(header + struct.pack(">I", len(encoded)) + encoded)

    async def _read_loop(self) -> None:
        assert self.ws is not None
        async for message in self.ws:
            if message.type == aiohttp.WSMsgType.ERROR:
                raise ConnectionError("ASR WebSocket error") from self.ws.exception()
            if message.type != aiohttp.WSMsgType.BINARY:
                continue
            data = message.data
            if len(data) < 8:
                continue
            message_type = data[1] >> 4
            flags = data[1] & 0x0F
            offset = 4
            if message_type == 0x0F:
                if len(data) < 12:
                    continue
                code = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
            else:
                code = None
                if flags in (0x01, 0x03):
                    offset += 4
            if len(data) < offset + 4:
                continue
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            body = data[offset + 4:offset + 4 + size]
            if data[2] & 0x0F == 0x01:
                body = gzip.decompress(body)
            payload: dict[str, Any] = json.loads(body.decode("utf-8")) if body else {}
            if code is not None:
                raise RuntimeError(f"Volcengine ASR error {code}: {payload}")
            for utterance in (payload.get("result") or {}).get("utterances") or []:
                text = str(utterance.get("text") or "").strip()
                if not text:
                    continue
                final = bool(utterance.get("definite"))
                if final:
                    if text != self.last_final:
                        self.last_final = text
                        await self.on_text(self.uid, text, True)
                    self.final_received.set()
                elif text != self.last_interim:
                    self.last_interim = text
                    await self.on_text(self.uid, text, False)

    async def close(self) -> None:
        if self.reader is not None and self.reader is not asyncio.current_task():
            if not self.reader.done():
                self.reader.cancel()
                try:
                    await self.reader
                except asyncio.CancelledError:
                    pass
        self.reader = None
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        self.ws = None
        if self.session is not None:
            await self.session.close()
        self.session = None
