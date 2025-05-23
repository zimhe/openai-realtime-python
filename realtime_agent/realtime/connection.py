import asyncio
import base64
import json
import logging
import os
import aiohttp

from typing import Any, AsyncGenerator
from .struct import InputAudioBufferAppend, ClientToServerMessage, ServerToClientMessage,UserMessageItemParam,ItemCreate, parse_server_message, to_json
from ..logger import setup_logger

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)


DEFAULT_VIRTUAL_MODEL = "gpt-4o-realtime-preview"

def smart_str(s: str, max_field_len: int = 128) -> str:
    """parse string as json, truncate data field to 128 characters, reserialize"""
    try:
        data = json.loads(s)
        if "delta" in data:
            key = "delta"
        elif "audio" in data:
            key = "audio"
        else:
            return s

        if len(data[key]) > max_field_len:
            data[key] = data[key][:max_field_len] + "..."
        return json.dumps(data)
    except json.JSONDecodeError:
        return s


class RealtimeApiConnection:
    def __init__(
        self,
        base_uri: str,
        api_key: str | None = None,
        path: str = "/v1/realtime",
        verbose: bool = False,
        model: str = DEFAULT_VIRTUAL_MODEL,
    ):
        
        self.url = f"{base_uri}{path}"
        if "model=" not in self.url:
            self.url += f"?model={model}"

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.verbose = verbose
        self.session = aiohttp.ClientSession()

    async def __aenter__(self) -> "RealtimeApiConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        await self.close()
        return False

    async def connect(self):
        auth = aiohttp.BasicAuth("", self.api_key) if self.api_key else None

        headers = {"OpenAI-Beta": "realtime=v1"}

        self.websocket = await self.session.ws_connect(
            url=self.url,
            auth=auth,
            headers=headers,
        )

    async def send_audio_data(self, audio_data: bytes):
        """audio_data is assumed to be pcm16 24kHz mono little-endian"""
        base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
        message = InputAudioBufferAppend(audio=base64_audio_data)
        await self.send_request(message)
        #return message.event_id
        
    async def send_text(self, text: str):
        item = UserMessageItemParam(
        content=[{"type": "input_text", "text": text}]
        )
        message = ItemCreate(item=item)
        
        await self.send_request(message)

    async def send_request(self, message: ClientToServerMessage):
        assert self.websocket is not None
        message_str = to_json(message)
        if self.verbose:
            logger.info(f"-> {smart_str(message_str)}")
        await self.websocket.send_str(message_str)

    

    async def listen(self) -> AsyncGenerator[ServerToClientMessage, None]:
        assert self.websocket is not None
        if self.verbose:
            logger.info("Listening for realtimeapi messages")
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if self.verbose:
                        logger.info(f"<- {smart_str(msg.data)}")
                    yield self.handle_server_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Error during receive: %s", self.websocket.exception())
                    break
        except asyncio.CancelledError:
            logger.info("Receive messages task cancelled")

    def handle_server_message(self, message: str) -> ServerToClientMessage:
        try:
            return parse_server_message(message)
        except Exception as e:
            logger.error("Error handling message: " + str(e))
            raise e

    async def close(self):
        # Close the websocket connection if it exists
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
