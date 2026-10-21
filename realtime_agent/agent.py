import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, List

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from agora_realtime_ai_api.rtc import ChatMessage, RtcOptions
from openai import OpenAI
from .agora_mixed.rtc_mixed import MixedChannel, MixedRtcEngine

from .logger import setup_logger
from .providers import (
    ProviderAudioChunk,
    ProviderError,
    ProviderResponseFinished,
    ProviderResponseStarted,
    ProviderSessionStarted,
    ProviderSpeechStarted,
    ProviderSpeechStopped,
    ProviderToolCall,
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    RealtimeProviderConnection,
    create_realtime_provider,
    get_realtime_provider_name,
    provider_event_to_json,
)
from .realtime.agent_functions import AgentToolsMetaWorkplaces
from .tools import ClientToolCallResponse, LocalToolCallExecuted, ToolContext
from .utils import PCMWriter

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)


def _get_queue_warning_threshold() -> int:
    raw_value = os.environ.get("AGENT_AUDIO_QUEUE_WARN_THRESHOLD", "20").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning("Invalid AGENT_AUDIO_QUEUE_WARN_THRESHOLD=%r, falling back to 20", raw_value)
        return 20

def _monitor_queue_size(queue: asyncio.Queue, queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logger.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_users(channel: MixedChannel, timeout: float = 15.0) -> List[int]:
    remote_users = list(channel.remote_users.keys())
    if remote_users:
        return remote_users

    future = asyncio.Future()

    def on_user_joined(conn, user_id):
        if not future.done():
            future.set_result(None)  # ✅ 不需要值，只为唤醒等待

    channel.once("user_joined", on_user_joined)

    try:
        await asyncio.wait_for(future, timeout=timeout)
        return list(channel.remote_users.keys())
    except asyncio.TimeoutError:
        logger.warning("Timeout while waiting for remote users.")
        return []


def contains_fuzzy_phrase(text, phrase, threshold=0.8):
    normalized_text = " ".join(re.findall(r"\w+", text.lower()))
    normalized_phrase = " ".join(re.findall(r"\w+", phrase.lower()))
    if not normalized_text or not normalized_phrase:
        return False

    words = normalized_text.split()
    phrase_words = normalized_phrase.split()
    phrase_len = len(phrase_words)
    target_phrase = " ".join(phrase_words)

    for i in range(len(words) - phrase_len + 1):
        window = " ".join(words[i:i + phrase_len])
        if window == target_phrase:
            return True
        if phrase_len > 1 and SequenceMatcher(None, window, target_phrase).ratio() >= 0.92:
            return True
    return False


@dataclass(frozen=True)
class FunctionCallOutcome:
    used_tools: bool
    final_text: str | None = None


class DoubaoFunctionCallOrchestrator:
    def __init__(self, tools: ToolContext) -> None:
        self.tools = tools
        api_key = os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY or VOLCENGINE_ARK_API_KEY must be set for Doubao function calling")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        )
        self.model = os.getenv("VOLCENGINE_FUNCTION_CALL_MODEL", "doubao-seed-2-0-lite-260428")
        self.parallel_tool_calls = os.getenv("VOLCENGINE_PARALLEL_TOOL_CALLS", "false").strip().lower() == "true"
        self.tools_payload = self._build_tools_payload()

    async def respond(self, transcript: str) -> FunctionCallOutcome:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the function-calling planner for a realtime voice assistant. "
                    "Use tools only when they are actually needed. "
                    "If a tool is not needed, answer the user directly in the user's language. "
                    "If required parameters are missing, ask a concise follow-up question instead of guessing."
                ),
            },
            {"role": "user", "content": transcript},
        ]
        used_tools = False

        for _ in range(4):
            completion = await asyncio.to_thread(self._create_completion, messages)
            choice = completion.choices[0]
            response_message = choice.message

            if choice.finish_reason != "tool_calls" or not response_message.tool_calls:
                return FunctionCallOutcome(
                    used_tools=used_tools,
                    final_text=(response_message.content or "").strip() or None,
                )

            used_tools = True
            messages.append(response_message.model_dump(exclude_none=True))

            for tool_call in response_message.tool_calls:
                tool_result = await self._execute_tool(tool_call.function.name, tool_call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return FunctionCallOutcome(used_tools=used_tools, final_text=None)

    def _build_tools_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for tool in self.tools.model_description():
            if tool.get("type") != "function":
                continue

            payload.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return payload

    def _create_completion(self, messages: list[dict[str, Any]]):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools_payload,
            parallel_tool_calls=self.parallel_tool_calls,
        )

    async def _execute_tool(self, tool_name: str, encoded_arguments: str) -> str:
        result = await self.tools.execute_tool(tool_name, encoded_arguments)

        if isinstance(result, LocalToolCallExecuted):
            return result.json_encoded_output

        if result is None:
            return json.dumps(
                {"status": "error", "message": f"Tool '{tool_name}' is not registered."},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "error",
                "message": f"Tool '{tool_name}' is not available for local execution in the voice agent.",
            },
            ensure_ascii=False,
        )

@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    system_message: str | None = None
    turn_detection: Any | None = None
    voice: str | None = None
    dialog_id: str | None = None


class RealtimeKitAgent:
    engine: MixedRtcEngine
    channel: MixedChannel
    provider: RealtimeProviderConnection
    tools: ToolContext | None = None

    _client_tool_futures: dict[str, asyncio.Future[ClientToolCallResponse]]

    @classmethod
    async def setup_and_run_agent(
        cls,
        *,
        engine: MixedRtcEngine,
        options: RtcOptions,
        inference_config: InferenceConfig,
        tools: ToolContext | None,
        screen_keys:list[str] = None,
    ) -> None:
        provider_name = get_realtime_provider_name()
        async with create_realtime_provider(verbose=False) as provider:
            session = await provider.start_session(
                system_message=inference_config.system_message,
                voice=inference_config.voice,
                tools=tools,
                turn_detection=inference_config.turn_detection,
                dialog_id=inference_config.dialog_id,
            )
            logger.info(
                "Session started: provider=%s session_id=%s model=%s",
                provider_name,
                session.session_id,
                session.model,
            )

            channel = engine.create_channel(options)
            channel.on
            await channel.connect()

            try:
                agent = cls(
                    provider=provider,
                    tools=tools,
                    channel=channel,
                    require_explicit_trigger=bool(
                        inference_config.turn_detection is not None
                        and getattr(inference_config.turn_detection, "create_response", True) is False
                    ),
                )

                if isinstance(tools, AgentToolsMetaWorkplaces):
                    tools.set_agent(agent)
                    tools.set_screen_context(screen_keys)

                await agent.run()
            finally:
                await channel.disconnect()

    def __init__(
        self,
        *,
        provider: RealtimeProviderConnection,
        tools: ToolContext | None,
        channel: MixedChannel,
        require_explicit_trigger: bool,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self._client_tool_futures = {}
        self.channel = channel
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.subscribe_users = set()
        self.write_pcm = os.environ.get("WRITE_AGENT_PCM", "false") == "true"
        logger.info(f"Write PCM: {self.write_pcm}")
        self.last_trigger_time = 0
        self.last_response_time = 0
        self.require_explicit_trigger = require_explicit_trigger
        self.queue_warning_threshold = _get_queue_warning_threshold()
        self._expecting_provider_response = not require_explicit_trigger
        self._user_speech_stopped = asyncio.Event()
        self._user_speech_stopped.set()
        self._doubao_response_task: asyncio.Task[None] | None = None
        self.doubao_function_call_orchestrator = None
        if tools is not None and provider.provider_name == "volcengine" and not provider.capabilities.supports_tools:
            try:
                self.doubao_function_call_orchestrator = DoubaoFunctionCallOrchestrator(tools)
            except ValueError as exc:
                logger.warning("Doubao function calling disabled: %s", exc)
        self.ACTIVE_WINDOW_SECONDS = 20  # 你可以自定义时间长度

        self.MENTION_PATTERN = "<color=#FF4500><b>@agent</b></color>"
        self.TRIGGER_PHRASES = ["hey agent", "hi agent", "hello agent", "hi, assistant", "hey assistant", "hello assistant"]
        self.STOP_PHRASES = ["stop", "cancel", "nevermind", "never mind", "abort", "quiet", "shut up", "be quiet", "stop talking", "mute"]

    def _in_active_window(self, current_time: float | None = None) -> bool:
        current_time = current_time or time.time()
        return (
            (current_time - self.last_trigger_time) < self.ACTIVE_WINDOW_SECONDS
            or (current_time - self.last_response_time) < self.ACTIVE_WINDOW_SECONDS
        )

    async def run(self) -> None:
        try:

            def log_exception(t: asyncio.Task[Any]) -> None:
                if not t.cancelled() and t.exception():
                    logger.error(
                        "unhandled exception",
                        exc_info=t.exception(),
                    )
                    
            async def on_user_joined( 
                agora_rtc_conn: RTCConnection, user_id: int
            ):
                logger.info(f"On User Joined Callback: {user_id}")

                if user_id not in self.subscribe_users:
                    self.subscribe_users.add(user_id)
                    await self.channel.subscribe_audio(user_id)
                    logger.info(f"Subscribed to user {user_id}, current users: {self.subscribe_users}")

            self.channel.on("user_joined", on_user_joined)


            async def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
                logger.info(f"Received stream message data {data} with length: {length}")
                await self._process_stream_message(user_id, stream_id, data, length)
                

            self.channel.on("stream_message", on_stream_message)

            logger.info("Waiting for remote user to join")
            
            user_ids = await wait_for_remote_users(self.channel)
            for user_id in user_ids:
                if user_id not in self.subscribe_users:
                    await self.channel.subscribe_audio(user_id)
                    self.subscribe_users.add(user_id)
                    logger.info(f"Subscribed to user outside <on_user_joined> {user_id}, current users: {self.subscribe_users}")

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if user_id in self.subscribe_users:
                    self.subscribe_users.discard(user_id)
                    logger.info(f"Subscribed user {user_id} left, removed from user list")
                    if len(self.subscribe_users) == 0:
                        logger.info("No more users to subscribe to, disconnecting")
                        self.channel.channel_event_observer.mixed_audio_stream.queue.put_nowait(None)
                        await self.channel.disconnect()

            self.channel.on("user_left", on_user_left)

            disconnected_future = asyncio.Future[None]()

            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)

            self.channel.on("connection_state_changed", callback)

            asyncio.create_task(self.rtc_to_model()).add_done_callback(log_exception)
            asyncio.create_task(self.model_to_rtc()).add_done_callback(log_exception)

            asyncio.create_task(self._process_model_messages()).add_done_callback(log_exception)

            await disconnected_future
            logger.info("Agent finished running")
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    async def _process_stream_message(self, user_id, stream_id, data, length) -> None:
        try:
            text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)

            if self.require_explicit_trigger and not self.provider.capabilities.supports_manual_responses:
                current_time = time.time()
                if self.MENTION_PATTERN in text:
                    self.last_trigger_time = current_time
                elif not self._in_active_window(current_time):
                    logger.info("Skipping stream message without mention while trigger gating is enabled")
                    return

            await self.provider.send_text(text)

            if self.provider.capabilities.supports_manual_responses and self.MENTION_PATTERN in text:
                logger.info("Mention pattern found in stream message, requesting response")
                await self.provider.request_response()
        except Exception as e:
            logger.error(f"Error Sending stream message: {e}")

    async def rtc_to_model(self) -> None:
        # 等待至少有一个用户的音频流可用
        while not self.subscribe_users or self.channel.get_mixed_audio_frames() is None:
            await asyncio.sleep(0.1)

        audio_frames = self.channel.get_mixed_audio_frames()

        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)

        try:
            async for audio_frame in audio_frames:
                # Process received audio (send to model)
                try:
                    await self.provider.send_audio_data(audio_frame.data)
                except ConnectionError as exc:
                    logger.info("Realtime transport closed while streaming mixed audio; stopping upload: %s", exc)
                    break

                # Write PCM data if enabled
                await pcm_writer.write(audio_frame.data)

                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            raise  # Re-raise the exception to propagate cancellation
        finally:
            await pcm_writer.flush()

    
    async def rtc_to_model_for_user(self, user_id:int) -> None:
        logger.info(f"rtc_to_model_for_user started: {user_id=}")
        while self.channel.get_audio_frames(user_id) is None:
            logger.info(f"No audio frames for user: {user_id}, waiting...")
            await asyncio.sleep(0.1)

        audio_frames = self.channel.get_audio_frames(user_id)
        
        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix=f"rtc_to_model_for_user_{user_id}", write_pcm=self.write_pcm)

        try:
            async for audio_frame in audio_frames:
                # Process received audio (send to model)
                try:
                    await self.provider.send_audio_data(audio_frame.data)
                except ConnectionError as exc:
                    logger.info("Realtime transport closed while streaming audio for user %s; stopping upload: %s", user_id, exc)
                    break

                await pcm_writer.write(audio_frame.data)

                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            raise  # Re-raise the exception to propagate cancellation
        finally:
            await pcm_writer.flush()

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)

        try:
            while True:
                # Get audio frame from the model output
                frame = await self.audio_queue.get()
                # Process sending audio (to RTC)
                await self.channel.push_audio_frame(frame)
                
                # Write PCM data if enabled
                await pcm_writer.write(frame)

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task

    async def handle_function_call(self, message: ProviderToolCall) -> None:
        if self.tools is None:
            logger.warning("Received tool call without configured tools: %s", message.name)
            return

        function_call_response = await self.tools.execute_tool(message.name, message.arguments)
        logger.info(f"Function call response: {function_call_response}")

        await self.provider.send_tool_output(
            call_id=message.call_id,
            output=function_call_response.json_encoded_output,
        )

    def _publish_transcript_event(self, event: ProviderTranscriptDelta | ProviderTranscriptDone) -> None:
        asyncio.create_task(
            self.channel.chat.send_message(
                ChatMessage(
                    message=provider_event_to_json(event),
                    msg_id=event.item_id or "",
                )
            )
        )

    async def _handle_user_transcript(self, transcript: str, user_item_id: str | None = None) -> None:
        normalized = transcript.lower().strip()
        supports_manual_responses = self.provider.capabilities.supports_manual_responses
        current_time = time.time()
        in_active_window = self._in_active_window(current_time)
        should_respond = not self.require_explicit_trigger
        matched_trigger = any(contains_fuzzy_phrase(normalized, phrase) for phrase in self.TRIGGER_PHRASES)

        if any(contains_fuzzy_phrase(normalized, phrase) for phrase in self.STOP_PHRASES):
            logger.info("Stop phrase detected. Aborting response.")
            self.last_trigger_time = 0
            self.last_response_time = 0
            self._expecting_provider_response = False
            await self.provider.interrupt_response()
            return

        if matched_trigger:
            should_respond = True
            self.last_trigger_time = current_time
        elif in_active_window:
            should_respond = True

        should_orchestrate_with_doubao = (
            should_respond
            and not supports_manual_responses
            and self.doubao_function_call_orchestrator is not None
        )

        if should_orchestrate_with_doubao:
            self._expecting_provider_response = False
            if self._doubao_response_task is not None and not self._doubao_response_task.done():
                self._doubao_response_task.cancel()
            self._doubao_response_task = asyncio.create_task(
                self._maybe_handle_doubao_function_call_response(transcript, user_item_id)
            )
            return

        if not supports_manual_responses:
            self._expecting_provider_response = should_respond

        if not self.require_explicit_trigger:
            return

        if matched_trigger:
            if supports_manual_responses:
                logger.info("Trigger word detected, requesting response")
                await self.provider.request_response()
            else:
                logger.info("Trigger word detected, allowing provider auto-response")
        elif in_active_window:
            if supports_manual_responses:
                logger.info("Still in active window. Responding without trigger phrase.")
                await self.provider.request_response()
            else:
                logger.info("Still in active window. Allowing provider auto-response.")
        else:
            logger.info("Trigger word not found, skipping response")
            if not supports_manual_responses:
                self.last_response_time = 0
                await self.provider.interrupt_response()

    async def _maybe_handle_doubao_function_call_response(self, transcript: str, user_item_id: str | None) -> None:
        if self.doubao_function_call_orchestrator is None:
            return

        try:
            outcome = await self.doubao_function_call_orchestrator.respond(transcript)
        except asyncio.CancelledError:
            logger.info("Cancelled stale Doubao orchestration for transcript '%s'", transcript)
            raise
        except Exception as exc:
            logger.warning("Doubao function calling failed for transcript '%s': %s", transcript, exc)
            return

        if not outcome.final_text:
            return

        logger.info("Doubao orchestration generated a response (used_tools=%s)", outcome.used_tools)

        await self._user_speech_stopped.wait()

        try:
            await self.provider.interrupt_response()
        except Exception as exc:
            logger.warning("Failed to interrupt provider before Doubao tool response: %s", exc)

        self._expecting_provider_response = True
        try:
            send_tts_text = getattr(self.provider, "send_tts_text", None)
            if callable(send_tts_text):
                await send_tts_text(outcome.final_text)
            else:
                await self.provider.send_text(
                    "Please say the following answer naturally in the same language, without adding extra information:\n"
                    f"{outcome.final_text}"
                )
        except Exception as exc:
            self._expecting_provider_response = False
            logger.warning("Failed to send Doubao tool-grounded answer back to provider: %s", exc)
            return

        upsert_conversation_turn = getattr(self.provider, "upsert_conversation_turn", None)
        if callable(upsert_conversation_turn):
            try:
                await upsert_conversation_turn(
                    question_id=user_item_id,
                    user_text=transcript,
                    assistant_text=outcome.final_text,
                )
            except Exception as exc:
                logger.warning("Failed to persist Doubao tool-grounded answer into conversation context: %s", exc)

    async def _process_model_messages(self) -> None:
        async for message in self.provider.listen():
            match message:
                case ProviderAudioChunk():
                    self.audio_queue.put_nowait(message.audio)
                    _monitor_queue_size(
                        self.audio_queue,
                        "audio_queue",
                        threshold=self.queue_warning_threshold,
                    )
                case ProviderTranscriptDelta() if message.role == "assistant":
                    if not self.provider.capabilities.supports_manual_responses and not self._expecting_provider_response:
                        logger.info("Suppressing unsolicited assistant transcript delta")
                    else:
                        self._publish_transcript_event(message)
                case ProviderTranscriptDelta() if message.role == "user":
                    self._publish_transcript_event(message)
                case ProviderTranscriptDone() if message.role == "assistant":
                    if not self.provider.capabilities.supports_manual_responses and not self._expecting_provider_response:
                        logger.info("Suppressing unsolicited assistant transcript done: %s", message.text)
                    else:
                        logger.info("Assistant transcript done: %s", message.text)
                        self._publish_transcript_event(message)
                case ProviderTranscriptDone() if message.role == "user":
                    logger.info("User transcript done: %s", message.text)
                    self._publish_transcript_event(message)
                    await self._handle_user_transcript(message.text, message.item_id)
                case ProviderSpeechStarted():
                    self._user_speech_stopped.clear()
                    await self.channel.clear_sender_audio_buffer()
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info("Speech started: item_id=%s", message.item_id)
                case ProviderSpeechStopped():
                    self._user_speech_stopped.set()
                    logger.info("Speech stopped: item_id=%s", message.item_id)
                case ProviderResponseStarted():
                    if not self.provider.capabilities.supports_manual_responses and not self._expecting_provider_response:
                        logger.info("Interrupting unsolicited provider response")
                        await self.provider.interrupt_response()
                        await self.channel.clear_sender_audio_buffer()
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                    else:
                        self.last_response_time = time.time()
                case ProviderResponseFinished():
                    self._expecting_provider_response = False
                case ProviderToolCall():
                    logger.info("Provider tool call: %s", message.name)
                    asyncio.create_task(
                        self.handle_function_call(message)
                    )
                case ProviderError():
                    logger.error("Provider error: %s (%s)", message.message, message.code)
                case ProviderSessionStarted():
                    logger.info("Unexpected session event during run: %s", message.session_id)
                case _:
                    logger.warning(f"Unhandled message {message=}")
