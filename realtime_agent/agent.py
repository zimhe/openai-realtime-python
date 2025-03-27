import asyncio
import base64
import logging
import os
from builtins import anext
from typing import Any

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from attr import dataclass

from agora_realtime_ai_api.rtc import Channel, ChatMessage, RtcEngine, RtcOptions

from .logger import setup_logger
from .realtime.struct import ErrorMessage, FunctionCallOutputItemParam, InputAudioBufferCommitted, InputAudioBufferSpeechStarted, InputAudioBufferSpeechStopped, InputAudioTranscription, ItemCreate, ItemCreated, ItemInputAudioTranscriptionDelta,ItemInputAudioTranscriptionCompleted, RateLimitsUpdated, ResponseAudioDelta, ResponseAudioDone, ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone, ResponseContentPartAdded, ResponseContentPartDone, ResponseCreate, ResponseCreated, ResponseDone, ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone, ResponseOutputItemAdded, ResponseOutputItemDone, ServerVADUpdateParams, SessionUpdate, SessionUpdateParams, SessionUpdated, Voices, to_json
from .realtime.connection import RealtimeApiConnection
from .realtime.agent_functions import AgentToolsMetaWorkplaces
from .tools import ClientToolCallResponse, ToolContext
from .utils import PCMWriter
from .multi_user_audio_stream import MultiUserAudioStream
from agora_realtime_ai_api.rtc import AudioStream,PcmAudioFrame
from agora.rtc.audio_pcm_data_sender import PcmAudioFrame
from typing import Any, AsyncIterator
import json

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

def _monitor_queue_size(queue: asyncio.Queue, queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logger.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel) -> int:
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout of 30 seconds
        remote_user = await asyncio.wait_for(future, timeout=15.0)
        return remote_user
    except KeyboardInterrupt:
        future.cancel()
        
    except Exception as e:
        logger.error(f"Error waiting for remote user: {e}")
        raise

@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    system_message: str | None = None
    turn_detection: ServerVADUpdateParams | None = None  # MARK: CHECK!
    voice: Voices | None = None


class RealtimeKitAgent:
    engine: RtcEngine
    channel: Channel
    connection: RealtimeApiConnection
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    message_queue: asyncio.Queue[ResponseAudioTranscriptDelta] = (
        asyncio.Queue()
    )
    message_done_queue: asyncio.Queue[ResponseAudioTranscriptDone] = (
        asyncio.Queue()
    )
    tools: ToolContext | None = None

    _client_tool_futures: dict[str, asyncio.Future[ClientToolCallResponse]]

    @classmethod
    async def setup_and_run_agent(
        cls,
        *,
        engine: RtcEngine,
        options: RtcOptions,
        inference_config: InferenceConfig,
        tools: ToolContext | None,
        screen_keys:list[str] = None,
    ) -> None:
        channel = engine.create_channel(options)
        await channel.connect()

        try:
            async with RealtimeApiConnection(
                base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
                api_key=os.getenv("OPENAI_API_KEY"),
                verbose=False,
            ) as connection:
                await connection.send_request(
                    SessionUpdate(
                        session=SessionUpdateParams(
                            # MARK: check this
                            turn_detection=inference_config.turn_detection,
                            tools=tools.model_description() if tools else [],
                            tool_choice="auto",
                            input_audio_format="pcm16",
                            output_audio_format="pcm16",
                            instructions=inference_config.system_message,
                            voice=inference_config.voice,
                            model=os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview"),
                            modalities=["text", "audio"],
                            temperature=0.8,
                            max_response_output_tokens="inf",
                            input_audio_transcription=InputAudioTranscription(model="whisper-1")
                        )
                    )
                )

                start_session_message = await anext(connection.listen())
                # assert isinstance(start_session_message, messages.StartSession)
                if isinstance(start_session_message, SessionUpdated):
                    logger.info(
                        f"Session started: {start_session_message.session.id} model: {start_session_message.session.model}"
                    )
                elif isinstance(start_session_message, ErrorMessage):
                    logger.info(
                        f"Error: {start_session_message.error}"
                    )

                agent = cls(
                    connection=connection,
                    tools=tools,
                    channel=channel
                )
                
                if isinstance(tools, AgentToolsMetaWorkplaces):
                    tools.set_agent(agent)
                    tools.set_screen_context(screen_keys)
                
                await agent.run()

        finally:
            await channel.disconnect()
            await connection.close()

    def __init__(
        self,
        *,
        connection: RealtimeApiConnection,
        tools: ToolContext | None,
        channel: Channel,
    ) -> None:
        self.connection = connection
        self.tools = tools
        self._client_tool_futures = {}
        self.channel = channel
        self.subscribe_users = []
        self.write_pcm = os.environ.get("WRITE_AGENT_PCM", "false") == "true"
        logger.info(f"Write PCM: {self.write_pcm}")
        self.muted=False
        # 创建多用户音频流管理器
        self.multi_user_audio_stream = MultiUserAudioStream(channel)

    async def run(self) -> None:
        try:

            def log_exception(t: asyncio.Task[Any]) -> None:
                if not t.cancelled() and t.exception():
                    logger.error(
                        "unhandled exception",
                        exc_info=t.exception(),
                    )

            async def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
                logger.info(f"Received stream message data {data} with length: {length}")
                await self._process_stream_message(user_id, stream_id, data, length)
                

            self.channel.on("stream_message", on_stream_message)

            logger.info("Waiting for remote user to join")
            
            any_user = await wait_for_remote_user(self.channel)
            self.subscribe_users.append(any_user)
            logger.info(f"Subscribing to user {any_user}")
            self.channel.local_user.subscribe_all_audio()
            await self.channel.subscribe_audio(any_user)
            
            await self.multi_user_audio_stream.update_users(self.subscribe_users)
            # 启动音频流处理
            await self.multi_user_audio_stream.start()

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if user_id in self.subscribe_users:
                    self.subscribe_users.remove(user_id)
                    logger.info(f"Subscribed user {user_id} left, removed from user list")
                    await self.multi_user_audio_stream.update_users(self.subscribe_users)
                    if len(self.subscribe_users) == 0:
                        logger.info("No more users to subscribe to, disconnecting")
                        await self.channel.disconnect()

            self.channel.on("user_left", on_user_left)
            
            async def on_user_joined(
                agora_rtc_conn: RTCConnection, user_id: int
            ):
                logger.info(f"User joined: {user_id}")
                if user_id not in self.subscribe_users:
                    self.subscribe_users.append(user_id)
                    logger.info(f"Subscribing to user {user_id}")
                    await self.channel.subscribe_audio(user_id)
                    logger.info(f"Subscribed to user {user_id}, current users: {self.subscribe_users}")
                    await self.multi_user_audio_stream.update_users(self.subscribe_users)
                    
            self.channel.on("user_joined", on_user_joined)

            disconnected_future = asyncio.Future[None]()

            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)

            self.channel.on("connection_state_changed", callback)

            asyncio.create_task(self.rtc_to_model()).add_done_callback(log_exception)
            asyncio.create_task(self.model_to_rtc()).add_done_callback(log_exception)

            asyncio.create_task(self._process_model_messages()).add_done_callback(
                log_exception
            )

            await disconnected_future
            logger.info("Agent finished running")
        except asyncio.CancelledError:
            await self.multi_user_audio_stream.stop()  # 确保停止音频流处理
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise
        
    def _get_audio_streams_for_users(self) -> (list[AudioStream]|None):
        audio_streams = []
        has_any_audio = False 
        for user_id in self.subscribe_users:
            audio_stream=self.channel.get_audio_frames(user_id)
            print(f"user_id:{user_id}, audio_stream:{audio_stream}")
            if audio_stream is not None:
                audio_streams.append(audio_stream)
                has_any_audio = True
                
        if not has_any_audio:
            return None  
        return audio_streams
    
    async def _process_stream_message(self,  user_id, stream_id, data, length) -> None:
        try:
          await self.connection.send_text(data)
        except Exception as e:
            logging.error(f"Error Sending stream message: {e}")
            
            
    def _handle_parsed_stream_message(self, json_data: dict[str, Any]) -> None:
        if "t2i_condition" in json_data:
            self.tools.t2i_condition=json_data["t2i_condition"]
            
            
    async def rtc_to_model(self) -> None:
        # 等待至少有一个用户的音频流可用
        while not self.multi_user_audio_stream.has_streams():
            await asyncio.sleep(0.1)
        # while len(self.subscribe_users) == 0 or self._get_audio_streams_for_users() is None:
        #     await asyncio.sleep(0.1)
            
        #combined_audio_stream = self._get_audio_streams_for_users()
        #print(f"current users:{self.subscribe_users}, audio_streams:{combined_audio_stream}")
        
        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)
        
        try:
            
             # 持续处理音频帧
            while True:
                audio_frame = await self.multi_user_audio_stream.get_next_frame()
                
                if audio_frame is None:
                    # 如果没有可用的音频帧，稍等片刻再尝试
                    await asyncio.sleep(0.01)
                    continue
                
                # 处理接收到的音频
                _monitor_queue_size(self.audio_queue, "audio_queue")
                await self.connection.send_audio_data(audio_frame.data)
                
                # 写入 PCM 数据
                await pcm_writer.write(audio_frame.data)
                
                # 让出控制权以允许其他任务运行
                await asyncio.sleep(0)
            
            # for stream in combined_audio_stream:
            #     async for audio_frame in stream:
            #         # Process received audio (send to model)
            #         _monitor_queue_size(self.audio_queue, "audio_queue")
            #         await self.connection.send_audio_data(audio_frame.data)

            #         # Write PCM data if enabled
            #         await pcm_writer.write(audio_frame.data)

            #         await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the exception to propagate cancellation

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)

        try:
            while True:
               
                # Get audio frame from the model output
                frame = await self.audio_queue.get()
                
                if not self.muted:
                    # Process sending audio (to RTC)
                    await self.channel.push_audio_frame(frame)
                    
                    
                # Write PCM data if enabled
                await pcm_writer.write(frame)

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task

    async def handle_funtion_call(self, message: ResponseFunctionCallArgumentsDone) -> None:
        function_call_response = await self.tools.execute_tool(message.name, message.arguments)
        logger.info(f"Function call response: {function_call_response}")
        
        await self.connection.send_request(
            ItemCreate(
                item = FunctionCallOutputItemParam(
                    call_id=message.call_id,
                    output=function_call_response.json_encoded_output
                )
            )
        )
        
        await self.connection.send_request(
            ResponseCreate()
        )
 
    async def _process_model_messages(self) -> None:
        async for message in self.connection.listen():
            # logger.info(f"Received message {message=}")
            match message:
                case ResponseAudioDelta():
                    # logger.info("Received audio message")
                    self.audio_queue.put_nowait(base64.b64decode(message.delta))
                    # loop.call_soon_threadsafe(self.audio_queue.put_nowait, base64.b64decode(message.delta))
                    logger.debug(f"TMS:ResponseAudioDelta: response_id:{message.response_id},item_id: {message.item_id}")
                case ResponseAudioTranscriptDelta():
                    # logger.info(f"Received text message {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))

                case ResponseAudioTranscriptDone():
                    logger.info(f"Text message done: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                case InputAudioBufferSpeechStarted():
                    await self.channel.clear_sender_audio_buffer()
                    # clear the audio queue so audio stops playing
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info(f"TMS:InputAudioBufferSpeechStarted: item_id: {message.item_id}")
                case InputAudioBufferSpeechStopped():
                    logger.info(f"TMS:InputAudioBufferSpeechStopped: item_id: {message.item_id}")
                    pass
                case ItemInputAudioTranscriptionDelta():
                    pass
                case ItemInputAudioTranscriptionCompleted():
                    logger.info(f"ItemInputAudioTranscriptionCompleted: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                #  InputAudioBufferCommitted
                case InputAudioBufferCommitted():
                    pass
                case ItemCreated():
                    pass
                # ResponseCreated
                case ResponseCreated():
                    pass
                # ResponseDone
                case ResponseDone():
                    pass

                # ResponseOutputItemAdded
                case ResponseOutputItemAdded():
                    pass

                # ResponseContenPartAdded
                case ResponseContentPartAdded():
                    pass
                # ResponseAudioDone
                case ResponseAudioDone():
                    pass
                # ResponseContentPartDone
                case ResponseContentPartDone():
                    pass
                # ResponseOutputItemDone
                case ResponseOutputItemDone():
                    pass
                case SessionUpdated():
                    pass
                case RateLimitsUpdated():
                    pass
                case ResponseFunctionCallArgumentsDone():
                    asyncio.create_task(
                        self.handle_funtion_call(message)
                    )
                case ResponseFunctionCallArgumentsDelta():
                    pass

                case _:
                    logger.warning(f"Unhandled message {message=}")
