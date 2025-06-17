import asyncio
import base64
import json
import logging
import math
import os
from typing import Any, AsyncIterator

from agora.rtc.agora_base import (
    AudioScenarioType,
    ChannelProfileType,
    ClientRoleType,
)
from agora.rtc.agora_service import (
    AgoraService,
    AgoraServiceConfig,
    RTCConnConfig,
)
from agora.rtc.audio_frame_observer import AudioFrame, IAudioFrameObserver
from agora.rtc.audio_pcm_data_sender import PcmAudioFrame
from agora.rtc.local_user import LocalUser
from agora.rtc.local_user_observer import IRTCLocalUserObserver
from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
from pyee.asyncio import AsyncIOEventEmitter

from agora_realtime_ai_api.rtc import AudioStream, RtcOptions,RtcEngine
from agora_realtime_ai_api.rtc import Chat,ChatMessage
from agora_realtime_ai_api.rtc import Channel,ChannelEventObserver


from agora_realtime_ai_api.token_builder.realtimekit_token_builder import RealtimekitTokenBuilder

from agora_realtime_ai_api.logger import setup_logger

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

class MixedChannelEventObserver(
    IRTCConnectionObserver, IRTCLocalUserObserver, IAudioFrameObserver
):
    def __init__(self, event_emitter: AsyncIOEventEmitter, options: RtcOptions) -> None:
        self.loop = asyncio.get_event_loop()
        self.emitter = event_emitter
        self.audio_streams = dict[int, AudioStream]()
        self.mixed_audio_stream=AudioStream()
        self.options = options
        self.active_speaker=None

    def emit_event(self, event_name: str, *args):
        """Helper function to emit events."""
        self.loop.call_soon_threadsafe(self.emitter.emit, event_name, *args)

    def on_connected(
        self, agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason
    ):
        logger.info(f"Connected to RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connection_state_changed", agora_rtc_conn, conn_info, reason)

    def on_disconnected(
        self, agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason
    ):
        logger.info(f"Disconnected from RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connection_state_changed", agora_rtc_conn, conn_info, reason)

    def on_connecting(
        self, agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason
    ):
        logger.info(f"Connecting to RTC: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connection_state_changed", agora_rtc_conn, conn_info, reason)

    def on_connection_failure(self, agora_rtc_conn, conn_info, reason):
        logger.error(f"Connection failure: {agora_rtc_conn} {conn_info} {reason}")
        self.emit_event("connection_state_changed", agora_rtc_conn, conn_info, reason)

    def on_user_joined(self, agora_rtc_conn: RTCConnection, user_id):
        logger.info(f"User joined: {agora_rtc_conn} {user_id}")
        self.emit_event("user_joined", agora_rtc_conn, user_id)

    def on_user_left(self, agora_rtc_conn: RTCConnection, user_id, reason):
        logger.info(f"User left: {agora_rtc_conn} {user_id} {reason}")
        self.emit_event("user_left", agora_rtc_conn, user_id, reason)

    def on_stream_message(
        self, agora_local_user: LocalUser, user_id, stream_id, data, length
    ):
        # logger.info(f"Stream message", agora_local_user, user_id, stream_id, length)
        self.emit_event("stream_message", agora_local_user, user_id, stream_id, data, length)

    def on_stream_message_error(self, agora_rtc_conn, user_id_str, stream_id, code, missed, cached):
        logger.warn(f"Stream message error: {user_id_str} {stream_id} {code} {missed} {cached}")


    def on_audio_subscribe_state_changed(
        self,
        agora_local_user,
        channel,
        user_id,
        old_state,
        new_state,
        elapse_since_last_state,
    ):
        logger.info(
            f"Audio subscribe state changed: {user_id} {new_state} {elapse_since_last_state}"
        )
        self.emit_event(
            "audio_subscribe_state_changed",
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        )

    def on_playback_audio_frame_before_mixing(
        self, agora_local_user: LocalUser, channelId, uid, frame: AudioFrame, vad_result_state:int, vad_result_bytearray:bytearray
    ):
        audio_frame = PcmAudioFrame()
        audio_frame.samples_per_channel = frame.samples_per_channel
        audio_frame.bytes_per_sample = frame.bytes_per_sample
        audio_frame.number_of_channels = frame.channels
        audio_frame.sample_rate = self.options.sample_rate
        audio_frame.data = frame.buffer

        # print(
        #     "on_playback_audio_frame_before_mixing",
        #     audio_frame.samples_per_channel,
        #     audio_frame.bytes_per_sample,
        #     audio_frame.number_of_channels,
        #     audio_frame.sample_rate,
        #     len(audio_frame.data),
        # )
        
        self.loop.call_soon_threadsafe(
            self.audio_streams[uid].queue.put_nowait, audio_frame
        )
        return 0

    def on_mixed_audio_frame(self, agora_local_user, channelId, frame):
        
        try:
            audio_frame = PcmAudioFrame()
            audio_frame.samples_per_channel = frame.samples_per_channel
            audio_frame.bytes_per_sample = frame.bytes_per_sample
            audio_frame.number_of_channels = frame.channels
            audio_frame.sample_rate = self.options.sample_rate
            audio_frame.data = frame.buffer
            
            self.loop.call_soon_threadsafe(
                self.mixed_audio_stream.queue.put_nowait, audio_frame
            )
              
        except Exception as e:
            logger.error(f"Error processing mixed audio frame: {e}")
            return 1

        return 0
        
    # def on_active_speaker(self, agora_local_user, userId):
    #     self.active_speaker = userId
    #     logger.info(f"Active speaker changed: {userId}")
    #     #return super().on_active_speaker(agora_local_user, userId)

class MixedChannel(Channel):
    def __init__(self, rtc: "RtcEngine", options: RtcOptions) -> None:
        self.loop = asyncio.get_event_loop()
        self.stream_message_queue = asyncio.Queue()

        # Create the event emitter
        self.emitter = AsyncIOEventEmitter(self.loop)
        self.connection_state = 0
        self.options = options
        self.remote_users = dict[int, Any]()
        self.rtc = rtc
        self.chat = Chat(self)
        self.channelId = options.channel_name
        self.uid = options.uid
        self.enable_pcm_dump = options.enable_pcm_dump
        self.token = options.build_token(rtc.appid, rtc.appcert) if rtc.appcert else ""
        conn_config = RTCConnConfig(
            client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
        )
        self.connection = self.rtc.agora_service.create_rtc_connection(conn_config)

        self.channel_event_observer = MixedChannelEventObserver(
            self.emitter,
            options=options,
        )
        self.connection.register_observer(self.channel_event_observer)

        self.local_user = self.connection.get_local_user()
        self.local_user.set_playback_audio_frame_before_mixing_parameters(
            options.channels, options.sample_rate
        )
        self.local_user.set_mixed_audio_frame_parameters(
            options.channels, options.sample_rate,1024
        )
        #self.local_user.set_audio_volume_indication_parameters(200,5,True)
        self.local_user.register_local_user_observer(self.channel_event_observer)
        self.local_user.register_audio_frame_observer(self.channel_event_observer, self.options.enable_vad, self.options.vad_configs)
        # self.local_user.subscribe_all_audio()

        self.media_node_factory = self.rtc.agora_service.create_media_node_factory()
        self.audio_pcm_data_sender = (
            self.media_node_factory.create_audio_pcm_data_sender()
        )
        self.audio_track = self.rtc.agora_service.create_custom_audio_track_pcm(
            self.audio_pcm_data_sender
        )
        self.audio_track.set_enabled(1)
        self.local_user.publish_audio(self.audio_track)

        self.stream_id = self.connection.create_data_stream(False, False)
        self.received_chunks = {}
        self.waiting_message = None
        self.msg_id = ""
        self.msg_index = ""

        self.on(
            "user_joined",
            lambda agora_rtc_conn, user_id: self.remote_users.update({user_id: True}),
        )
        
        def handle_user_left(agora_rtc_conn, user_id, reason):
            if user_id in self.remote_users:
                self.remote_users.pop(user_id, None)
            if user_id in self.channel_event_observer.audio_streams:
                audio_stream = self.channel_event_observer.audio_streams.pop(user_id, None)
                audio_stream.queue.put_nowait(None)
                
            if len(self.remote_users) == 0:
                # 如果没有用户了，清空混音流
                self.channel_event_observer.mixed_audio_stream.queue.put_nowait(None)
        
        self.on(
            "user_left",
            handle_user_left,
        )

        def handle_audio_subscribe_state_changed(
            agora_local_user,
            channel,
            user_id,
            old_state,
            new_state,
            elapse_since_last_state,
        ):
            if new_state == 3:  # Successfully subscribed
                if user_id not in self.channel_event_observer.audio_streams:
                    self.channel_event_observer.audio_streams.update(
                        {user_id: AudioStream()}
                    )
                    #self.channel_event_observer.mixed_audio_stream = AudioStream()

        self.on("audio_subscribe_state_changed", handle_audio_subscribe_state_changed)
        self.on(
            "connection_state_changed",
            lambda agora_rtc_conn, conn_info, reason: setattr(
                self, "connection_state", conn_info.state
            ),
        )
        
        
        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logger.error(
                    "unhandled exception",
                    exc_info=t.exception(),
                )

        asyncio.create_task(self._process_stream_message()).add_done_callback(log_exception)


    def get_mixed_audio_frames(self) -> AudioStream:
        """
        获取混合后的playback音频流
        """
        return self.channel_event_observer.mixed_audio_stream


class MixedRtcEngine(RtcEngine):

    def create_channel(self, options: RtcOptions) -> MixedChannel:
        """
        Creates a channel.

        Parameters:
            channelId: The channel ID.
            uid: The user ID.

        Returns:
            Channel: The created channel.
        """
        return MixedChannel(self, options)

