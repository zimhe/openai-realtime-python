import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator
import numpy as np

from agora_realtime_ai_api.rtc import AudioStream, Channel, PcmAudioFrame
import struct

from .logger import setup_logger

logger = setup_logger(name=__name__, log_level=logging.INFO)

class MultiUserAudioStream:
    """管理多个用户的音频流并支持动态更新"""
    
    def __init__(self, channel: Channel):
        self.channel = channel
        self.user_streams: Dict[int, AudioStream] = {}
        self._combined_stream = None
        self._update_event = asyncio.Event()
        self._running = True
        self.mixer = WeightedPCMFrameMixer()
        
    async def start(self) -> None:
        """启动音频流处理任务"""
        self._running = True
        asyncio.create_task(self._stream_updater())
        
    async def stop(self) -> None:
        """停止音频流处理"""
        self._running = False
        self._update_event.set()
        
    async def update_users(self, user_ids: List[int]) -> None:
        """更新用户列表并重建音频流"""
        logger.info(f"Updating audio streams for users: {user_ids}")
        
        # 移除不再需要的用户
        to_remove = set(self.user_streams.keys()) - set(user_ids)
        for user_id in to_remove:
            logger.info(f"Removing audio stream for user: {user_id}")
            del self.user_streams[user_id]
            
        # 添加新用户
        for user_id in user_ids:
            if user_id not in self.user_streams:
                audio_stream = self.channel.get_audio_frames(user_id)
                if audio_stream is not None:
                    logger.info(f"Adding audio stream for user: {user_id}")
                    self.user_streams[user_id] = audio_stream
                else:
                    logger.warning(f"Could not get audio stream for user: {user_id}")
                    
        # 通知更新事件
        self._update_event.set()
        
    
        
    def has_streams(self) -> bool:
        """检查是否有可用的音频流"""
        return len(self.user_streams) > 0
    
    async def get_next_frame(self) -> Optional[PcmAudioFrame]:
        """
        获取当前时间点所有用户的一帧音频，混合后返回
        """
        if not self.has_streams():
            return None

        frames_to_mix = []

        for user_id, stream in list(self.user_streams.items()):
            try:
                if not stream.queue.empty():
                    frame = stream.queue.get_nowait()
                    if frame and frame.data:
                        frames_to_mix.append(frame.data)
            except asyncio.QueueEmpty:
                continue
            except Exception as e:
                logger.warning(f"Error reading frame from user {user_id}: {e}")
                continue

        if len(frames_to_mix) == 0:
            return None
        
        if len(frames_to_mix) == 1:
            # 只有一个音频流，直接返回
            return PcmAudioFrame(data=frames_to_mix[0]) 

        # 混音
        mixed_pcm = self.mixer.mix(frames_to_mix)

        return PcmAudioFrame(data=mixed_pcm)


    
    async def _stream_updater(self) -> None:
        """后台任务，监听更新事件"""
        while self._running:
            await self._update_event.wait()
            self._update_event.clear()
            logger.debug(f"Audio streams updated, current users: {list(self.user_streams.keys())}")
            
    def _calculate_rms(self, pcm_data: bytes) -> float:
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    
    


class WeightedPCMFrameMixer:
    def __init__(self):
        pass

    def _calculate_rms(self, pcm_data: bytes) -> float:
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))

    def mix(self, pcm_frames: List[bytes]) -> bytes:
        if not pcm_frames:
            return b""

        num_frames = len(pcm_frames)
        frame_length = len(pcm_frames[0])

        decoded = [
            struct.unpack("<" + "h" * (frame_length // 2), frame)
            for frame in pcm_frames
        ]

        weights = [max(1, self._calculate_rms(frame)) for frame in pcm_frames]
        total_weight = sum(weights)

        mixed = []
        for i in range(len(decoded[0])):
            weighted_sum = sum(decoded[j][i] * weights[j] for j in range(num_frames))
            sample = int(weighted_sum // total_weight)
            sample_clipped = max(-32768, min(32767, sample))
            mixed.append(sample_clipped)

        return struct.pack("<" + "h" * len(mixed), *mixed)
