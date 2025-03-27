import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator

from agora_realtime_ai_api.rtc import AudioStream, Channel, PcmAudioFrame

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
        """获取下一个音频帧，从所有用户流中轮询"""
        if not self.has_streams():
            return None
            
        # 如果需要处理混音，这里可以添加更复杂的逻辑
        # 当前简单实现：从所有用户中轮询获取音频帧
        for user_id, stream in list(self.user_streams.items()):
            try:
                if not stream.queue.empty():
                    frame = stream.queue.get_nowait()
                    if frame is not None:
                        return frame
            except asyncio.QueueEmpty:
                pass
            except Exception as e:
                logger.error(f"Error getting frame from user {user_id}: {e}")
                
        return None
    
    async def _stream_updater(self) -> None:
        """后台任务，监听更新事件"""
        while self._running:
            await self._update_event.wait()
            self._update_event.clear()
            logger.debug(f"Audio streams updated, current users: {list(self.user_streams.keys())}")