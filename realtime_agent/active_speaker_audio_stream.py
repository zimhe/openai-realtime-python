import asyncio
import logging
import time
import random
from typing import Dict, List, Optional

import numpy as np
from agora_realtime_ai_api.rtc import AudioStream, Channel, PcmAudioFrame
from .logger import setup_logger

logger = setup_logger(name=__name__, log_level=logging.INFO)

class ActiveSpeakerAudioStream:
    def __init__(self, channel: Channel):
        self.channel = channel
        self.user_streams: Dict[int, AudioStream] = {}
        self._update_event = asyncio.Event()
        self._running = True

        self.active_speaker_id: Optional[int] = None
        self.speaker_rms_history: Dict[int, List[tuple]] = {}

        # === 参数配置 ===
        self.ACTIVE_SPEAKER_WINDOW = 1.5  # RMS 滑动时间窗口（秒）
        self.MIN_DOMINANCE_DURATION = 1.0  # 说话人需主导持续时长（秒）
        self.SAMPLING_INTERVAL = 0.1       # RMS 采样间隔（秒）
        self.SPEAKER_LOCK_DURATION = 2.0   # 说话人锁定时长，避免频繁切换
        self.last_switch_time = 0.0        # 上次切换活跃说话者时间戳
        self.SOFT_SELECTION_TEMPERATURE = 1.0  # softmax 平滑程度，越小越偏向最大值

    async def start(self) -> None:
        # 启动音频流更新监听任务
        self._running = True
        asyncio.create_task(self._stream_updater())

    async def stop(self) -> None:
        # 停止更新流程
        self._running = False
        self._update_event.set()

    async def update_users(self, user_ids: List[int]) -> None:
        # 同步活跃用户列表，增删用户音频流
        logger.info(f"Updating audio streams for users: {user_ids}")

        # 移除不再存在的用户
        to_remove = set(self.user_streams.keys()) - set(user_ids)
        for user_id in to_remove:
            logger.info(f"Removing audio stream for user: {user_id}")
            del self.user_streams[user_id]

        # 添加新用户音频流
        for user_id in user_ids:
            if user_id not in self.user_streams:
                audio_stream = self.channel.get_audio_frames(user_id)
                if audio_stream is not None:
                    logger.info(f"Adding audio stream for user: {user_id}")
                    self.user_streams[user_id] = audio_stream
                else:
                    logger.warning(f"Could not get audio stream for user: {user_id}")

        self._update_event.set()

    def has_streams(self) -> bool:
        # 检查是否存在任何用户音频流
        return len(self.user_streams) > 0

    async def get_next_frame(self) -> Optional[PcmAudioFrame]:
        # 生成下一帧输出：采集所有用户帧，判断活跃说话人，并返回其音频帧
        if not self.has_streams():
            return None

        frame_rms_data = {}
        now = time.time()

        for user_id, stream in list(self.user_streams.items()):
            try:
                if not stream.queue.empty():
                    frame = stream.queue.get_nowait()
                    if frame and frame.data:
                        # 计算 RMS，记录历史
                        rms = self._calculate_rms(frame.data)
                        frame_rms_data[user_id] = (frame.data, rms)
                        self.speaker_rms_history.setdefault(user_id, []).append((now, rms))

                        # 清除过期 RMS 记录
                        self.speaker_rms_history[user_id] = [
                            (t, r) for t, r in self.speaker_rms_history[user_id]
                            if now - t <= self.ACTIVE_SPEAKER_WINDOW
                        ]
            except Exception as e:
                logger.warning(f"Error reading frame from user {user_id}: {e}")

        # 评估是否更新活跃说话人
        self._update_active_speaker()

        # 返回当前活跃说话人的音频帧
        if self.active_speaker_id and self.active_speaker_id in frame_rms_data:
            return PcmAudioFrame(data=frame_rms_data[self.active_speaker_id][0])
        return None

    def _update_active_speaker(self):
        # 根据 softmax 概率决定新的活跃说话人（带锁定与持续性判断）
        now = time.time()
        avg_rms_scores = {
            user_id: np.mean([r for t, r in history])
            for user_id, history in self.speaker_rms_history.items() if history
        }

        if not avg_rms_scores:
            return

        selected_speaker = None

        # 若只有一个用户，直接选取
        if len(avg_rms_scores) == 1:
            selected_speaker = next(iter(avg_rms_scores))
        else:
            user_ids = list(avg_rms_scores.keys())
            rms_raw = np.array([avg_rms_scores[uid] for uid in user_ids], dtype=np.float32)
            rms_total = np.sum(rms_raw)
            if rms_total == 0:
                logger.warning("Total RMS is zero; skipping active speaker update.")
                return
            rms_values = rms_raw / rms_total  # 归一化 RMS，避免数值过大

            if not np.all(np.isfinite(rms_values)):
                logger.warning(f"Invalid RMS values detected: {rms_values}")
                return

            
            shifted = rms_values / self.SOFT_SELECTION_TEMPERATURE
            exp_values = np.exp(shifted)
            total = np.sum(exp_values)

            if not np.isfinite(total) or total <= 0:
                logger.warning(f"Invalid softmax denominator: {total}")
                return

            # 将反比概率翻转为正比，越大 RMS 越大概率
            probs = exp_values / total
            
            selected_speaker = random.choices(user_ids, weights=probs, k=1)[0]

        # 若选说话人失败则跳过
        if selected_speaker is None:
            return

        # 若与当前活跃者不同，判断是否可切换
        if self.active_speaker_id != selected_speaker:
            # if now - self.last_switch_time < self.SPEAKER_LOCK_DURATION:
            #     return  # 锁定期中，忽略切换

            # 判断是否持续主导发声
            dominant_duration = sum(
                1 for t, _ in self.speaker_rms_history[selected_speaker]
                if now - t <= self.MIN_DOMINANCE_DURATION
            ) * self.SAMPLING_INTERVAL

            if dominant_duration >= self.MIN_DOMINANCE_DURATION:
                logger.info(f"Active speaker changed to user: {selected_speaker}")
                self.active_speaker_id = selected_speaker
                self.last_switch_time = now

    async def _stream_updater(self) -> None:
        # 后台协程，响应用户列表更新请求
        while self._running:
            await self._update_event.wait()
            self._update_event.clear()
            logger.debug(f"Audio streams updated, current users: {list(self.user_streams.keys())}")

    def _calculate_rms(self, pcm_data: bytes) -> float:
        # 计算单帧音频的 RMS 值（能量估计）
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2))
