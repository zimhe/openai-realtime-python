import asyncio
from array import array
import functools
from datetime import datetime


def write_pcm_to_file(buffer: bytearray, file_name: str) -> None:
    """Helper function to write PCM data to a file."""
    with open(file_name, "ab") as f:  # append to file
        f.write(buffer)


def generate_file_name(prefix: str) -> str:
    # Create a timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.pcm"


def resample_pcm16(audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
    if not audio_data or source_rate == target_rate:
        return audio_data

    source = array("h")
    source.frombytes(audio_data)
    if not source:
        return b""

    target_length = max(1, int(round(len(source) * target_rate / source_rate)))
    if len(source) == 1:
        repeated = array("h", [source[0]] * target_length)
        return repeated.tobytes()

    step = (len(source) - 1) / max(1, target_length - 1)
    target = array("h")
    for index in range(target_length):
        position = index * step
        left_index = int(position)
        right_index = min(left_index + 1, len(source) - 1)
        ratio = position - left_index
        value = int(source[left_index] * (1.0 - ratio) + source[right_index] * ratio)
        target.append(max(-32768, min(32767, value)))

    return target.tobytes()


class PCMChunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        if data:
            self.buffer.extend(data)

        chunks: list[bytes] = []
        while len(self.buffer) >= self.chunk_size:
            chunks.append(bytes(self.buffer[: self.chunk_size]))
            del self.buffer[: self.chunk_size]
        return chunks

    def clear(self) -> None:
        self.buffer.clear()


class PCMWriter:
    def __init__(self, prefix: str, write_pcm: bool, buffer_size: int = 1024 * 64):
        self.write_pcm = write_pcm
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.file_name = generate_file_name(prefix) if write_pcm else None
        self.loop = asyncio.get_event_loop()

    async def write(self, data: bytes) -> None:
        """Accumulate data into the buffer and write to file when necessary."""
        if not self.write_pcm:
            return

        self.buffer.extend(data)

        # Write to file if buffer is full
        if len(self.buffer) >= self.buffer_size:
            await self._flush()

    async def flush(self) -> None:
        """Write any remaining data in the buffer to the file."""
        if self.write_pcm and self.buffer:
            await self._flush()

    async def _flush(self) -> None:
        """Helper method to write the buffer to the file."""
        if self.file_name:
            await self.loop.run_in_executor(
                None,
                functools.partial(write_pcm_to_file, self.buffer[:], self.file_name),
            )
        self.buffer.clear()
