import abc
import json

from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Literal, Union

from realtime_agent.tools import ToolContext


DEFAULT_OPENAI_VOICE = "marin"
DEFAULT_VOLCENGINE_VOICE = "zh_female_vv_jupiter_bigtts"
OPENAI_REALTIME_VOICES = {
    "alloy",
    "echo",
    "fable",
    "nova",
    "nova_2",
    "nova_3",
    "nova_4",
    "nova_5",
    "onyx",
    "shimmer",
}

ProviderName = Literal["openai", "volcengine", "volcengine_split", "volcengine_duplex"]
TranscriptRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_manual_responses: bool
    supports_tools: bool
    default_voice: str


@dataclass(frozen=True)
class ProviderSessionStarted:
    session_id: str
    model: str | None = None
    type: str = "session.started"


@dataclass(frozen=True)
class ProviderError:
    message: str
    code: str | None = None
    type: str = "error"


@dataclass(frozen=True)
class ProviderSpeechStarted:
    item_id: str | None = None
    type: str = "input_audio.speech_started"


@dataclass(frozen=True)
class ProviderSpeechStopped:
    item_id: str | None = None
    type: str = "input_audio.speech_stopped"


@dataclass(frozen=True)
class ProviderAudioChunk:
    audio: bytes
    item_id: str | None = None
    type: str = "response.audio.delta"


@dataclass(frozen=True)
class ProviderTranscriptDelta:
    role: TranscriptRole
    text: str
    item_id: str | None = None
    type: str = "conversation.transcript.delta"


@dataclass(frozen=True)
class ProviderTranscriptDone:
    role: TranscriptRole
    text: str
    item_id: str | None = None
    type: str = "conversation.transcript.done"


@dataclass(frozen=True)
class ProviderResponseStarted:
    response_id: str | None = None
    type: str = "response.started"


@dataclass(frozen=True)
class ProviderResponseFinished:
    response_id: str | None = None
    type: str = "response.finished"


@dataclass(frozen=True)
class ProviderToolCall:
    name: str
    arguments: str
    call_id: str
    type: str = "response.function_call"


ProviderEvent = Union[
    ProviderSessionStarted,
    ProviderError,
    ProviderSpeechStarted,
    ProviderSpeechStopped,
    ProviderAudioChunk,
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    ProviderResponseStarted,
    ProviderResponseFinished,
    ProviderToolCall,
]


def _serialize_transcript_event(event: ProviderTranscriptDelta | ProviderTranscriptDone) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "item_id": event.item_id,
        "role": event.role,
        "text": event.text,
    }
    if isinstance(event, ProviderTranscriptDelta):
        payload["type"] = (
            "response.audio_transcript.delta"
            if event.role == "assistant"
            else "conversation.item.input_audio_transcription.delta"
        )
        payload["delta"] = event.text
        return payload

    payload["type"] = (
        "response.audio_transcript.done"
        if event.role == "assistant"
        else "conversation.item.input_audio_transcription.completed"
    )
    payload["transcript"] = event.text
    return payload


def provider_event_to_json(event: ProviderEvent) -> str:
    if isinstance(event, (ProviderTranscriptDelta, ProviderTranscriptDone)):
        return json.dumps(_serialize_transcript_event(event), ensure_ascii=False)

    payload = asdict(event)
    if isinstance(event, ProviderAudioChunk):
        payload["audio"] = "<binary>"
    return json.dumps(payload, ensure_ascii=False)


class RealtimeProviderConnection(abc.ABC):
    provider_name: ProviderName
    capabilities: ProviderCapabilities

    @abc.abstractmethod
    async def __aenter__(self) -> "RealtimeProviderConnection":
        raise NotImplementedError

    @abc.abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def start_session(
        self,
        *,
        system_message: str | None,
        voice: str | None,
        tools: ToolContext | None,
        turn_detection: Any | None,
        dialog_id: str | None,
    ) -> ProviderSessionStarted:
        raise NotImplementedError

    @abc.abstractmethod
    async def send_audio_data(self, audio_data: bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def send_text(self, text: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def request_response(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def send_tool_output(self, call_id: str, output: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def interrupt_response(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def listen(self) -> AsyncGenerator[ProviderEvent, None]:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
