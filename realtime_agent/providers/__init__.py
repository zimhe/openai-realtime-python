import os

from .openai_provider import OpenAIRealtimeProvider
from .volcengine_provider import VolcengineRealtimeProvider
from .volcengine_split_provider import DEFAULT_SPLIT_TTS_VOICE, VolcengineSplitRealtimeProvider
from .volcengine_duplex_provider import VolcengineDuplexRealtimeProvider

from .base import (
    DEFAULT_OPENAI_VOICE,
    DEFAULT_DUPLEX_VOICE,
    DEFAULT_VOLCENGINE_VOICE,
    OPENAI_REALTIME_VOICES,
    ProviderAudioChunk,
    ProviderCapabilities,
    ProviderError,
    ProviderEvent,
    ProviderName,
    ProviderResponseFinished,
    ProviderResponseStarted,
    ProviderSessionStarted,
    ProviderSpeechStarted,
    ProviderSpeechStopped,
    ProviderToolCall,
    ProviderTranscriptDelta,
    ProviderTranscriptDone,
    RealtimeProviderConnection,
    provider_event_to_json,
)

__all__ = [
    "DEFAULT_OPENAI_VOICE",
    "DEFAULT_DUPLEX_VOICE",
    "DEFAULT_VOLCENGINE_VOICE",
    "ProviderAudioChunk",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderEvent",
    "ProviderName",
    "ProviderResponseFinished",
    "ProviderResponseStarted",
    "ProviderSessionStarted",
    "ProviderSpeechStarted",
    "ProviderSpeechStopped",
    "ProviderToolCall",
    "ProviderTranscriptDelta",
    "ProviderTranscriptDone",
    "RealtimeProviderConnection",
    "create_realtime_provider",
    "get_default_voice",
    "get_realtime_provider_name",
    "normalize_provider_voice",
    "provider_event_to_json",
]

def get_realtime_provider_name() -> ProviderName:
    provider = os.getenv("REALTIME_PROVIDER", "volcengine_split").strip().lower()
    if provider not in {"openai", "volcengine", "volcengine_split", "volcengine_duplex"}:
        raise ValueError(f"Unsupported REALTIME_PROVIDER: {provider}")
    return provider  # type: ignore[return-value]


def get_default_voice(provider: ProviderName) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_VOICE
    if provider == "volcengine_split":
        return DEFAULT_SPLIT_TTS_VOICE
    if provider == "volcengine_duplex":
        return DEFAULT_DUPLEX_VOICE
    return DEFAULT_VOLCENGINE_VOICE


def normalize_provider_voice(provider: ProviderName, voice: str | None) -> str:
    normalized_voice = (voice or "").strip()
    if not normalized_voice:
        return get_default_voice(provider)

    if provider == "volcengine_split" and normalized_voice in OPENAI_REALTIME_VOICES | {DEFAULT_VOLCENGINE_VOICE}:
        return DEFAULT_SPLIT_TTS_VOICE

    if provider == "volcengine_duplex" and normalized_voice in OPENAI_REALTIME_VOICES | {DEFAULT_VOLCENGINE_VOICE}:
        return DEFAULT_DUPLEX_VOICE

    if provider in {"volcengine", "volcengine_split", "volcengine_duplex"} and normalized_voice in OPENAI_REALTIME_VOICES:
        return DEFAULT_VOLCENGINE_VOICE

    return normalized_voice


def create_realtime_provider(*, verbose: bool = False) -> RealtimeProviderConnection:
    provider = get_realtime_provider_name()
    if provider == "openai":
        return OpenAIRealtimeProvider(verbose=verbose)
    if provider == "volcengine_split":
        return VolcengineSplitRealtimeProvider(verbose=verbose)
    if provider == "volcengine_duplex":
        return VolcengineDuplexRealtimeProvider(verbose=verbose)
    return VolcengineRealtimeProvider(verbose=verbose)
