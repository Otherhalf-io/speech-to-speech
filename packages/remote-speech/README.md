# Headless OpenAI-compatible speech transport

`HttpSpeechOperation` makes one OpenAI-compatible `/v1/audio/speech` request
without loading the speech-to-speech conversation pipeline. The existing TTS
handler uses its synchronous iterator; an application-owned async service can
use the asynchronous iterator. Both expose response bytes, not decoded audio.

Install the package from `packages/remote-speech` in a separate environment if
you do not need the full application. The main `speech-to-speech` distribution
also includes the module so its existing TTS handler keeps working.

```python
from hf_s2s_remote_speech import HttpSpeechOperation

async def speak(publish_audio_bytes, is_cancelled):
    operation = HttpSpeechOperation(
        endpoint_url="http://127.0.0.1:8000/v1/audio/speech",
        api_key=None,
        payload={
            "model": "my-tts-model",
            "input": "Hello there.",
            "voice": "my-voice",
            "response_format": "pcm",
        },
        timeout_s=10.0,
        response_format="pcm",
    )
    try:
        async for chunk in operation.aiter_bytes(cancel_check=is_cancelled):
            await publish_audio_bytes(chunk)
    finally:
        operation.cancel()
```

`is_cancelled` (a zero-argument predicate) and `publish_audio_bytes` are
application-owned in this example. A `SpeechRequestError.retryable` value
classifies the transport error, not permission to replay speech: retry only
before audio has been published.

`iter_bytes(cancel_check)` offers the same operation to synchronous callers.
Each operation is single-use. The deadline covers headers and the full stream;
the accepted response media type must match the requested format unless an
explicit `accepted_content_types` set is supplied. Transport and protocol
failures become sanitized `SpeechRequestError`s; `SpeechRequestCancelled`
represents caller cancellation. Never log the payload or bearer token.

This component does **not** load a TTS model or choose a voice. The caller owns
request/session identity, audio decoding and framing, retry policy, response
publication, and cancellation of its own user turn. In particular, adopting
this operation does not replace an inference engine such as vLLM-Omni.

For development, run `uv sync --project packages/remote-speech --group dev`,
then `uv run --project packages/remote-speech pytest` and the corresponding
Ruff check. The repository CI also checks the existing TTS handler after the
extraction.
