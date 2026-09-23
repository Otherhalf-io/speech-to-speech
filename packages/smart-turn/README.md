# Headless Smart Turn

This package exposes the existing Smart Turn v3.2 ONNX classifier without
loading the speech-to-speech conversation pipeline. The full application keeps
its old `speech_to_speech.VAD.smart_turn` import through a compatibility shim.
Applications with their own microphone, VAD and turn policy can install only
`packages/smart-turn` and its CPU inference dependencies.

```python
import numpy as np
from hf_s2s_smart_turn import SmartTurnAnalyzer

analyzer = SmartTurnAnalyzer(model_path="/models/smart-turn-v3.2-cpu.onnx")
audio = np.asarray(recent_mono_audio, dtype=np.float32)
result = analyzer.predict(audio, sample_rate=16000)
if result.complete:
    # The application, not this classifier, decides when to release the turn.
    schedule_turn_release()
```

`recent_mono_audio` and `schedule_turn_release` are application-owned. Supply
the post-processing waveform that the caller intends to classify. The analyzer
resamples to 16 kHz, keeps at most the latest eight seconds, left-pads shorter
input, extracts Whisper features and returns `complete`, `probability` and
`inference_ms`. `predict` is synchronous CPU work; an async application should
run it off its event loop and reject stale results after cancellation or
resumed speech.

Pass a locally verified `model_path` for a reproducible deployment. Omitting
it downloads the upstream `pipecat-ai/smart-turn-v3` model via Hugging Face
Hub, which is convenient for development but does not pin a revision. The
caller owns model/artifact pinning, VAD boundaries, audio buffering, session
identity, maximum wait and timeout policy. Smart Turn is an end-of-turn
classifier, not a replacement for VAD or STT.

For development, run `uv sync --project packages/smart-turn --group dev`,
then `uv run --project packages/smart-turn pytest` and the corresponding Ruff
check. CI also exercises the full application's compatibility import.
