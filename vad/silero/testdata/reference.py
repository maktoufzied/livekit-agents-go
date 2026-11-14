"""Reference values from the upstream Python plugin (livekit-plugins-silero).

Usage: ref.py <clip.pcm> <out.txt>

The clip is mono PCM16 at 16 kHz. Output, one line per item:
  raw <probability>            model output per 512-sample window (OnnxModel)
  ev <type> <samples_index> <speech_duration> <silence_duration> <probability>
                               events of the plugin's VAD stream, fed 20 ms frames
"""

import asyncio
import sys

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.plugins import silero
from livekit.plugins.silero import onnx_model, version


async def main() -> None:
    clip = np.fromfile(sys.argv[1], dtype="<i2")
    lines = [f"# livekit-plugins-silero {version.__version__}"]

    sess = onnx_model.new_inference_session(True)
    m = onnx_model.OnnxModel(onnx_session=sess, sample_rate=16000)
    w = m.window_size_samples
    for i in range(0, len(clip) - w + 1, w):
        x = np.divide(clip[i : i + w], np.iinfo(np.int16).max, dtype=np.float32)
        lines.append(f"raw {m(x):.6f}")

    detector = silero.VAD.load()
    stream = detector.stream()
    for i in range(0, len(clip), 320):
        chunk = clip[i : i + 320]
        stream.push_frame(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
        )
    stream.end_input()
    async for ev in stream:
        kind = {
            agents_vad.VADEventType.START_OF_SPEECH: "start",
            agents_vad.VADEventType.END_OF_SPEECH: "end",
            agents_vad.VADEventType.INFERENCE_DONE: "inference",
        }[ev.type]
        lines.append(
            f"ev {kind} {ev.samples_index} {ev.speech_duration:.3f} "
            f"{ev.silence_duration:.3f} {ev.probability:.6f}"
        )

    with open(sys.argv[2], "w") as f:
        f.write("\n".join(lines) + "\n")
    print(len(lines), "lines")


asyncio.run(main())
