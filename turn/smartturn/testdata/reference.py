"""Reference values from the Smart Turn reference inference code.

Follows inference.py and audio_utils.py of github.com/pipecat-ai/smart-turn:
Whisper features over the last 8 seconds, then the ONNX model.

Usage, from the repository root:

  uv run --python 3.13 --with onnxruntime==1.32.0 --with transformers==4.48.2 \
      --with numpy python turn/smartturn/testdata/reference.py

Writes turn/smartturn/testdata/reference.txt, one line per case:
  p <clip> <seconds> <probability>     prediction on the first <seconds> of the clip
  q <clip> <seconds> <probability>     the same with the unquantized model
  f <clip> <seconds> <index> <value>   one value of the model input, [mel][frame] order

The q lines need the unquantized model, which is too large for the
repository: download smart-turn-v3.2-gpu.onnx from
huggingface.co/pipecat-ai/smart-turn-v3 and set SMARTTURN_FP32_MODEL to its
path. The embedded model is quantized to 8 bits and its output moves by up
to 0.2 when the input changes in the sixth decimal; the unquantized model
does not, so it is the one that shows the two implementations compute the
same thing.
"""

import os

import numpy as np
import onnxruntime as ort
import transformers
from transformers import WhisperFeatureExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
CLIPS = {
    "speech_16k": "vad/silero/testdata/speech_16k.pcm",
    "apollo11_16k": "vad/silero/testdata/apollo11_16k.pcm",
    "unfinished_16k": "turn/smartturn/testdata/unfinished_16k.pcm",
    "finished_16k": "turn/smartturn/testdata/finished_16k.pcm",
}
# Seconds of each clip given to the model.
CASES = {
    "speech_16k": [0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.3],
    "apollo11_16k": [1.0, 2.0, 2.6, 3.0, 5.0, 6.0, 7.5, 8.2, 10.0, 12.0, 12.98],
    "unfinished_16k": [1.0, 1.5, 2.12],
    "finished_16k": [1.0, 2.0, 3.0, 3.8],
}
FEATURE_STEP = 499  # every 499th input value is recorded



def load(path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so)


session = load(os.path.join(HERE, "..", "smart_turn_v3.2_cpu.onnx"))
fp32 = load(os.environ["SMARTTURN_FP32_MODEL"])
extractor = WhisperFeatureExtractor(chunk_length=8)

lines = [f"# onnxruntime {ort.__version__}, transformers {transformers.__version__}"]
for name, path in CLIPS.items():
    clip = np.fromfile(os.path.join(ROOT, path), dtype="<i2").astype(np.float32) / 32767
    for seconds in CASES[name]:
        audio = clip[: int(round(seconds * 16000))]
        if len(audio) > 8 * 16000:
            audio = audio[-8 * 16000 :]
        else:
            audio = np.pad(audio, (8 * 16000 - len(audio), 0))
        features = extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        probability = session.run(None, {"input_features": features})[0].flatten()[0]
        lines.append(f"p {name} {seconds} {probability:.6f}")
        probability = fp32.run(None, {"input_features": features})[0].flatten()[0]
        lines.append(f"q {name} {seconds} {probability:.6f}")
        flat = features.flatten()
        for i in range(0, len(flat), FEATURE_STEP):
            lines.append(f"f {name} {seconds} {i} {flat[i]:.5f}")

with open(os.path.join(HERE, "reference.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "lines")
