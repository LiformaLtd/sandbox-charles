# Compact WebGPU activation regression: “packed”

## Reproduction and cause

Reported on both compact v3 and v4 with Phonon and Supertonic. The passage
starts: “Peter and Ben packed a big blue bag, then walked to the village by
the river.” Use the complete 470-character dialogue for reproduction: window
context and generated audio matter. Archive the actual browser PCM; an offline
Phonon invocation did not produce identical audio to the browser invocation.

The shared WebGPU convolution shader used the standard tanh-approximate GELU.
On the tested desktop WebKit/Metal implementation, `tanh` returned NaN for
large positive arguments. An isolated shader probe returned a normal result
at argument 43.656 but NaN at 49.678 and above. Mathematically both are
effectively 1. This is a measured implementation problem; the internal Metal
implementation was not inspected.

The last convolution reached GELU arguments 46.321 in v3 and 88.503 in v4
around “packed.” Invalid activations propagated through the projection and
became all-zero final weights. Checking only that final weights were finite
missed the failure. It produced a brief neutral-face snap during articulation.

Frozen browser-audio results, full streaming crop:

| Audio | v3 zeroed frames before | v4 zeroed frames before | After, both |
| --- | ---: | ---: | ---: |
| Phonon, full passage | 1 | 9 | 0 |
| Supertonic, full passage | 0 | 5 | 0 |
| Phonon first 6 s, gain 0.25 | 0 | 0 | 0 |
| Phonon first 6 s, gain 2 | 4 | 10 | 0 |
| Phonon first 6 s, gain 4 | 91 | 92 | 0 |

The Phonon failures were at 0.900–1.033 s (v3 at 0.933 s), following the
aligned /p/ at 0.88 s. Supertonic v4 failures were at 1.450–1.517 s, with
“packed” aligned at 1.42–1.74 s. The sampled Supertonic recording did not
reproduce the v3 failure; this does not contradict its occurrence in another
stochastic rendering. The amplified cases are numerical stress tests, not
natural-speech quality measurements.

## Fix and validation

Clamp the **tanh argument**, not the convolution output, to [-10, 10]. In
float32 tanh is already saturated there. This preserves the trained activation
while avoiding the failing large-argument path. Do not fill missing poses,
smooth over the defect, or retrain the model to compensate for shader errors.

The fixed runner matched native ONNX Runtime CPU on 492,856 coefficients over
the five cases and both models, with maximum absolute discrepancy below
0.000001. No zeroed or non-finite output frames remained. The same bounded
activation also passes an independent PyTorch GELU comparison over 200,001
float32 inputs spanning [-64, 64]. All ten multivoice research unit tests pass.

The fix changes only `src/compact-conv.wgsl`. Model weights, architecture,
downloads and inference scheduling are unchanged. It adds no GPU buffers or
pipelines; it does not establish a new physical-iPhone memory measurement.
WebGPU remains the default and James continues animating while idle.

Local diagnostic artifacts are in `.local/packed-diagnostic/` (ignored):
browser PCM, TinyAligner timestamps, native reference tracks, isolated shader
probe, before/after GPU tracks, parity JSON and a four-lane review viewer.
`verify_activation_regression.mjs` consumes `parity-cases.json`: each entry
contains `name`, `audio` (16 kHz mono float32 PCM filename), and `reference`
mapping `v3` and `v4` to native float32 52-channel streaming tracks. References
must use the exact same PCM and `audioWindows` centre crop. Serve these files
from a hosted-mode Vite source server and set `DIAGNOSTIC_URL` to run it.
`COMPARE_ORIGINAL=1` additionally requires the archived original runner/shader
alongside the fixtures.

`build_activation_review.py` builds the scientific comparison viewer from the
archived tracks. `verify_packed_deployment.mjs` runs the complete paragraph on
an actual hosted build with both engines and models, inspecting GPU readbacks.
It reads the archived full passage from `.local/packed-diagnostic/text.txt`;
set `DEPLOYMENT_URL`, and optionally `PLAYWRIGHT_MODULE` and
`WEBKIT_EXECUTABLE`, to run against the chosen deployment. This is desktop
WebKit with iPhone viewport/settings, not a physical iPhone test.

Production build: `dpl_2Qz8HhgweQgN34LNjoVntjG2j6Ms`. Before promotion, the
hosted full-passage test passed all four combinations (v3/v4 × Phonon/Supertonic)
with no zeroed centre-crop frames or page errors. Short hosted checks also
confirmed audio, WebGPU as default, model switching and ongoing idle animation.
Both public aliases completed promotion; CLI inspection of the main URL
confirmed this exact deployment ID.
Public demo: https://james-audio2face.vercel.app/?model=compact-v4&sta=webgpu.
The CPU model-quality metrics remain valid. This fix removes an additional
browser implementation defect; it does not eliminate the remaining difference
between compact predictions and the full teacher.
