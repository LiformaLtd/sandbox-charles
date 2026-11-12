# Training-only articulation-state supervision

Freeze before training and confirmation scoring. Published v4 coverage-r2 remains
unchanged. Independent width expansion failed; test supervision next, not width.

Both arms start from coverage-r2 SHA
`5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd`.
Keep the deployed 96-channel network, frontend, loss, broader-speech corpus,
historical replay, normalization, seed 37, 24 epochs × 31 updates, learning rate
0.00012 and development checkpoint criterion identical. Both instantiate the
same training modules/RNG sequence; only the auxiliary loss differs.

Control: unchanged geometric objective. State: add a training-only linear
96-to-4 head, cross-entropy weighted 0.1. States derive from teacher lip-gap
geometry (centimetres, 60 fps), not phoneme times: closed if median gap <= 0.1;
otherwise closing/opening if centred one-frame derivative is below -0.025 or
above +0.025; otherwise open/steady. Ignore frames without both valid neighbours,
including padded samples and block edges. Fixed class weights [1,2,2,2]. These
are engineering state labels, not reviewed anatomical ground truth.

Remove the state head before ONNX export: expected 331,047 parameters and
2,251,827 bytes, identical runtime operators, no phoneme aligner, state logic or
additional browser buffers. Verify head gradients and ignored padding in tests.

Freeze 24 fresh texts across eight categories, synthesize four training-held-out
pilot presets (96 new recordings, both engines and catalogue genders), with
teacher-labelled original/half/headroom-boost/quarter levels (384 cases).
Exclude normalized text overlap with all prior known corpora, including the
capacity confirmation. These are new recordings/texts, not unseen voice presets.
Select checkpoints only on shared development data; lock exports before scoring.

Primary screen: fresh original macro-voice lip-gap RMS improves >=5% versus
matched control, no engine-average or contact-F1 regression; half/boost mean RMS
must not regress. Report quarter separately. Historical/coverage original RMS
may regress at most 2% and F1 at most .01 versus published v4. Previously inspected
capacity confirmation is a regression cohort. Report opening error and teacher
contact onset/hold/release timing separately, using unchanged geometry event
association (100 ms lead / 35 ms trail), never shifting reference closures to
phoneme onset. Inspect complete synchronized James mouth tracks if promising.

One seed is exploratory. Independently seeded confirmation is required before
release claims. No deployment in this phase. Check native/WebGPU agreement on
full speech tracks and warm block cost; preserve idle animation. Explicit buffer
counts are not full-page peak memory or physical iPhone measurements.
Artifacts, sources, data and model locks stay under ignored `.local/compact-state-v1`.
