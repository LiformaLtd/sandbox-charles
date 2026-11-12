# Closure-protected training: negative result

Both arms retain the current 331,047-parameter architecture and 2,251,827-byte
ONNX export. Neither changes production. See the [frozen protocol](compact-closure-protected.md)
and [reproduction guide](../../scripts/research/closure_protected/README.md).
Generated evidence remains under `.local/compact-protected-v1`.

## Design and integrity

Both arms initialize from the preceding ordinary control (SHA256
`63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867`).
They share seed37, 3,720 scheduled updates, batches, replay, loss weights,
AdamW schedule and development selector; initialization is eligible. Both select
step2976. This matches scheduled exposure, not accepted parameter motion or time.

At the fixed initialization, fitting/contact gradients conflict on65.625% of64
training probes, and fitting/timing on32.8125%. However, only8/64 actual first
AdamW proposals violate either guard: pairwise conflict alone overstates the
number of proposals requiring intervention.

The treatment projects the actual AdamW displacement onto contact/timing
non-increase halfspaces, then checks both nonlinear losses on the same batch.
It backtracks through scales1,.5,.25,.125,.0625 or skips parameter motion.
Adam moments and counters retain the attempted update. This is not standard
PCGrad and cannot guarantee unseen closure metrics. Known-solution, random
projection and actual nonlinear-step checks pass. Over the full run1,823 steps
are projected,1,266 backtracked and535 skipped; all recorded accepted steps meet
same-batch tolerances. The sampler hashes match. Complete guard history is saved
atomically with optimizer/model/best/RNG state.

The fresh confirmation set has24 disjoint texts across four held-out voice
presets:96 original recordings,384 separately labelled gain cases. Previous
planned texts are excluded. Corpus and checkpoint locks precede scoring. This
new cohort differs from previous test sets; its absolute RMS must not be compared
directly with the earlier2.44mm score.

## Quality

Original-gain macro gap RMS in millimetres; test F1 and phone hits use the fresh
cohort and retain the established anticipatory closure scorer.

| Model | Training RMS | Test RMS | Contact F1 | /m,b,p/ hits |
|---|---:|---:|---:|---|
| Published v4 | 1.83882 | 2.67389 | .79595 | 36 / 65 / 55 |
| Shared initialization | 1.74204 | 2.62794 | .79797 | 35 / 64 / 54 |
| Ordinary matched control | 1.66929 | 2.58937 | .79979 | 34 / 64 / 52 |
| Protected update | 1.70073 | 2.61176 | .79815 | 34 / 64 / 52 |

The treatment is1.8834% worse on training RMS and.8648% worse on test RMS than
the control. All-primary-gain training RMS is1.72690 control /1.76167 protected.
Both engines worsen: Phonon2.45698→2.47894mm; Supertonic2.72177→2.74458mm.
A2,000-replicate text-cluster bootstrap gives extra improvement relative to
control of−1.1284% to−.5802% (95% interval, conditional on this cohort, presets and
one training seed). It is not evidence across independent training seeds.

False-closure rate worsens .07409→.07583, while missed-contact rate improves
.15772→.15655. Conditional onset MAE improves16.631→16.475ms but release MAE
worsens11.236→11.519ms. Read timing alongside event counts. Existing-cohort dense
surface RMS is2.80344 control /2.81921 protected; it is regression evidence.
The predeclared training, test-gap, engine, F1, false-closure and release guards
fail. Ordinary fine-tuning gains over published v4 are not gains from protection.
Neither arm is promoted.

## Browser and memory

Native/export parity stays within1.02e-6 for the two new models. The unchanged
production WebGPU runner passes three parity trials per model, identical reset
repeats, both-engine/gain examples, short tails and the packed-phrase regression.
All sixteen synchronized review cases pass audio/seek checks; the first case was
also inspected as a screenshot. This is not a blind human quality study.

On the M5 Max64GiB, three warmed WebKit trials give median block times1.10ms
published v4,1.03ms control and1.07ms protected per nominal500ms emitted audio.
These correspond to approximately455×,485× and467× real time. Identical
operations and these small differences do not establish a speed improvement.
Every model allocates1,492,236 explicit GPU bytes across25 buffers.

Three fresh physical-footprint trials per model sum WebContent and GPU processes
at20ms intervals with James continuously animating. Peaks span524–538MiB across
session startup, short/long clips and three accelerated repeats. They exclude
TTS, initial rig loading, network and browser-parent processes. They do not prove
full-pipeline iPhone memory use or a treatment memory saving. Startup medians are
8.56/8.72/8.54ms respectively.

## Artifacts and decision

Control export SHA256:
`97850c5416d0ea90a4f8e328071d277a6d4c7c5810bc69d9110f50a06d02baee`.
Protected export:
`156e1aea3ac1f1b447a63d2a8e1a1ba92cb1352f8092255b8813bfa510f9b29c`.
Training protocol:
`c171eb2a3a061746cbfc8f6499f627e999f69cb47725ba2afa1556aee1945b21`.
`fit-summary.json`, `analysis.json`, model reports and browser trial files retain
the detailed evidence. Reproduce using the linked source guide; keep generated
weights, audio and arrays outside Git.

Reject this update rule. Gradient conflict exists, but enforcing these particular
same-batch contact/timing guards does not solve the training-fit bottleneck.
This does not disprove every constrained or multi-objective optimizer. Preserve
the deployed v4, default WebGPU and continuous idle animation.
