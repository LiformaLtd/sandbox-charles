# Closure/timing-protected optimizer-step pilot

The published v4 stays unchanged. Both arms initialize from the prior ordinary
fine-tuning control, SHA256
`63ddc0c64851edc445eba1364e8b160a8a680c4d3b66638fc57a17b782cfe867`.
All331,047 parameters,2,251,827 model bytes, waveform frontend, audio context,
lookahead and production WebGPU runner remain unchanged. This is a training-only
update-rule experiment, not a new decoder, loss-weight sweep or runtime correction.

## Conflicts before training

At the frozen initialization,64 training-only batches use seed1037 and the
existing six-windows-per-eight-voices plus16-replay sampler. Fitting comprises
shape +1.2gap +.12velocity; the two guards are.22contact and.45timing, all exactly
the existing mixed-loss terms. Their sum-gradient matches the full objective.
Fitting/contact cosines are negative on65.625% of batches and fitting/timing on
32.8125%. Each fresh-optimizer one-step probe also records whether the actual
AdamW displacement increases the linearized guard losses. These are local batch
observations, not proof that conflict uniquely explains earlier null results.

Proceed only if at least10% of either conflict type occurs, projection contract
tests pass and at least8/16 real guarded probes permit a nonzero step. Actual
preflight:14/16 nonzero; checked nonlinear guard values stay within tolerance.
The previous source revision only lacked aliases needed by shared evaluation
helpers; its identical preflight is retained under ignored preflight-v1.

## Protected arm

Both arms retain the same total mixed objective, optimizer, schedule, clipping,
main batches and development selector. The control takes the ordinary AdamW step.
The protected arm additionally computes contact/timing gradients at the same old
parameters and projects the **actual proposed parameter displacement**, including
Adam adaptation, momentum and weight decay, onto two halfspaces requiring
nonpositive first-order change in these losses. This is an asymmetric two-guard
step projection, inspired by gradient-conflict methods, not standard symmetric
PCGrad or a theorem about held-out F1.

A two-constraint active-set calculation chooses the nearest feasible displacement.
Numerical feasibility failure falls back to zero. Verify known solutions and
nonexpansive/feasibility properties. Then evaluate both actual weighted guard
losses on the same batch. Try scales1,.5,.25,.125,.0625; accept the first with
neither guard increasing beyond1e-7×max(1,abs(old loss)), otherwise skip the
parameter change. Scaling cannot increase the projected displacement norm.

Adam moments and step counters retain the attempted raw-gradient update even
when the parameter displacement is shortened or skipped. Checkpoints preserve
parameters, optimizer state, main RNG, best model and complete guard history.
Logs include raw/projected/accepted displacement norms, before/after guard dots,
old/new guard losses, accepted scale and skip/projection rates. Do not equate
attempted updates with equal accepted parameter movement.

These constraints protect aggregate **training-batch surrogate losses**. They do
not guarantee per-voice contact counts, absence of false closures or preservation
of held-out events. Those require the independent quality checks below.

## Matched budget and selection

One seed37 pilot:3,720 scheduled optimizer updates, all existing primary/replay
training data, six windows from each of eight voices plus sixteen replay, shared
cosine AdamW learning rate. Both arms use identical starting weights, sampler,
clip norm1, betas.9/.999, weight decay.015 and eps1e-8. Protection may reduce/zero
accepted motion; budgets match scheduled updates/examples, not wall time or
parameter-path length. Both select by the same development macro gap RMS +
.25×(1−contactF1), every124 updates, including initialization. No test-set tuning.

Fresh confirmation is24 disjoint texts×four training-held-out presets:96 original
recordings and independently teacher-labelled original/half/boost/quarter cases.
Exclude all prior planned/manifest texts, including the latest SyncNet cohort,
before synthesis. Freeze both checkpoint selections before scoring this cohort.
Report release v4, initialization, new ordinary control and protected model.
Existing test sets are separately labelled regressions.

## Endpoints and guards

Measure complete primary/replay training fit and contact metrics, including all
gains and original-gain primary separately, with the previous fit evaluator and
waveform ORT parity. Training improvement must be≥3% versus matched control on
original primary speech. The fresh original test must also improve RMS≥3%, with
no engine RMS regression. Preserve macro and engine contact F1 (tolerance1e-6),
no loss of well-timed /m,b,p/ hits individually, no higher macro false-closure or
missed-contact rate (tolerance1e-6), and no worse mean per-phone onset/release
MAE beyond0.1ms. Timing MAE is conditional on overlapping events; interpret it
alongside the protected hit counts, never alone. Retain existing0–100ms
anticipatory event association. Historical/feature-cohort RMS/F1 guards versus
public v4 remain the previous2%/.01 tolerances.

Failing a guard is a failed screen, not permission to retune against inspected
confirmation. This one seed cannot establish a robust universal gain.

Verify identical exported graph/constants/bytes and native/actual WebGPU parity.
Measure three desktop WebKit speed/physical-footprint trials with animated James,
short/long/repeated accelerated prerecorded STA and unchanged GPU runner. TTS,
initial rig loading, network/browser-parent processes are excluded. No claim about
physical-iPhone full-pipeline peaks follows. No automatic publication/deployment.
