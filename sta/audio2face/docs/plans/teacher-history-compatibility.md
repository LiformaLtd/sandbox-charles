# Teacher history/noise compatibility: diagnostic only

No training, teacher relabelling, model publication or production change.
Current v4 finetune-r4 stays fixed. Measure whether full-teacher lips depend on
information unavailable to its stateless one-second-window student.

## Controlled inputs

Choose16 four-second suffixes, two per eight existing training presets, from
existing original Phonon/Supertonic base recordings at least four seconds long.
Choose by fixed hash, not model error, with16 distinct normalized suffix texts.
For each voice select two other natural speech prefixes at least four seconds
long, with no suffix-text overlap. Use first four seconds. The same voice's
prefix pair is shared by its two suffixes. No new synthesis is necessary.

Suffix audio is exactly the same across all conditions. Fade its first20ms in
identically, and fade speech-prefix final40ms out (last20ms silent) to avoid an
arbitrary splice click. Four histories: native cold start (no prefix), four
seconds silence, speech prefixA, speech prefixB. All prefix lengths are exact
multiples of the500ms hop, so output/window phase is unchanged.

Full factorial: four histories × noise seeds0/1/2/3 ×16 suffixes. Seed0 is the
current label policy: a deterministic noise tensor reused at every window,
NOT newly sampled noise each inference call. Alternate seeds measure sensitivity;
they are not evidence that current labels vary randomly between runs.

Use untouched teacher weights/identity James/neutral emotion and native animate
priming/state/crop schedule. An offline wrapper returns selected2267 mouth-region
vertices (allXYZ) and the existing nine gaps, with no change to teacher state.
Verify this projection against original geometry before the factorial runs.
Freeze audio/source/model/rig hashes and selection before inspecting differences.

## Separate observable audio from hidden history

For common suffix output0–250ms, a one-second window can still contain prefix
audio: report this separately as an observable-prefix effect. From250ms onward,
corresponding current windows are byte-identical across histories. Save/compare
window hashes to prove this; use only these frames for hidden-history claims.
Primary time bins after common onset: .25–.75,.75–1.25,1.25–2.25,2.25–3.25sec.
Retain common output through3.75sec; the four-second suffix supplies complete
input windows. No end-padding artifact enters these primary bins.

Add two reset-state controls per suffix at seed0: cold start and speechA, zeroing
input_latents every call. They must match after250ms. Resetting is a diagnostic,
not a proposed new teacher or training target. Repeat cold seed0 for every suffix
to measure deterministic numerical variation. Do not feed random arbitrary states;
history conditions use states naturally reached by the actual prefix streams.

Run current stateless student on all four audio histories once. Its corresponding
inputs/outputs must be identical after250ms. Compare its seed0 baseline gap and
surface errors with measured teacher-history/noise differences for scale only.

## Measurements and interpretation

Measure raw teacher-to-teacher dense3D mouth RMS in mm, nine-gap RMS in mm,
contact-frame disagreement (median gap<=.1cm), per-case/time-bin/engine/preset
results and continuous time curves. Measure history with noise fixed, noise with
history fixed, and report interactions. Separate cold→warm silence from warm
speech→warm speech, so priming transients are not confused with speech history.

For these constructed equally weighted histories, report per-frame teacher
variance around its history mean as a conditional stateless MSE floor. This is
only a uniform-mixture diagnostic, NOT an irreducible error bound for production
speech or the existing data distribution. Alternative noise seeds are another
sensitivity calculation, not part of current fixed-seed label randomness.

Interpret effects below0.1mm as small in this probe; effects around0.5mm or more
persisting after identical-window onset would materially motivate a stateful
experiment. Report actual magnitudes/distributions rather than turn these bands
into statistical proof. Compare with current-student error; do not subtract RMS
values as additive error contributions. No seed/ROI/history selection changes
based on results. No stateful model is trained in this diagnostic.
