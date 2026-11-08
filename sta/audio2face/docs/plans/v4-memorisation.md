# Current v4 tiny-set memorisation diagnostic

Freeze before training. This repeats a previously successful four-clip diagnostic
with current v4 finetune-r4 and a different eight-clip fixed training subset.
Do not represent it as an entirely new hypothesis or held-out quality test.

Select one original-gain recording per eight training voices from scale old-train,
using salted transcript hash `memorisation-v4-73:` and excluding duplicate texts.
Do not select clips using measured error. Retain original frontend maps, one-second
windows, crops, labels, architecture, normalization and lookahead. All trainable
backbone weights fine-tune from current v4. No recurrent state or new capacity.

Two full-batch arms, same data/initialization/optimizer/update budget: existing
coefficient/gap/contact/velocity/timing mixed loss versus pure normalized fitted
coefficient MSE. AdamW, no weight decay, gradient clipping 1.0, seed37, 3,000 steps,
cosine learning rate .0003→.00001. Evaluate step1 and every100. Select lowest
student-to-fitted-target gap RMS on the memorised data; include epoch zero. Save
final and selected measurements to expose optimization/objective trade-offs.
No test/development selection or deployment follows this diagnostic.

Report student versus raw-teacher nine-gap RMS, student versus fitted-target gap
RMS, the fixed fitted-target versus raw-teacher discrepancy, contact F1 and
coefficient error. Report both pooled and voice-macro raw/reference RMS. The
0.373 mm reference previously reported belongs to a different 96-clip cohort;
this subset has its own measured reference. A fitted reference is not a proven
irreducible minimum of raw-gap error. Do not subtract RMS values as independent
error components.

Predeclared closeness screen: selected raw RMS at most 1.25× this subset's fitted
reference. Also report fitted-target error continuously. Success would show
learnability on a small known set, not prove unlimited capacity/generalization,
identify one unique cause of production error, or justify publishing memorised
weights. Any other search/budget requires a separately reported extension.
Export unchanged waveform ONNX and verify original streaming/native parity.
