# Learnability diagnostic, then data-and-compute scaling

Published v4 stays unchanged. First determine whether the existing 96-channel
network can fit four known training clips, before generating thousands more.
No architecture, frontend, precision or runtime changes in this phase.

## Frozen memorization screen

Initialize published coverage-r2 (SHA 5f28ddbbaf6f063ab51c90b90c711d2afff7aa1f7fffd1d813813b0a3297ebbd).
Select four original-level training clips deterministically: Phonon Easton /m/,
Phonon Zoey /b/, Supertonic M1 /p/, Supertonic F1 vowels. Both engines and catalogue
genders are represented. Never use held-out data for this diagnostic.

Two independent objectives, same initialization, PCM/labels, full valid-window
batch, seed 37, 1500 AdamW updates, LR .0003 cosine to .00003, no weight decay:
(1) existing mixed coefficient/geometry/contact/velocity/timing loss;
(2) normalized fitted-coefficient MSE alone, testing loss conflict separately
from capacity/optimization. Inspect every 50 steps, retain minimum training
student-to-fitted-gap RMS. This is deliberately training-set selection, not a
quality candidate or generalization result.

A useful fit has <=0.5 mm student-to-fitted-target RMS and <=1.0 mm student-to-raw
teacher RMS on the chosen recordings. Report fitted-target-to-teacher reference,
normalized coefficient error and contact F1. If only coefficient MSE succeeds,
investigate objective conflict before a major data build. If neither succeeds,
investigate representation/optimization before scaling. Do not call failure a
universal architecture limit or irreducible teacher-noise proof.

If both objectives fit accurately, proceed to real data/training scaling, retaining
the current topology. Freeze a separate corpus/compute protocol before synthesis
and test scoring. Compare original/expanded data at original/increased updates,
with common development data, clean replay, untouched confirmation and per-engine
closure/timing guards. Existing gain copies are not distinct recordings. Do not
attribute extra-training gains to data volume, or promote without browser parity,
speed/memory and independent confirmation. Target 3000–5000 new independent TTS
recordings with unchanged engines/voices, broad contexts and no held-out overlap.
