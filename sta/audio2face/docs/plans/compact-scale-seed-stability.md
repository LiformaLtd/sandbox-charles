# Post-pilot fixed-seed stability check

The seed-37 pilot missed its predeclared >=5% RMS improvement screen against the
matched long-budget control: observed 4.4457%, with all other guards passing.
Its `promising=false` result remains unchanged. Versus deployed v4 it improved
6.5178%, and well-timed /m b p/ hits rose 380→435. This secondary signal motivates
a post-pilot stability check, not a lower threshold or a release decision.

Before additional runs, retain the exact two extra seeds already registered in
the original training protocol, 73 and 101. Repeat both original/expanded data
trajectories using exactly the same data, 3720-step schedules, loss, backend,
normalization and development-only checkpoint selection. Do not change any
training source or choose a seed against confirmation. Report all three seeds,
individual and mean RMS/contact/event effects, including regressions. Never
publish only the strongest seed or relabel the initial pilot screen as passed.

Use the existing 352-recording confirmation cohort only to estimate optimizer-
seed stability of this fixed comparison. These recordings are now inspected;
this is not confirmation on a new population of texts or voices. No new recipe
or hyperparameter is selected. The original plan's conditional repeat trigger
was not reached, so this extension is explicitly post-pilot/exploratory.

Published v4 remains unchanged. Even a stable improvement needs reviewed browser
comparisons, native/WebGPU parity and actual-device review before release.
