# Intermediate animation teacher and compact feature distillation

LIFORMA-104 — frozen pilot protocol, 2026-11-11.

Goal: test whether animation features from a stronger same-family predictor teach the compact network details that reconstruction/contact objectives alone miss. The earlier large-teacher speech-encoder auxiliary experiments are different. Inspired by EA's intermediate-teacher supervision (https://arxiv.org/html/2507.18352v1, section 3.2), adapted to Liforma's temporal convolutional network.

Current reference is the verified `finetune-r4` compact checkpoint, SHA256 ef64e331e226ac23bbf07ad5566f818c49d5a850df4a3bb39941526883d4d4da. Do not modify its weights, sandbox default, 2.5D mapping, Player, assets or deployment. An isolated sandbox worktree based on a00ff68 reuses newer research scripts and existing expanded-corpus caches read-only; primary checkout contains unrelated uncommitted work.

1. Test width-192 initialization preserves baseline predictions and has independent, nonreplicated extra channels. Train 7,440 supervised updates with original animation losses and balanced eight-voice / historical replay sampling. Select by development objective with epoch zero eligible.
2. Continue only if selected teacher improves development gap RMS by at least 1% and macro F1 declines by no more than .005. This screen does not establish generalization.
3. Freeze the teacher. For seeds 37/73/101, train matched width-96 controls and distilled students for 3,720 updates from the current baseline. Same sample stream, optimizer, schedule, train-only projection initialization and checkpoint criterion. Distillation alone adds masked cosine matching of two high-level animation features, weight ramp 0→.1 over 372 updates.
4. Lock all exports and sampler hashes before held-out scoring. Existing feature/scale/gain held-out cohorts are regression screens, not fresh confirmation. Primary pilot threshold: 3% mean gap reduction versus matched controls, no mean contact F1 loss, no engine mean regression, no opening error or timed-closure-total regression. Historical mean RMS must remain within 2% of baseline. Report each seed, not only the best.
5. Only an encouraging result warrants new independent recordings and matched visual review. Nothing promotes automatically. If the teacher fails the initial screen, report it and stop the feature trial rather than choose another teacher based on test scores.

All provenance and immutable source/input/model hashes are in ignored `.local/intermediate-teacher-v1/protocol.json`. Research entry points and input environment variables are documented in `scripts/research/intermediate_teacher/README.md`. No app dependencies or private credentials are copied. Per-run checkpoints preserve states for recovery; no automatic resume is claimed.


## Completed pilot result — 2026-11-11

The teacher passed the development gate: gap RMS 0.242181→0.226962 source units (~6.28% lower), contact F1 0.821219→0.832234, selected at step 6,200. Its added channels remained distinct (median best correlation to original channels ~0.44/0.46 at the two feature layers) and gained nonzero output weights. The frozen width-192 export is 1,251,879 learned parameters / 5,935,208 bytes, SHA256 679e18bf1f42f31fc7144642b07ce432ab6de2186f3365e21ebc82645710ce11. It is an offline teacher only.

All six compact runs finished. Paired sampler/initialization hashes match. Each compact export remains 331,047 parameters / 2,251,827 bytes with the original graph structure. The three-seed mean primary gap reduction over matched controls is only **0.0639%**, while mean contact F1 changes **−0.001291**. Secondary scale/history reductions are 0.0282%/0.3512%. Well-timed closure totals change by +4/+1/0 for seeds 37/73/101, but opening/contact and engine guards fail. The predeclared 3% gap threshold fails. This recipe is not a useful overall improvement; do not promote it.

Primary cohort: 96 previously inspected held-out recordings; secondary cohorts: 352 scale and 48 historical original-condition recordings. These are regression cohorts, not newly independent confirmation. No fresh recording or full-avatar preference study was pursued after the failed screen. This is a negative result for this specific two-layer cosine/projection recipe, not evidence that all forms of intermediate-teacher distillation fail.

Five boundary tests and the separate end-to-end encoder-gradient/frozen-teacher diagnostic pass. All seven native MLX/waveform-ONNX export parity checks pass, max coefficient error 6.56e-7 on three training clips per export. Final integrity audit covers the protected primary model / Player speech-shape sources, frozen input arrays and training sources. No model, 2.5D mapping, Player, default URL or deployment was changed. The first parity diagnostic's NumPy/Accelerate matmul warnings were resolved with explicit finite-checked reduction; accepted parity-v2 reports are warning-free and the earlier diagnostic source/report are preserved.

Readable report and deterministic gap plots are in the chat's outputs directory; exact protocol, selections, per-clip evaluations, checkpoints, logs and source hashes remain in `.local/intermediate-teacher-v1`. Experiment source stays isolated on research/intermediate-animation-teacher.
