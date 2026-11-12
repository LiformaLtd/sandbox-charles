# Trainable mel-filter bank, LIFORMA-105

Offline experiment only; current model, mapping, Player and deployment untouched. This is spectral-filter adaptation, not a learned phase-sensitive waveform encoder. FFT, Hann window, power, compression, normalization, width96 temporal predictor, 60Hz output and geometry losses are fixed in form. Both predictor copies are fine-tuned with matched updates.

Inputs: `A2F_FRONTEND_INPUT` points to the existing feature-distillation app checkout with expanded data caches and current finetune-r4; `A2F_RESEARCH_WORK` points to the historical `arkit-student*` corpus root. Inputs are opened read-only. New caches/checkpoints use `A2F_FRONTEND_OUTPUT` (default `.local/trainable-frontend-v2`). No shared writable dependencies are installed.

Python3.12, MLX0.31.2, torch2.4.1, NumPy2.2.6, ONNX1.16.2 and ORT1.19.2 are available in the existing training interpreter; the protocol records actual versions.

```sh
python scripts/research/trainable_frontend/data.py
python scripts/research/trainable_frontend/test_frontend.py
python scripts/research/trainable_frontend/freeze.py
for seed in 37 73 101; do
  python scripts/research/trainable_frontend/train.py --arm fixed --seed "$seed"
  python scripts/research/trainable_frontend/train.py --arm learned --seed "$seed"
  python scripts/research/trainable_frontend/check_export.py "fixed-s$seed"
  python scripts/research/trainable_frontend/check_export.py "learned-s$seed"
done
python scripts/research/trainable_frontend/evaluate.py
```

The learned arm uses direct bank weights initialized to the current matrix. We project weights nonnegative after each optimizer update; no forward ReLU blocks initially zero coefficients. The fixed arm uses stop-gradient on identical weights. Both use the same student optimizer; the separate frontend AdamW uses .01 times its learning rate and zero weight decay. Separate norm1 clipping prevents frontend gradients from directly shrinking student updates. No new loss is added. Three seeds37/73/101 share batches,3,720 updates and development-only selection with epoch zero eligible. Interrupted/completed runs refuse overwrite; recovery states are retained without an automatic-resume claim.

Power caches reconstruct the original log-mel arrays exactly. Caching only pre-frontend power is essential: training on previously computed log-mel would bypass the parameter being tested. Existing label arrays/valid masks/voice order stay aligned and read-only.

Export copies the selected learned bank into the original ONNX `mel` initializer and only the selected predictor weights into their original locations. Graph structure/file size remain unchanged. Learned bank introduces16,448 additional learned scalars in an already existing constant tensor; model file size is not a learned-parameter count. FFT/window buffers remain identical. Existing browser frontend reads the matrix, but a denser matrix can increase its summation work; unchanged file size does not imply unchanged inference cost. No candidate is installed in the live renderer.

All selected exports are locked and paired sampler hashes checked before held-out evaluation. Existing recordings are regression screens, not new independent confirmation. The 3% primary gap screen, contact/opening/closure/engine/gain and historical guards are predeclared. Only a promising pilot warrants fresh independent recordings, browser speed/memory/parity and visual acceptance. No automatic promotion.


## Corrected parameterization / retained aborted pilot

The initial partial v1 pilot used `max(base+.01*delta,0)` inside the forward graph. Inspection found zero gradients at initially zero coefficients, so it only changed existing support. We stopped before held-out scoring and retained sources/logs/protocol under `.local/trainable-frontend-v1`, with `aborted.json`. It is not a completed quality result. V2 adds an explicit test that zero-support coefficients receive gradients and become positive after projected updates. Power caches are reused read-only via `A2F_FRONTEND_POWER` (default the verified v1 power directory), avoiding a duplicate3.8GB cache. Their creation metadata and the exact original builder source remain preserved. New training source/protocol hashes are frozen separately.
