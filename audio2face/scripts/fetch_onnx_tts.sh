#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

fetch() {
  url=$1
  dest=$2
  if [ -s "$dest" ]; then return; fi
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 "$url" -o "$dest.part"
  mv "$dest.part" "$dest"
}

supertonic=https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46
pocket=https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/d0c0c79b7712256a32d691c67f20b8ae2e020d00

for file in onnx/duration_predictor.onnx onnx/text_encoder.onnx onnx/vector_estimator.onnx onnx/vocoder.onnx onnx/tts.json onnx/unicode_indexer.json voice_styles/M1.json LICENSE; do
  fetch "$supertonic/$file" ".local/supertonic/$file"
done &
supertonic_pid=$!
for file in bundle.json flow_lm_flow_int8.onnx flow_lm_main_int8.onnx mimi_decoder_int8.onnx text_conditioner_int8.onnx tokenizer.model voices.bin; do
  fetch "$pocket/onnx/english_2026-04/$file" ".local/pocket/$file"
done
fetch "$pocket/onnx/ONNX-LICENSE" .local/pocket/ONNX-LICENSE
fetch "$pocket/sentencepiece.js" .local/pocket/sentencepiece.js
wait "$supertonic_pid"
