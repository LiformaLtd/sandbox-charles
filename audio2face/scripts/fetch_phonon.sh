#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
checkpoint='https://huggingface.co/gradium/phonon-31m-en/resolve/8c64b55ec301728e2d1695bfea87a08a267d0c2d'
catalogue='https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7'
mkdir -p .local/phonon
fetch() {
  curl --fail --silent --show-error --location --retry 2 --max-time 300 "$1" -o "$2.part"
  mv "$2.part" "$2"
}
for file in config.json tokenizer.json model.q8.gguf; do
  fetch "$checkpoint/$file" ".local/phonon/$file"
done
fetch "$catalogue/voices/LMDIYE0Oa377Qh6Z.safetensors" .local/phonon/easton.safetensors
