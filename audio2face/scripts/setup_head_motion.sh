#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
assets=.local/head-motion
mkdir -p "$assets/pose" "$assets/hubert"

if [ ! -d "$assets/dawn" ]; then
  git init -q "$assets/dawn"
  git -C "$assets/dawn" fetch -q --depth 1 https://github.com/Hanbo-Cheng/DAWN-pytorch.git 76c44b8396c342a9cb523fee1d2dc4c5342d5934
  git -C "$assets/dawn" checkout -q --detach FETCH_HEAD
fi

fetch() {
  [ -f "$2" ] && return 0
  curl --fail --silent --show-error --location --retry 2 --max-time 300 "$1" -o "$2.part"
  mv "$2.part" "$2"
}

pose=https://huggingface.co/Hanbo-Cheng/DAWN/resolve/3894f3b3ad4dab484428d8fd52e2cd41fd00ddad/pbnet/pose
hubert=https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/ece5fabbf034c1073acae96d5401b25be96709d8
for file in opt.yaml checkpoint_40000.pth.tar; do fetch "$pose/$file" "$assets/pose/$file"; done
for file in config.json preprocessor_config.json pytorch_model.bin; do fetch "$hubert/$file" "$assets/hubert/$file"; done

[ -x "$assets/.venv/bin/python" ] || uv venv --python 3.12 "$assets/.venv"
uv pip install --python "$assets/.venv/bin/python" \
  torch==2.15.0 transformers==5.22.1 numpy==2.5.4 scipy==1.18.1 pyyaml==6.0.3 \
  einops==0.8.2 einops-exts==0.0.4 rotary-embedding-torch==0.9.1
