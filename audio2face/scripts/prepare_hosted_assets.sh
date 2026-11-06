#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
fetch() {
  curl -fsSL --retry 3 "$1" -o "$2"
}

# Use the same rig data as local setup. Build-only dependencies stay outside dist.
fetch https://astral.sh/uv/0.11.21/install.sh "$temp_dir/install-uv.sh"
UV_UNMANAGED_INSTALL="$temp_dir/bin" sh "$temp_dir/install-uv.sh"
fetch https://media.githubusercontent.com/media/NVIDIA/Maya-ACE/release/2.0/sample_project/maya_geom/james_geom_v3.ma "$temp_dir/james_geom_v3.ma"
fetch https://huggingface.co/nvidia/Audio2Face-3D-v3.0/resolve/b74132732fd9a9d29b237bec193ded64c9745e91/model_data_James.npz "$temp_dir/model_data_James.npz"
"$temp_dir/bin/uv" run --no-project --python 3.12 --with numpy==2.5.4 python scripts/prepare_rig.py \
  "$temp_dir/james_geom_v3.ma" "$temp_dir/model_data_James.npz" --out "$temp_dir/assets"

# This ES module must be served with a JavaScript content type.
mkdir -p public/assets/pocket
fetch https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/d0c0c79b7712256a32d691c67f20b8ae2e020d00/sentencepiece.js public/assets/pocket/sentencepiece.js
cp "$temp_dir/assets/rig.bin" "$temp_dir/assets/rig.json" public/assets/
