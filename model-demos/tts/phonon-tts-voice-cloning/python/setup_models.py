"""Prepare pinned, local-only experimental assets. No weights or upstream code enter Git."""

import hashlib
import importlib.util
import json
import os
import struct
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("PHONON_CLONE_MODELS", ROOT / ".models")).resolve()
COMMUNITY_REV = "8cad11e932a6883b23433176522e09fba5ffa974"
PHONON_REV = "92741c5b07d085ff840e26fea8ce43110b87852f"
MIMI_REV = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    if path.is_file():
        return
    print(f"Downloading {path.name}", flush=True)
    temp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temp.replace(path)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    vendor = MODELS / "community_encoder.py"
    download(
        f"https://raw.githubusercontent.com/connerohnesorge/phonon-encoder/{COMMUNITY_REV}/src/mimi_torch.py",
        vendor,
    )
    head_path = MODELS / "speaker_encoder_head.fp16.safetensors"
    download(
        f"https://raw.githubusercontent.com/connerohnesorge/phonon-encoder/{COMMUNITY_REV}/weights/{head_path.name}",
        head_path,
    )
    combined = MODELS / "speaker_encoder.safetensors"
    if not combined.exists():
        from safetensors.torch import load_file

        # Only the pretrained SEANet front is needed; the community head replaces
        # the transformer/downsampler. Fetch its contiguous safetensors byte range.
        url = f"https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/{MIMI_REV}/tokenizer-e351c8d8-checkpoint125.safetensors"
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=0-65535"}), timeout=60
        ) as response:
            size = struct.unpack("<Q", response.read(8))[0]
            header = json.loads(response.read(size))
        front = {k: v for k, v in header.items() if k.startswith("encoder.")}
        lo = min(v["data_offsets"][0] for v in front.values())
        hi = max(v["data_offsets"][1] for v in front.values())
        start, end = 8 + size + lo, 8 + size + hi - 1
        print("Downloading Kyutai Mimi encoder front (50.5 MB)", flush=True)
        with urllib.request.urlopen(
            urllib.request.Request(
                url + "?encoder-front=1", headers={"Range": f"bytes={start}-{end}"}
            ),
            timeout=60,
        ) as response:
            if response.status != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {start}-"):
                raise RuntimeError("Model host did not honour the encoder byte range.")
            data = response.read(hi - lo)
        if len(data) != hi - lo:
            raise RuntimeError("Incomplete encoder download. Run setup again.")
        state = {}
        for key, value in front.items():
            a, b = value["data_offsets"]
            dtype = {
                "F32": torch.float32,
                "BF16": torch.bfloat16,
                "F16": torch.float16,
            }[value["dtype"]]
            tensor = torch.frombuffer(
                bytearray(data[a - lo : b - lo]), dtype=dtype
            ).reshape(value["shape"])
            state[key.replace(".conv.conv.", ".conv.")] = tensor.float().contiguous()
        state.update({k: v.float() for k, v in load_file(head_path).items()})
        spec = importlib.util.spec_from_file_location("community_encoder", vendor)
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.MimiEncoder(module.EncCfg()).load_state_dict(state, strict=True)
        temporary = combined.with_suffix(".part")
        save_file(
            state,
            temporary,
            metadata={"community_revision": COMMUNITY_REV, "mimi_revision": MIMI_REV},
        )
        temporary.replace(combined)

    base = (
        Path(os.environ.get("PHONON_BASE_MODEL", MODELS / "phonon-31m-en"))
        .expanduser()
        .resolve()
    )
    base.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "model.q8.gguf",
        "default-voice.safetensors",
    ):
        if "PHONON_BASE_MODEL" in os.environ:
            if not (base / name).is_file():
                raise FileNotFoundError(f"Configured Phonon model is missing {name}")
        else:
            download(
                f"https://huggingface.co/gradium/phonon-31m-en/resolve/{PHONON_REV}/{name}",
                base / name,
            )
    files = [vendor, combined] + [
        base / name
        for name in (
            "config.json",
            "tokenizer.json",
            "model.q8.gguf",
            "default-voice.safetensors",
        )
    ]
    manifest = {
        "community_revision": COMMUNITY_REV,
        "phonon_revision": PHONON_REV,
        "mimi_revision": MIMI_REV,
        "base_model": str(base),
        "files": {str(path): digest(path) for path in files},
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Local encoder and Phonon 31M are ready.", flush=True)


if __name__ == "__main__":
    main()
