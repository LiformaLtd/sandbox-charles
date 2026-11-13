"""Build the public 37-voice feature catalogue once; no user recordings are used."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import ptts
import torch
from linear_fit import BANK_TEXTS
from refinement import Scorer
from safetensors.torch import load_file
from setup_models import MODELS, digest, download
from setup_refinement import VOICES_REV
from synthesis import PLAYBACK_SETTINGS


def main():
    os.umask(0o077)
    torch.set_num_threads(4)
    models = MODELS / "refinement"
    if not (models / "manifest.json").is_file():
        raise RuntimeError("Run npm run setup:refinement before setup:linear.")
    model_manifest = json.loads((MODELS / "manifest.json").read_text())
    base = Path(model_manifest["base_model"])
    catalogue = f"https://huggingface.co/gradium/phonon-voices/resolve/{VOICES_REV}/"
    with urllib.request.urlopen(catalogue + "voices.jsonl", timeout=60) as response:
        entries = [
            json.loads(line)
            for line in response.read().decode().splitlines()
            if line.strip()
        ]
    entries = [v for v in entries if v.get("language") == "en"]
    if len(entries) != 37:
        raise RuntimeError("The pinned catalogue must contain 37 English voices.")
    folder = models / "linear"
    folder.mkdir(parents=True, exist_ok=True)
    sources = folder / "sources"
    sources.mkdir(exist_ok=True)
    synth = ptts.TTS(str(base), lang="en", device="cpu", quant="q8")
    scorer = Scorer(models / "speaker.onnx")
    features, voices, names = [], [], []
    for index, entry in enumerate(entries):
        voice_id = entry["voice_id"]
        path = sources / f"{voice_id}.safetensors"
        download(catalogue + f"voices/{voice_id}.safetensors", path)
        voice = load_file(path)["speaker_wavs"].numpy()
        if voice.shape != (1, 512, 125) or not np.isfinite(voice).all():
            raise RuntimeError("Unexpected pinned catalogue voice format.")
        synth.add_voice(voice_id, str(path))
        embeddings = []
        for text in BANK_TEXTS:
            chunks, length = [], 0
            with synth.stream(text, voice=voice_id, **PLAYBACK_SETTINGS) as stream:
                for chunk in stream:
                    pcm = np.array(chunk, dtype=np.float32, copy=True)
                    length += len(pcm)
                    if length > 30 * 24000:
                        raise RuntimeError(
                            "A catalogue sample produced overlong speech."
                        )
                    chunks.append(pcm)
            if not chunks:
                raise RuntimeError("A catalogue sample produced no speech.")
            embeddings.append(scorer.features(np.concatenate(chunks))["embedding"])
        feature = np.mean(embeddings, axis=0)
        feature /= max(np.linalg.norm(feature), 1e-8)
        features.append(feature)
        voices.append(voice[0])
        names.append(entry["name"])
        print(f"Prepared {index + 1}/37: {entry['name']}", flush=True)
    temporary = folder / "catalogue.tmp.npz"
    np.savez_compressed(
        temporary,
        features=np.stack(features),
        voices=np.stack(voices),
        names=np.array(names),
    )
    temporary.replace(folder / "catalogue.npz")
    manifest = {
        "voicesRevision": VOICES_REV,
        "texts": BANK_TEXTS,
        "sampling": PLAYBACK_SETTINGS,
        "baseModel": str(base),
        "speakerSha256": digest(models / "speaker.onnx"),
        "baseManifestSha256": hashlib.sha256(
            json.dumps(model_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files": {"catalogue.npz": digest(folder / "catalogue.npz")},
    }
    (folder / "manifest.tmp.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "manifest.tmp.json").replace(folder / "manifest.json")
    print("Fast linear fitting is ready: 37 voices, all 36 centred directions.")


if __name__ == "__main__":
    main()
