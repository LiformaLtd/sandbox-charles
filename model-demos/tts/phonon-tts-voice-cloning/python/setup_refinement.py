"""Optional, pinned public assets for local transcription and voice fitting."""

import json

from huggingface_hub import snapshot_download
from setup_models import MODELS, digest, download

SPEAKER_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
SPEAKER_SHA = "c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5"
WHISPER_REV = "911407f4214e0e1d82085af863093ec0b66f9cd6"
VOICES_REV = "0ee17205f254b5abc43b266c8a0f215565c352f7"


def main():
    import urllib.request

    from refinement import make_basis

    folder = MODELS / "refinement"
    folder.mkdir(exist_ok=True)
    download(SPEAKER_URL, folder / "speaker.onnx")
    if digest(folder / "speaker.onnx") != SPEAKER_SHA:
        raise RuntimeError("The speaker comparison model failed its checksum check.")
    snapshot_download(
        "openai/whisper-base.en",
        revision=WHISPER_REV,
        local_dir=folder / "whisper",
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    )
    if not (folder / "basis.npz").exists():
        catalogue = (
            f"https://huggingface.co/gradium/phonon-voices/resolve/{VOICES_REV}/"
        )
        with urllib.request.urlopen(catalogue + "voices.jsonl", timeout=60) as stream:
            voices = [
                json.loads(line)
                for line in stream.read().decode().splitlines()
                if line.strip()
            ]
        sources = folder / "public-voices"
        sources.mkdir(exist_ok=True)
        paths = []
        for voice in voices:
            if voice.get("language") != "en":
                continue
            voice_id = voice["voice_id"]
            path = sources / f"{voice_id}.safetensors"
            download(catalogue + f"voices/{voice_id}.safetensors", path)
            paths.append(path)
        make_basis(paths, folder / "basis.npz")
    files = [folder / "speaker.onnx", folder / "basis.npz"] + [
        p
        for p in (folder / "whisper").iterdir()
        if p.is_file() and not p.name.startswith(".")
    ]
    manifest = {
        "speaker_url": SPEAKER_URL,
        "whisper_revision": WHISPER_REV,
        "voices_revision": VOICES_REV,
        "files": {str(p.resolve()): digest(p) for p in files},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Local transcription and experimental voice fitting are ready.")


if __name__ == "__main__":
    main()
