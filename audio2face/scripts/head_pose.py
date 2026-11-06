"""Run DAWN's pretrained pose branch without its video renderer."""
import argparse
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import yaml

FPS = 25
SAMPLE_RATE = 16000
IDLE_SECONDS = 12


class HeadPose:
    def __init__(self, assets, device):
        self.device = device
        sys.path.insert(0, str(assets / "dawn" / "PBnet"))
        from src.models.get_model import get_model

        checkpoint = assets / "pose"
        config = yaml.safe_load((checkpoint / "opt.yaml").read_text())
        config.update(device=device, audio_dim=1024, pos_dim=6, eye_dim=0)
        self.pose = get_model(config)
        self.pose.load_state_dict(torch.load(checkpoint / "checkpoint_40000.pth.tar", map_location="cpu", weights_only=True))
        self.pose.eval()
        hubert = assets / "hubert"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert, local_files_only=True)
        self.hubert = HubertModel.from_pretrained(hubert, local_files_only=True).eval().to(device)
        # DAWN's normalized neutral reference, including its camera-space offsets.
        self.neutral = torch.tensor([[[0.5, 0.5, 0.5, 4.79e-4, 56.5 / 720, 64.9 / 1080]]], device=device)
        self.latent_size = config["latent_dim"]
        self.rng = torch.Generator(device="cpu").manual_seed(173)
        self.noise = torch.randn(4096, 1, self.latent_size, generator=self.rng).to(device)

    @torch.inference_mode()
    def predict(self, pcm, start_sample=0):
        pcm = np.asarray(pcm, dtype=np.float32)
        count = max(1, round(len(pcm) * FPS / SAMPLE_RATE))
        if len(pcm) < 400:
            pcm = np.pad(pcm, (0, 400 - len(pcm)))
        audio = self.processor(pcm, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        encoded = self.hubert(audio).last_hidden_state
        features = F.interpolate(encoded.transpose(1, 2), size=count, mode="linear", align_corners=True).transpose(1, 2)
        # Index the latent sequence by clip time for repeatable overlapping windows.
        start = round(start_sample * FPS / SAMPLE_RATE)
        indices = (torch.arange(count, device=self.device) + start) % self.noise.shape[0]
        batch = {
            "x": self.neutral,
            "z": self.noise[indices],
            "y": features,
            "mask": torch.ones((1, count), dtype=torch.bool, device=self.device),
            "lengths": torch.tensor([count], device=self.device),
        }
        result = self.pose.decoder(batch)["output"][0, :, :3].float().cpu().numpy() * 180
        # DAWN uses yaw, pitch, roll. The rig's Euler axes are pitch, yaw, roll.
        return result[:, [1, 0, 2]] * np.array([1, -1, 1], dtype=np.float32)

    def predict_idle(self):
        track = self.predict(np.zeros(IDLE_SECONDS * SAMPLE_RATE, dtype=np.float32))
        # Overlap one second at the loop join, easing into the start of the track.
        blend = (np.arange(FPS) / FPS)[:, None]
        blend = blend * blend * (3 - 2 * blend)
        track[-FPS:] += (track[:FPS] - track[-FPS:]) * blend
        loop = track[FPS:]
        # A centered Gaussian damps quick changes. Wrap the filter around
        # the loop so it smooths the join the same way as any other frame.
        sigma = 0.20 * FPS
        radius = round(3 * sigma)
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(loop, ((radius, radius), (0, 0)), mode="wrap")
        smoothed = np.stack([np.convolve(axis, kernel, mode="valid") for axis in padded.T], axis=1)
        return (smoothed - smoothed.mean(axis=0)) * [1.25, 1.3, 1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with redirect_stdout(sys.stderr):
        model = HeadPose(args.assets.resolve(), args.device)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with redirect_stdout(io.StringIO()):
                if request.get("idle"):
                    track = model.predict_idle()
                else:
                    pcm = np.frombuffer(base64.b64decode(request["pcm"], validate=True), dtype="<f4")
                    if not 0 < len(pcm) <= 32000 or not np.isfinite(pcm).all():
                        raise ValueError("Expected up to two seconds of finite 16 kHz mono PCM.")
                    track = model.predict(pcm, request["startSample"])
            if not np.isfinite(track).all():
                raise ValueError("The pose model produced a non-finite result.")
            reply = {"id": request["id"], "fps": FPS, "angles": track.tolist()}
        except Exception as error:
            reply = {"id": request["id"], "error": str(error)}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
