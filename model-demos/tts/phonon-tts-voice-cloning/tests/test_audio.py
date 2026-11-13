import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from audio import prepare_reference, read_recording


def wav(samples, rate=24000):
    output = BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class RecordingValidation(unittest.TestCase):
    def test_rejects_silence_short_clipped_and_wrong_rate(self):
        cases = [
            wav(np.zeros(24000 * 6)),
            wav(np.ones(24000) * 0.1),
            wav(np.ones(24000 * 6)),
            wav(np.ones(48000 * 6) * 0.1, 48000),
            b"not a wav",
        ]
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                read_recording(data)

    def test_preparation_caps_reference_and_preserves_finite_audio(self):
        t = np.arange(24000 * 15) / 24000
        samples = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        decoded = read_recording(wav(samples))
        prepared = prepare_reference(decoded)
        self.assertEqual(prepared.shape, (240000,))
        self.assertTrue(np.isfinite(prepared).all())
        self.assertGreater(float(np.sqrt(np.mean(prepared**2))), 0.01)


if __name__ == "__main__":
    unittest.main()
