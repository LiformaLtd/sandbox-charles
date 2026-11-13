import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from refinement import Cancelled
from shared_comparison import (
    comparison_key,
    latest_scores,
    remove_scores,
    run_comparison,
)


class SharedComparison(unittest.TestCase):
    def test_context_key_changes_for_transcript_recording_and_settings(self):
        settings = {"temperature": 0.3, "metric": "v1"}
        key = comparison_key("recording-a", "Words spoken.", settings)
        self.assertEqual(
            key, comparison_key("recording-a", " Words spoken. ", settings)
        )
        self.assertNotEqual(
            key, comparison_key("recording-b", "Words spoken.", settings)
        )
        self.assertNotEqual(
            key, comparison_key("recording-a", "Other words.", settings)
        )
        self.assertNotEqual(
            key,
            comparison_key(
                "recording-a", "Words spoken.", {**settings, "temperature": 0}
            ),
        )

    def run_batch(self, folder, render, cancel):
        with (
            patch(
                "shared_comparison.protocol",
                return_value={
                    "metric": "shared-acoustic-v1",
                    "parameterPenalty": False,
                },
            ),
            patch("shared_comparison.Scorer") as scorer,
        ):
            scorer.return_value.features.return_value = {"seconds": 1}
            scorer.return_value.compare.return_value = {
                "loss": 2,
                "speakerCosine": 0.5,
                "melDistance": 0,
                "spectrumDistance": 0,
                "durationDistance": 0,
            }
            result = run_comparison(
                folder,
                folder,
                folder / "voices" / "a",
                ["a", "b"],
                "Words spoken.",
                render,
                lambda _: None,
                cancel,
            )
            for call in scorer.return_value.compare.call_args_list:
                self.assertEqual(call.kwargs, {"transcript": True, "focus": "voice"})
            return result

    def sources(self, folder):
        reference = folder / "voices" / "a" / "reference.wav"
        reference.parent.mkdir(parents=True)
        sf.write(reference, np.sin(np.arange(24000) * 0.05) * 0.1, 24000)
        other = folder / "voices" / "b" / "reference.wav"
        other.parent.mkdir(parents=True)
        other.write_bytes(reference.read_bytes())

    def test_batch_uses_one_render_per_version_and_saves_comparable_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.sources(folder)
            calls = []

            def render(text, voice_id, **kwargs):
                calls.append((text, voice_id))
                return np.ones(24000, dtype=np.float32)

            batch = self.run_batch(folder, render, threading.Event())
            self.assertEqual(calls, [("Words spoken.", "a"), ("Words spoken.", "b")])
            scores = latest_scores(folder)
            self.assertEqual(scores["a"]["comparisonKey"], scores["b"]["comparisonKey"])
            self.assertEqual(scores["a"]["error"], 2)
            self.assertFalse(batch["protocol"]["parameterPenalty"])
            remove_scores(folder, "a")
            self.assertEqual(set(latest_scores(folder)), {"b"})
            remove_scores(folder, "b")
            self.assertEqual(latest_scores(folder), {})

    def test_cancel_does_not_publish_a_partial_batch_or_replace_old_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.sources(folder)
            self.run_batch(
                folder,
                lambda *args, **kwargs: np.ones(24000, dtype=np.float32),
                threading.Event(),
            )
            previous = latest_scores(folder)
            event = threading.Event()

            def render(*args, **kwargs):
                event.set()
                return np.ones(24000, dtype=np.float32)

            with self.assertRaises(Cancelled):
                self.run_batch(folder, render, event)
            self.assertEqual(latest_scores(folder), previous)
            self.assertEqual(len(list((folder / "evaluations").glob("*.json"))), 1)

    def test_different_recordings_are_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.sources(folder)
            (folder / "voices" / "b" / "reference.wav").write_bytes(
                b"different recording"
            )
            with self.assertRaisesRegex(ValueError, "same saved recording"):
                self.run_batch(
                    folder, lambda *_: self.fail("Must not render"), threading.Event()
                )
            self.assertFalse((folder / "evaluations").exists())


if __name__ == "__main__":
    unittest.main()
