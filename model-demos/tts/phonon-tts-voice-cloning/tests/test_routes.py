import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import server


class LocalAccess(unittest.TestCase):
    def setUp(self):
        self.token = patch.object(server, "TOKEN", "test-token")
        self.token.start()
        self.client = TestClient(server.app)
        self.headers = {"x-lab-token": "test-token"}

    def tearDown(self):
        self.client.close()
        self.token.stop()

    def test_service_rejects_missing_internal_token(self):
        self.assertEqual(self.client.get("/health").status_code, 403)

    def test_voice_requires_own_voice_consent(self):
        response = self.client.post("/voices", headers=self.headers, content=b"audio")
        self.assertEqual(response.status_code, 400)
        self.assertIn("own voice", response.json()["detail"])

    def test_unknown_or_unsafe_file_ids_are_not_served(self):
        for path in ["/voices/not-an-id/reference", "/audio/not-an-id"]:
            self.assertEqual(
                self.client.get(path, headers=self.headers).status_code, 404
            )

    def test_blank_or_oversized_speech_is_rejected(self):
        for text in [" ", "a" * 1001]:
            response = self.client.post(
                "/speech",
                headers=self.headers,
                json={"voiceId": "default", "text": text},
            )
            self.assertIn(response.status_code, [400, 422])

    def test_fitting_requires_confirmed_recording_transcript(self):
        with patch.object(server, "voice_folder", return_value=Path("/unused")):
            response = self.client.post(
                "/voices/00000000-0000-0000-0000-000000000001/refine",
                headers=self.headers,
                json={
                    "transcript": "The words I actually recorded.",
                    "confirmed": False,
                },
            )
        self.assertEqual(response.status_code, 400)

    def test_cancelling_unknown_fitting_job_is_rejected(self):
        response = self.client.post(
            "/refinements/00000000-0000-0000-0000-000000000001/cancel",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
