import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from refinement import Cancelled, check_cancel


class FittingCancellation(unittest.TestCase):
    def test_cancelled_search_exits_before_next_synthesis(self):
        event = threading.Event()
        check_cancel(event)
        event.set()
        with self.assertRaises(Cancelled):
            check_cancel(event)


if __name__ == "__main__":
    unittest.main()
