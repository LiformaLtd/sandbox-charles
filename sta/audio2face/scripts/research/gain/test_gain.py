import unittest
import numpy as np
from prepare import gain_for_peak
from evaluate import aggregate


class GainPolicy(unittest.TestCase):
    def test_boost_respects_headroom_without_attenuating_existing_peaks(self):
        self.assertEqual(gain_for_peak(.25, 2), 2)
        self.assertAlmostEqual(gain_for_peak(.7, 2) * .7, .98)
        self.assertEqual(gain_for_peak(1, 2), 1)
        self.assertEqual(gain_for_peak(.7, .5), .5)

    def test_macro_gives_short_and_long_voices_equal_weight_and_converts_cm(self):
        rows = [dict(engine='phonon', voice=v, squared_gap_error=error,
                     gap_values=n, teacher_contacts=2, student_contacts=2, contact_hits=2,
                     opening_values=n, opening_squared_error=error)
                for v, error, n in [('short', 1., 1), ('long', 400., 100)]]
        result = aggregate(rows)
        self.assertAlmostEqual(result['macro_gap_rmse_mm'], 15.)
        self.assertEqual(result['macro_contact_f1'], 1.)


if __name__ == '__main__':
    unittest.main()
