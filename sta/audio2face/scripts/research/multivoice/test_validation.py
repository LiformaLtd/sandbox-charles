"""Check units, macro averaging and paired bootstrap on an analytic fixture."""
import copy
import unittest
import numpy as np
import validate_candidate as validation


class Comparison(unittest.TestCase):
    def test_known_improvement_is_preserved_by_crossed_bootstrap(self):
        rows = []
        for engine in ['phonon', 'supertonic']:
            for voice in ['male', 'female']:
                for text in ['first prompt', 'second prompt']:
                    rows.append(dict(engine=engine, voice=voice, text=text,
                                     squared_gap_error=16., gap_values=4,
                                     teacher_contacts=2, student_contacts=2, contact_hits=1))
        baseline = dict(clips=rows)
        candidate = copy.deepcopy(baseline)
        for row in candidate['clips']:
            row.update(squared_gap_error=4., contact_hits=2)
        result = validation.compare([candidate] * 3, baseline, 37)
        self.assertAlmostEqual(result['baseline_macro_gap_rmse'], 2.)
        self.assertAlmostEqual(result['candidate_macro_gap_rmse'], 1.)
        self.assertAlmostEqual(result['gap_relative_change'], -.5)
        self.assertAlmostEqual(result['baseline_macro_contact_f1'], .5)
        self.assertAlmostEqual(result['candidate_macro_contact_f1'], 1.)
        np.testing.assert_allclose(result['gap_delta_95ci'], [-1., -1.])
        np.testing.assert_allclose(result['contact_f1_delta_95ci'], [.5, .5])
        self.assertEqual(result['seed_gap_std'], 0.)


if __name__ == '__main__':
    unittest.main()
