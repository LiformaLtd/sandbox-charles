import unittest
from prepare import text_coverage, normalize
from evaluate import aggregate


class Coverage(unittest.TestCase):
    def test_occurrence_mass_counts_each_observed_pair_once(self):
        result = text_coverage(['One two one.', 'One two one.'],
                               {'ONE': ['W', 'AH', 'N'], 'TWO': ['T', 'UW']},
                               [{'pair': 'W AH', 'occurrences': 80}, {'pair': 'X Y', 'occurrences': 20}])
        self.assertEqual(result['distinct_texts'], 1)
        self.assertAlmostEqual(result['subtitle_occurrence_coverage'], .8)
        self.assertEqual(normalize('One, TWO one!'), 'one two one')

    def test_macro_preserves_voice_balance_and_mm_units(self):
        rows = [dict(engine='phonon', voice=v, gap_values=n, squared_gap_error=e,
                     teacher_contacts=2, student_contacts=2, contact_hits=1,
                     opening_values=n, opening_squared_error=e)
                for v, n, e in [('short', 1, 1), ('long', 100, 400)]]
        result = aggregate(rows)
        self.assertAlmostEqual(result['macro_gap_rmse_mm'], 15)
        self.assertAlmostEqual(result['macro_contact_f1'], .5)


if __name__ == '__main__':
    unittest.main()
