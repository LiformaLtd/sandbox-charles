import unittest
import numpy as np
import closure_metrics as metrics


class Timing(unittest.TestCase):
    def test_closure_entirely_before_acoustic_phone_is_associated(self):
        # 83ms ahead, reopening 50ms before the acoustic phone: old 45ms
        # association excluded this. The actual teacher episode remains fixed.
        phone = dict(phone='p', start=.5, end=.6)
        event = (25, 27)
        self.assertEqual(metrics.associate([phone], [event], 60)[0]['event'], event)
        self.assertEqual(metrics.associate([phone], [event], 60, lead=.045)[0]['status'], 'no_teacher_contact')

    def test_neighboring_phones_do_not_both_claim_one_closure(self):
        phones = [dict(phone='m', start=.4, end=.47), dict(phone='b', start=.5, end=.6)]
        results = metrics.associate(phones, [(24, 28)], 60)
        self.assertEqual([r['status'] for r in results], ['shared_with_neighbor'] * 2)

    def test_early_teacher_timing_is_not_shifted_for_scoring(self):
        truth = np.ones((60, 9))
        truth[25:29] = 0
        row = dict(stem='early', phones=[dict(phone='b', start=.5, end=.6)])
        correct = metrics.score([(row, truth, truth)])['summary']['b']
        self.assertEqual(correct['well_timed_hits'], 1)
        self.assertEqual(correct['onset_errors_ms']['mae'], 0)
        late = np.ones((60, 9))
        late[30:34] = 0
        self.assertEqual(metrics.score([(row, truth, late)])['summary']['b']['overlap_hits'], 0)

    def test_one_frame_overlap_is_not_called_well_timed(self):
        truth = np.ones((60, 9)); truth[25:30] = 0
        pred = np.ones((60, 9)); pred[29:34] = 0
        row = dict(stem='late', phones=[dict(phone='p', start=.5, end=.6)])
        value = metrics.score([(row, truth, pred)])['summary']['p']
        self.assertEqual(value['overlap_hits'], 1)
        self.assertEqual(value['well_timed_hits'], 0)


if __name__ == '__main__':
    unittest.main()
