import struct
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/lib'))
from motion_features import MotionFeatures


class MotionFeatureTests(unittest.TestCase):
    def fixture(self):
        measure = MotionFeatures.__new__(MotionFeatures)
        points = np.tile([0., 168., 6.], (18, 1)).astype(np.float32)
        points[:2, 0] = -1
        points[-2:, 0] = 1
        measure.native_skin = points.copy()
        measure.fit_skin = points.copy()
        measure.neutral_width, measure.neutral_protrusion = measure.lip_shape(points[None])
        measure.origin = np.array([0, 168, 6], dtype=np.float32)
        measure.incisors = np.zeros((1, 3), dtype=np.float32)
        measure.inc_norm = np.zeros(1, dtype=np.float32)
        measure.skin_basis = np.zeros((52, 54), dtype=np.float32)
        measure.jaw_basis = np.zeros((53, 5, 3), dtype=np.float32)
        return measure, points

    def test_shape_and_proximity_are_in_head_coordinates(self):
        measure, points = self.fixture()
        jaw = np.zeros((1, 5, 3), dtype=np.float32)
        jaw[:, :, 1] = -.3
        values = measure.measure(points[None], jaw)[0]
        np.testing.assert_allclose(values, [0, 0, .3, 0], atol=1e-6)
        points[[7, 9, 11], 2] += .5
        values = measure.measure(points[None], jaw)[0]
        self.assertAlmostEqual(values[3], -.5, places=5)
        self.assertAlmostEqual(values[1], .25, places=5)

    def test_moving_corners_cannot_cancel_true_forward_lip_motion(self):
        measure, points = self.fixture()
        points[6:12, 2] += 1
        points[:2, 2] += 2
        points[-2:, 2] += 2
        _, protrusion = measure.lip_shape(points[None])
        self.assertAlmostEqual(float(protrusion[0] - measure.neutral_protrusion[0]), 1)

    def test_local_origin_keeps_small_tooth_distances_numerically_resolved(self):
        measure, points = self.fixture()
        points[[7, 9, 11], 2] += .001
        values = measure.measure(points[None], np.zeros((1, 5, 3), dtype=np.float32))[0]
        self.assertAlmostEqual(values[3], -.001, delta=5e-6)

    def test_native_and_fitted_routes_agree_for_equivalent_poses(self):
        measure, _ = self.fixture()
        geometry = np.zeros((1, 88831), dtype=np.float32)
        geometry[:, 88813:88827:3] = -.2
        measure.jaw_basis[17, :, 1] = -.2
        weights = np.zeros((1, 52), dtype=np.float32)
        weights[:, 17] = 1
        np.testing.assert_allclose(measure.native(geometry), measure.fitted(weights), atol=1e-6)

    def test_jaw_binary_parser_preserves_channel_and_marker_indices(self):
        binary = b'JBS1' + struct.pack('<III', 0, 0, 1) + struct.pack('<IIf', 17, 1, .01)
        binary += np.array([[4, 0, -10, 0]], dtype='<i2').tobytes()
        jaw = MotionFeatures.read_jaw_basis(binary)
        self.assertAlmostEqual(float(jaw[17, 4, 1]), -.1, places=6)
        self.assertEqual(np.count_nonzero(jaw), 1)
        with self.assertRaises(ValueError):
            MotionFeatures.read_jaw_basis(binary + b'bad')


if __name__ == '__main__':
    unittest.main()
