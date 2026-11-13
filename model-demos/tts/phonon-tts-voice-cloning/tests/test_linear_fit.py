import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from linear_fit import solve_weights


class LinearSolve(unittest.TestCase):
    def test_recovers_a_known_affine_combination_using_all_36_directions(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(37, 64))
        expected = rng.normal(size=37)
        expected += (1 - expected.sum()) / 37
        weights, rank = solve_weights(features, expected @ features, ridge=1e-10)
        self.assertEqual(rank, 36)
        self.assertAlmostEqual(weights.sum(), 1)
        np.testing.assert_allclose(weights, expected, atol=1e-8)

    def test_catalogue_order_does_not_change_the_fitted_voice(self):
        rng = np.random.default_rng(43)
        features, target = rng.normal(size=(37, 64)), rng.normal(size=64)
        order = rng.permutation(37)
        weights, _ = solve_weights(features, target)
        reordered, _ = solve_weights(features[order], target)
        np.testing.assert_allclose(reordered, weights[order], atol=1e-10)

    def test_rank_deficient_catalogue_has_a_stable_regularized_solution(self):
        weights, rank = solve_weights(np.ones((37, 64)), np.zeros(64))
        self.assertEqual(rank, 0)
        np.testing.assert_allclose(weights, np.ones(37) / 37)

    def test_invalid_reference_and_regularization_are_rejected(self):
        features = np.ones((37, 64))
        for target, ridge in [
            (np.full(64, np.nan), 0.05),
            (np.zeros(64), float("nan")),
            (np.zeros(64), 0),
        ]:
            with self.assertRaises(ValueError):
                solve_weights(features, target, ridge)


if __name__ == "__main__":
    unittest.main()
