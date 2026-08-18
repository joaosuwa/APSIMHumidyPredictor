from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modeling.src.evaluate import evaluate_prediction, persistence_baseline


class EvaluationTests(unittest.TestCase):
    def test_persistence_is_zero_variation_and_current_deficit(self):
        frame = pd.DataFrame({"dr_mm": [10.0, 20.0]})
        prediction = persistence_baseline(frame)
        np.testing.assert_array_equal(prediction, np.zeros(2))

        metrics = evaluate_prediction(
            "persistence",
            pd.Series([2.0, -3.0]),
            prediction,
            frame["dr_mm"],
            pd.Series([12.0, 17.0]),
        )
        self.assertAlmostEqual(metrics["mae_variation"], 2.5)
        self.assertAlmostEqual(metrics["mae_next_deficit"], 2.5)


if __name__ == "__main__":
    unittest.main()
