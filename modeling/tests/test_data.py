from __future__ import annotations

import unittest

import numpy as np

from modeling.src.data import feature_target, load_dataset, make_temporal_splits
from scripts.model_dataset import METADATA_COLUMNS, TARGET_COLUMNS


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        cls.splits = make_temporal_splits(cls.dataset)

    def test_target_is_next_day_in_same_cycle(self):
        self.assertTrue(((self.dataset["data_alvo"] - self.dataset["data"]).dt.days == 1).all())
        next_rows = self.dataset[
            ["simulation_name", "cycle_id", "data", "dr_mm"]
        ].rename(columns={"data": "data_alvo", "dr_mm": "expected_next_dr"})
        matched = self.dataset.merge(
            next_rows,
            on=["simulation_name", "cycle_id", "data_alvo"],
            how="inner",
        )
        self.assertGreater(len(matched), 0)
        self.assertTrue(
            np.allclose(matched["deficit_agua_proximo_dia_mm"], matched["expected_next_dr"])
        )

    def test_temporal_split_has_no_2025_in_train(self):
        self.assertLessEqual(self.splits.final_train["ano_semeadura"].max(), 2024)
        self.assertEqual(set(self.splits.test["ano_semeadura"]), {2025})

    def test_features_do_not_contain_metadata_or_targets(self):
        features, _ = feature_target(self.splits.final_train)
        forbidden = set(METADATA_COLUMNS + TARGET_COLUMNS)
        self.assertFalse(forbidden.intersection(features.columns))


if __name__ == "__main__":
    unittest.main()
