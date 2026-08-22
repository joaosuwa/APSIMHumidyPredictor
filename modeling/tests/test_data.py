from __future__ import annotations

import pandas as pd
import pytest

from modeling.config import DEFAULT_CONFIG, DataConfig
from modeling.data import (
    DIRECT_IRRIGATION_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    filter_simulations,
    prepare_data,
    split_test_cycle_ids,
)


def test_default_preparation_matches_plan4() -> None:
    prepared = prepare_data(DEFAULT_CONFIG)

    assert len(prepared.filtered) == 4036
    assert len(prepared.development) == 3422
    assert len(prepared.test) == 614
    assert set(prepared.development["cycle_id"]) == set(range(6))
    assert set(prepared.test["cycle_id"]) == {6}
    assert len(prepared.folds) == 6

    validation_indices = []
    for fold in prepared.folds:
        train = prepared.development.loc[fold.train_indices]
        validation = prepared.development.loc[fold.validation_indices]
        assert set(validation["cycle_id"]) == {fold.validation_cycle_id}
        assert set(validation["SimulationName"]) == set(DEFAULT_CONFIG.included_simulations)
        assert len(set(train["cycle_id"])) == 5
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        validation_indices.extend(fold.validation_indices)

    assert sorted(validation_indices) == prepared.development.index.tolist()
    forbidden = {*METADATA_COLUMNS, *DIRECT_IRRIGATION_COLUMNS, TARGET_COLUMN}
    assert forbidden.isdisjoint(prepared.feature_columns)


def test_filter_returns_independent_copy() -> None:
    source = pd.DataFrame(
        {"SimulationName": ["A", "B", "A"], "value": [1, 2, 3]}
    )
    filtered = filter_simulations(source, ("A",))
    filtered.loc[0, "value"] = 999
    assert source.loc[0, "value"] == 1
    assert filtered["SimulationName"].tolist() == ["A", "A"]


def test_split_rejects_non_final_or_missing_cycle() -> None:
    prepared = prepare_data(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="posteriores"):
        split_test_cycle_ids(prepared.filtered, (5,))
    with pytest.raises(ValueError, match="não encontrados"):
        split_test_cycle_ids(prepared.filtered, (99,))


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"included_simulations": ()}, ValueError),
        ({"included_simulations": ("A", "A")}, ValueError),
        ({"included_simulations": ("A",), "test_cycle_ids": ()}, ValueError),
        ({"included_simulations": ("A",), "test_cycle_ids": (True,)}, TypeError),
    ],
)
def test_data_config_validation(kwargs: dict, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        DataConfig(**kwargs)
