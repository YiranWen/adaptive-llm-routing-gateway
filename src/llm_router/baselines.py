"""Fixed routing baselines."""

from __future__ import annotations

import numpy as np

from llm_router.offline_data import OfflineSplit
from llm_router.utilities import LARGE, SMALL, SLAMode, oracle_actions


def always_small(split: OfflineSplit) -> np.ndarray:
    return np.full(len(split.features), SMALL, dtype=np.int64)


def always_large(split: OfflineSplit) -> np.ndarray:
    return np.full(len(split.features), LARGE, dtype=np.int64)


def oracle_best_utility(split: OfflineSplit, sla: SLAMode) -> np.ndarray:
    return oracle_actions(split, sla)
