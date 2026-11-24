from collections import Counter

import pytest

from tests.evals.test_classifier_eval import prf


def test_prf_matches_hand_computation() -> None:
    confusion = Counter({("mixed", "mixed"): 3, ("mixed", "logical"): 1, ("logical", "mixed"): 1})
    precision, recall, f1 = prf(confusion, "mixed")
    assert precision == pytest.approx(3 / 4)
    assert recall == pytest.approx(3 / 4)
    assert f1 == pytest.approx(3 / 4)
