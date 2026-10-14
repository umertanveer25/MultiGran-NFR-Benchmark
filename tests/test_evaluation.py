"""Unit tests for Evaluation and Statistical suites."""

import pytest
import numpy as np
from src.statistics import cohens_d, cliffs_delta, wilcoxon_signed_rank

def test_cohens_d():
    x = [0.95, 0.96, 0.94, 0.97, 0.95]
    y = [0.85, 0.86, 0.84, 0.87, 0.85]
    d = cohens_d(x, y)
    assert d > 1.5

def test_cliffs_delta():
    x = [0.95, 0.96, 0.94, 0.97, 0.95]
    y = [0.85, 0.86, 0.84, 0.87, 0.85]
    delta = cliffs_delta(x, y)
    assert delta == 1.0

def test_wilcoxon():
    x = [0.95, 0.96, 0.94, 0.97, 0.95, 0.98, 0.96, 0.95, 0.97, 0.96]
    y = [0.85, 0.86, 0.84, 0.87, 0.85, 0.88, 0.86, 0.85, 0.87, 0.86]
    stat, p = wilcoxon_signed_rank(x, y)
    assert p < 0.05
