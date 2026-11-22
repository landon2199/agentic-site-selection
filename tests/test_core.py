"""Offline tests of the core maths in sitesel/core.py, each on a case small enough to check by hand."""

import math
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sitesel import core  # noqa: E402


# ------------------------------------------------------------------ AUC
def test_auc_partial_order():
    # positives score 4 and 2, negatives 3 and 1: 3 of the 4 positive-negative pairs are ordered right
    assert core.auc(np.array([4, 3, 2, 1]), np.array([1, 0, 1, 0])) == pytest.approx(0.75)


def test_auc_ties_count_half():
    # positive 1 vs negatives 1 (tie, 0.5) and 0 (win, 1): (0.5 + 1) / 2
    assert core.auc(np.array([1, 1, 0]), np.array([1, 0, 0])) == pytest.approx(0.75)


def test_auc_needs_both_classes():
    assert math.isnan(core.auc(np.array([1.0, 2.0]), np.array([1, 1])))
    assert math.isnan(core.auc(np.array([1.0, 2.0]), np.array([0, 0])))


# ------------------------------------------------------------------ precision@k
def test_precision_at_k():
    s, y = np.array([4, 3, 2, 1]), np.array([1, 0, 1, 0])
    assert core.precision_at_k(s, y, 1) == 1.0
    assert core.precision_at_k(s, y, 3) == pytest.approx(2 / 3)
    assert core.precision_at_k(s, y, 4) == 0.5
    assert math.isnan(core.precision_at_k(s, y, 0))


def test_precision_at_k_ties_keep_input_order():
    # all scores equal: the stable sort keeps the first k rows
    assert core.precision_at_k(np.zeros(4), np.array([1, 1, 0, 0]), 2) == 1.0


# ------------------------------------------------------------------ set and rank agreement
def test_top_share_and_jaccard():
    s = pd.Series(np.arange(10.0), index=list("abcdefghij"))
    assert core.top_share(s, 0.2) == {"i", "j"}
    assert core.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert math.isnan(core.jaccard(set(), set()))


def test_spearman_inner_join_and_sign():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("wxyz"))
    assert core.spearman(a, a ** 3) == pytest.approx(1.0)
    assert core.spearman(a, -a) == pytest.approx(-1.0)
    # only the shared index labels (x, y, z) are compared
    b = pd.Series([30.0, 20.0, 10.0, 99.0], index=list("xyzq"))
    assert core.spearman(a, b) == pytest.approx(-1.0)


# ------------------------------------------------------------------ rescale
def test_rescale_log_linear_midpoint():
    # log1p turns these into 0, 1, 2; with the 0th and 100th percentiles that maps to 1, 5.5, 10
    x = pd.Series([0.0, math.e - 1, math.e ** 2 - 1])
    r = core.rescale(x, "log_linear", "increasing", 0, 100)
    assert r.tolist() == pytest.approx([1.0, 5.5, 10.0])


def test_rescale_clips_outside_percentiles():
    r = core.rescale(pd.Series(np.arange(101.0)), "linear", "increasing", 10, 90)
    assert r.iloc[0] == 1 and r.iloc[100] == 10 and r.iloc[50] == pytest.approx(5.5)


# ------------------------------------------------------------------ cosine similarity variants
def test_similar_numpy_mean_vs_max():
    emb = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=["a", "b", "c"])
    mean = core.similar_numpy(emb, ["a", "b"], how="mean")
    best = core.similar_numpy(emb, ["a", "b"], how="max")
    # the mean query (0.5, 0.5) points the same way as c; c's best single match is at 45 degrees
    assert mean["c"] == pytest.approx(1.0)
    assert best["c"] == pytest.approx(1 / math.sqrt(2))
    assert best["a"] == pytest.approx(1.0)


# ------------------------------------------------------------------ Getis-Ord Gi*
def test_gi_star_hand_worked_case():
    # 19 hexes: a centre, its 6 neighbours (value 1 with the centre) and the 12 hexes of ring 2 (value 0).
    # Centre: n = 19, w = 7, sum of w*x = 7, mean = 7/19, S = sqrt(84)/19
    #   z = (7 - 7 * 7/19) / (S * sqrt((19*7 - 7**2) / 18)) = (84/19) / (84 / (19*sqrt(18))) = sqrt(18)
    centre = h3.latlng_to_cell(29.76, -95.37, 7)
    ring1 = set(h3.grid_disk(centre, 1))
    ids = sorted(h3.grid_disk(centre, 2))
    x = np.array([1.0 if h in ring1 else 0.0 for h in ids])
    g = core.gi_star(ids, x).set_index("h3")
    assert len(ids) == 19
    assert g.loc[centre, "gi_z"] == pytest.approx(math.sqrt(18))
    # flipping the values flips the sign
    g2 = core.gi_star(ids, 1 - x).set_index("h3")
    assert g2.loc[centre, "gi_z"] == pytest.approx(-math.sqrt(18))
    # two-sided normal p-value of z = sqrt(18)
    assert g.loc[centre, "gi_p"] == pytest.approx(math.erfc(math.sqrt(18) / math.sqrt(2)), rel=1e-6)
