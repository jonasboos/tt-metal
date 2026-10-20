# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_domains.py``: what a ``domains`` list means, the
rule that draws one from a cell's binade profile, and its one spelling.

No device and no torch: the rule is pure, and a profile is a dict. The tensor side
(``ulp_sweep.binade_maxima``, ``domain_figures``) is held in test_ulp_sweep.py.
"""

import math

import pytest
import yaml
from helpers.ulp_domains import (
    BINADE_OFFSET,
    CEILINGS,
    DOMAIN_MIN_BIN_LANES,
    DOMAIN_MIN_LANES,
    Domain,
    binade,
    binade_edges,
    check_domains,
    contract_of,
    domains_text,
    number_text,
    partition,
)

INF = math.inf
BF16 = CEILINGS["Float16_b"]


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


def _profile(worst_of, lanes=128):
    """A bfloat16-like profile: 254 binades a side, *lanes* present in each, the worst
    ranked lane from ``worst_of(k, sign)`` (``None`` for an unranked binade), and the
    single zero lane."""
    bins = {}
    for k in range(-126, 128):
        for sign in (1, -1):
            worst = worst_of(k, sign)
            index = sign * (k + BINADE_OFFSET)
            bins[index] = (
                lanes,
                0 if worst is None else lanes,
                -1 if worst is None else worst,
            )
    bins[0] = (1, 1, 0)
    return bins


def test_binades_order_as_the_values_do():
    assert binade(-0.3) < binade(-1e-30) < binade(0.0) < binade(1e-30) < binade(0.3)
    assert binade_edges(binade(0.3)) == (0.25, 0.5)
    assert binade_edges(binade(-0.3)) == (-0.5, -0.25)
    assert binade_edges(binade(1.0)) == (1.0, 2.0)
    assert binade(2.0**-126) == -126 + BINADE_OFFSET == 24  # bf16's smallest normal
    assert binade_edges(0) == (0.0, 0.0)


def test_a_flat_profile_is_one_entry_over_the_whole_range():
    assert partition(_profile(lambda k, s: 1), BF16) == ((-INF, INF, True),)


def test_a_profile_splits_where_the_worst_lane_crosses_the_ceiling():
    """Approximate tanh: 10 steps below |x| = 1, 3 steps above; bf16's ceiling is 6."""
    spans = partition(_profile(lambda k, s: 10 if k < 0 else 3), BF16)
    assert spans == ((-INF, -1.0, True), (-1.0, 1.0, True), (1.0, INF, True))
    entries = (Domain(-INF, -1, 3), Domain(-1, 1, 10), Domain(1, INF, 3))
    assert contract_of(entries, BF16) == ("tolerance", 10)
    assert contract_of(entries[:1] + (Domain(-1, INF, 5),), BF16) == ("ulp", 5)


def test_a_step_class_flickering_between_binades_is_one_entry():
    """The pack path rounds some binades and not others, so the worst lane alternates
    between 0 and 1 from one binade to the next on hundreds of cells. Both are within
    the ceiling, and the binary rule does not tell them apart."""
    assert partition(_profile(lambda k, s: k % 2), BF16) == ((-INF, INF, True),)


def test_an_unranked_region_is_its_own_entry_with_zero_for_an_edge():
    """`log` below zero: no lane is ranked. The edge between the two is where the
    swept values stop, which is written as 0 rather than 2**-126."""
    spans = partition(_profile(lambda k, s: None if s < 0 else 1), BF16)
    assert spans == ((-INF, 0.0, False), (0.0, INF, True))


def test_a_stray_ranked_lane_in_an_unranked_binade_is_noise():
    """erfinv's [1, 2) binade has one ranked lane, at x = 1: fewer than
    DOMAIN_MIN_BIN_LANES, so the binade is unranked and the list ends at 1."""

    def profile():
        bins = _profile(lambda k, s: 4 if k < 0 else None)
        for sign in (1, -1):
            bins[sign * BINADE_OFFSET] = (128, 1, 0)  # [1, 2) and (-2, -1]: one lane
        return bins

    assert partition(profile(), BF16) == (
        (-INF, -1.0, False),
        (-1.0, 1.0, True),
        (1.0, INF, False),
    )


def test_a_narrow_gateable_island_folds_into_its_looser_neighbour_but_a_bad_one_stays():
    """One good binade (128 lanes, under DOMAIN_MIN_LANES) inside a bad range is not
    advertised on its own; one bad binade inside a good range always is."""
    assert DOMAIN_MIN_LANES > 128 >= DOMAIN_MIN_BIN_LANES
    good_island = partition(_profile(lambda k, s: 2 if (k, s) == (3, 1) else 10), BF16)
    assert good_island == ((-INF, INF, True),)
    bad_island = partition(_profile(lambda k, s: 10 if (k, s) == (3, 1) else 1), BF16)
    assert bad_island == ((-INF, 8.0, True), (8.0, 16.0, True), (16.0, INF, True))


def test_the_zero_lane_never_forms_an_entry():
    for spans in (
        partition(_profile(lambda k, s: 10 if k < 0 else 3), BF16),
        partition(_profile(lambda k, s: None if s < 0 else 1), BF16),
        partition({0: (1, 1, 0)}, BF16),
    ):
        assert all(low < high for low, high, _ in spans)
        assert any(low <= 0 < high for low, high, _ in spans)


def test_the_profile_of_a_strided_float32_input_partitions_the_same_way():
    """~129 samples per binade rather than 128 values: the rule reads lanes, not
    values, so nothing about it is bf16-specific."""
    spans = partition(
        _profile(lambda k, s: 500000 if k > 20 else 2, lanes=129), CEILINGS["Float32"]
    )
    assert spans == (
        (-INF, -(2.0**21), True),
        (-(2.0**21), 2.0**21, True),
        (2.0**21, INF, True),
    )


_OK = [
    {"low": -INF, "high": -1, "ulp": 3},
    {"low": -1, "high": 1, "ulp": 10},
    {"low": 1, "high": INF, "ulp": 3},
]


def test_a_well_formed_list_loads_as_its_entries():
    assert check_domains("Tanh row 1", _OK) == (
        Domain(-INF, -1, 3),
        Domain(-1, 1, 10),
        Domain(1, INF, 3),
    )
    assert check_domains(
        "t", [{"low": -INF, "high": 0, "ulp": None}, {"low": 0, "high": INF, "ulp": 1}]
    )


@pytest.mark.parametrize(
    "value, problem",
    [
        ([], "non-empty list"),
        ("[]", "non-empty list"),
        ([{"low": -INF, "high": INF}], "exactly the keys"),
        ([{"high": INF, "low": -INF, "ulp": 1}], "in that order"),
        ([{"low": -INF, "high": INF, "ulp": 1, "note": "x"}], "exactly the keys"),
        ([{"low": -INF, "high": INF, "ulp": True}], "integer >= 0 or null"),
        ([{"low": -INF, "high": INF, "ulp": -1}], "integer >= 0 or null"),
        ([{"low": -INF, "high": INF, "ulp": 1.5}], "integer >= 0 or null"),
        ([{"low": 1, "high": 1, "ulp": 0}], "low < high"),
        ([{"low": -1, "high": INF, "ulp": 0}], "cover \\(-inf, inf\\)"),
        ([{"low": -INF, "high": 1, "ulp": 0}], "cover \\(-inf, inf\\)"),
        (_OK[:1] + [{"low": 0, "high": INF, "ulp": 1}], "contiguous"),
        (
            [{"low": -INF, "high": 3, "ulp": 1}, {"low": 3, "high": INF, "ulp": 1}],
            "0 or a power of two",
        ),
        ([{"low": -INF, "high": INF, "ulp": None}], "no entry carries a figure"),
    ],
    ids=[
        "empty",
        "not-a-list",
        "missing-key",
        "keys-out-of-order",
        "extra-key",
        "ulp-bool",
        "ulp-negative",
        "ulp-float",
        "low-not-below-high",
        "no-minus-inf",
        "no-plus-inf",
        "gap",
        "edge-not-a-power-of-two",
        "all-null",
    ],
)
def test_a_malformed_list_says_what_is_wrong(value, problem):
    with _refuses(problem):
        check_domains("Tanh row 1", value)


def test_the_one_spelling_reads_back_as_the_same_numbers():
    """``.inf`` as YAML reads it, integral values without a decimal point, anything
    else as ``repr``: what the emitter writes, PyYAML loads back as the same floats."""
    assert number_text(INF) == ".inf" and number_text(-INF) == "-.inf"
    assert number_text(0.0) == "0" and number_text(65536.0) == "65536"
    assert (
        number_text(-1.0) == "-1"
        and number_text(3) == "3"
        and number_text(None) == "null"
    )
    assert number_text(2.0**-8) == "0.00390625"
    assert number_text(2.0**126) == "8.507059173023462e+37"
    text = domains_text(check_domains("t", _OK))
    assert (
        text
        == "[{low: -.inf, high: -1, ulp: 3}, {low: -1, high: 1, ulp: 10}, {low: 1, high: .inf, ulp: 3}]"
    )
    assert yaml.safe_load("domains: " + text)["domains"] == _OK
    assert domains_text(_OK) == text  # the loaded mappings spell the same


def test_the_ceilings_are_the_registrys():
    """The slim runner cannot import the registry, so the ceilings are spelled out here;
    they must be ``usable_budget_ceiling``'s to the step."""
    from helpers.format_config import DataFormat
    from helpers.sfpu_accuracy_budget import usable_budget_ceiling

    assert CEILINGS == {
        name: usable_budget_ceiling(DataFormat[name]) for name in CEILINGS
    }
