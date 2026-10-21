# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where an op holds its figure: the domains of a swept cell.

One line of ``sfpu_accuracy_budget.yaml`` says how an op behaves over the whole swept
range of one format combination::

    - {in: Float16_b, out: Float16_b, approx: "Yes", dest: "No",
       domains: [{low: -.inf, high: -1, ulp: 3}, {low: -1, high: 1, ulp: 10},
                 {low: 1, high: .inf, ulp: 3}]}

An entry ``{low: L, high: H, ulp: U}`` holds the lanes whose input, as the kernel
received it, satisfies ``L <= x < H``. ``U`` is the worst ranked lane there, in steps,
outside the row's near-zero floor if it has one -- the same quantity a plain row's
``measured`` is, over fewer lanes -- or ``null`` where too few lanes are ranked to draw
an edge on (fewer than :data:`DOMAIN_MIN_BIN_LANES` in each of its binades): past the
op's claim (``cos`` beyond 65504), undefined (``log`` below 0), or an answer the Dest
cannot hold. The sweep holds the few ranked lanes a null entry has to the row's largest
figure, so the emitter writes ``null`` only while they fit it (:func:`entries_of`).
An entry whose figure is within the output format's usable ceiling
(:data:`CEILINGS`) is gated at it by the sweep; one past the ceiling is measured and
held to its figure by the headroom report. The whole-cell contract a caller gets is
derived: gated at the largest figure when every ranked entry is within the ceiling,
tolerance otherwise. A row with a list carries no cell-level ``measured`` or
``metric``: the list is the statement.

The list is a contiguous partition of ``(-inf, inf)``: the first ``low`` is ``-.inf``,
the last ``high`` is ``.inf``, each ``high`` is the next ``low``, and every interior edge
is ``0`` or a power of two. The emitter draws it from the sweep's per-lane distances
(:func:`partition`); nothing else writes it, and a one-domain op gets a one-entry list,
so every per-element row has the same shape.

Standalone on purpose -- the standard library only -- because the PR budget guard
(``ulp_budget_diff.py``) runs on a slim runner and imports it by path, and the registry
loader, the table renderer and the guard share one rule here so they cannot drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

#: The row field that holds the list.
DOMAINS_FIELD = "domains"

#: What a row with a list may not carry: the list is the whole statement of what was
#: measured and what gates, so a cell-level figure or metric beside it would be a second
#: statement to drift from the first. The registry's loader and the PR guard both read
#: it here.
NOT_BESIDE_DOMAINS = ("measured", "metric", "atol", "rtol", "nonfinite")

#: An entry's keys, in the order they are written.
ENTRY_KEYS = ("low", "high", "ulp")

#: ``sfpu_accuracy_budget.usable_budget_ceiling`` by format name: the largest figure
#: that is still *tighter* than the tolerance it replaces. Its keys are the formats a
#: row may name (``ulp.ULP_FORMATS``), in and out: the SFPU computes in 16- and 32-bit
#: floats, and a block float's error is the unpacker's and packer's, not the op's.
#: Spelled out because this module may not import the registry; a host test ties them.
CEILINGS: Dict[str, float] = {
    "Float16_b": 6.0,
    "Float16": 51.0,
    "Float32": 419430.0,
}

#: How finely the gateable range is split. ``None`` is the binary rule: a binade is
#: either within the ceiling or past it, and adjacent binades on the same side merge.
#: An integer *n* would split the gateable range into step classes a factor of ``2**n``
#: apart as well (0 and 1 merged). Measured on a full sweep, factor-2 classes gave
#: 27,741 extra entries, almost all of them the worst lane flickering between 0 and 1
#: step from one binade to the next where the pack path rounds; the binary rule gave
#: 3,472, and is the question asked: where is the op safe.
DOMAIN_CLASS_BITS: Optional[int] = None

#: A binade with fewer lanes *present* than this (the single zero lane; a format's last
#: partial binade) joins its neighbour; one with that many lanes but fewer *ranked* is
#: unranked -- the stray ranked lane at x = 1 in erfinv's [1, 2) binade is noise, and
#: the list ends its ranked domain at 1.
DOMAIN_MIN_BIN_LANES = 16

#: A gateable domain with fewer ranked lanes than this (1.6% of a 16-bit format) is
#: folded into the neighbour at least as loose as itself, so a one-binade island does
#: not become an entry; a domain past the ceiling is never folded into a gateable one,
#: so a narrow bad region is never hidden.
DOMAIN_MIN_LANES = 1024

#: Makes every float32 binade index positive, so the sign of the index is the sign of
#: x: the binade of the smallest subnormal, -2**-149, is ``[-2**-149, -2**-150)``, k =
#: -150 (see :func:`binade`), so ``k + 151`` is at least 1.
BINADE_OFFSET = 151

_INF = float("inf")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Domain:
    """One entry: the lanes with ``low <= x < high``, and the worst of them in steps,
    or ``None`` where none is ranked."""

    low: float
    high: float
    ulp: Optional[int]

    def __post_init__(self) -> None:
        if not (_is_number(self.low) and _is_number(self.high)):
            raise ValueError(f"a domain's edges must be numbers, got {self!r}")
        if not self.low < self.high:
            raise ValueError(f"a domain needs low < high, got {self!r}")
        if self.ulp is not None and not (_is_int(self.ulp) and self.ulp >= 0):
            raise ValueError(
                f"a domain's ulp must be an integer >= 0 or null, got {self!r}"
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))

    def gated(self, ceiling: float) -> bool:
        """Whether the sweep holds this entry's lanes to its figure."""
        return self.ulp is not None and self.ulp <= ceiling

    def as_entry(self) -> Dict[str, Any]:
        return {"low": self.low, "high": self.high, "ulp": self.ulp}


def binade(x: float) -> int:
    """The signed binade index of a finite input: 0 for zero, otherwise
    ``sign(x) * (k + BINADE_OFFSET)`` for the *k* whose binade (:func:`binade_edges`)
    holds x under the lists' own ``low <= x < high``: ``2**k <= x < 2**(k+1)``, or
    ``-2**(k+1) <= x < -2**k``. A negative power of two is the *low* edge of the binade
    above it, as -1 is of ``[-1, -0.5)``, so its k is one less than its magnitude's;
    otherwise a binade's lanes would straddle an entry's edge. Orders binades as the
    values do."""
    if x == 0:
        return 0
    mantissa, exponent = math.frexp(abs(x))
    k = exponent - 1
    if x < 0 and mantissa == 0.5:
        k -= 1
    index = k + BINADE_OFFSET
    return index if x > 0 else -index


def binade_edges(index: int) -> Tuple[float, float]:
    """``[low, high)`` of the binade *index* names: ``[2**k, 2**(k+1))`` on the positive
    side, ``[-2**(k+1), -2**k)`` on the negative, ``[0, 0)`` for zero's own."""
    if index == 0:
        return (0.0, 0.0)
    k = abs(index) - BINADE_OFFSET
    return (2.0**k, 2.0 ** (k + 1)) if index > 0 else (-(2.0 ** (k + 1)), -(2.0**k))


#: One binade of a cell: lanes present, lanes ranked, worst ranked lane (-1 if none).
Bin = Tuple[int, int, int]


def _step_class(worst: int, ceiling: float):
    if worst > ceiling:
        return "past"
    if DOMAIN_CLASS_BITS is None:
        return "gateable"
    return 1 + int(math.floor(math.log2(max(worst, 1)) / DOMAIN_CLASS_BITS))


def _looseness(cls) -> float:
    if cls == "past":
        return _INF
    return 0.0 if cls == "gateable" else float(cls)


def partition(
    bins: Dict[int, Bin], ceiling: float
) -> Tuple[Tuple[float, float, bool], ...]:
    """The domains the rule draws on a cell's binade profile, as ``(low, high, ranked)``
    triples covering ``(-inf, inf)``; the caller measures each ranked domain's figure over
    its own lanes.

    Per binade: fewer than :data:`DOMAIN_MIN_BIN_LANES` lanes present means it takes the
    class of the binade before it (or after, at the start); fewer ranked means it is
    unranked; otherwise its worst lane is within the ceiling or past it (or, with
    :data:`DOMAIN_CLASS_BITS`, in a step class). Adjacent binades of one class merge; a
    gateable run under :data:`DOMAIN_MIN_LANES` ranked lanes folds into the neighbour at
    least as loose as itself, the loosest first, and never into an unranked one. The
    outer edges are infinite; an edge with no swept lane between it and zero is
    written as 0.
    """
    if not bins:
        return ((-_INF, _INF, False),)
    labelled: List[list] = []
    for index in sorted(bins):
        present, ranked, worst = bins[index]
        if present < DOMAIN_MIN_BIN_LANES:
            cls = None
        elif ranked < DOMAIN_MIN_BIN_LANES:
            cls = "unranked"
        else:
            cls = _step_class(worst, ceiling)
        labelled.append([index, ranked, cls])
    last = None
    for row in labelled:
        if row[2] is None:
            row[2] = last if last is not None else "pending"
        else:
            last = row[2]
    nxt = "unranked"
    for row in reversed(labelled):
        if row[2] == "pending":
            row[2] = nxt
        else:
            nxt = row[2]
    runs: List[dict] = []
    for index, ranked, cls in labelled:
        if runs and runs[-1]["cls"] == cls:
            runs[-1]["last"] = index
            runs[-1]["ranked"] += ranked
        else:
            runs.append({"first": index, "last": index, "cls": cls, "ranked": ranked})
    changed = True
    while changed:
        changed = False
        narrow = sorted(
            (
                i
                for i, r in enumerate(runs)
                if r["cls"] not in ("unranked", "past")
                and r["ranked"] < DOMAIN_MIN_LANES
            ),
            key=lambda i: runs[i]["ranked"],
        )
        for i in narrow:
            own = _looseness(runs[i]["cls"])
            candidates = [
                j
                for j in (i - 1, i + 1)
                if 0 <= j < len(runs)
                and runs[j]["cls"] != "unranked"
                and _looseness(runs[j]["cls"]) >= own
            ]
            if not candidates:
                continue
            j = max(candidates, key=lambda j: _looseness(runs[j]["cls"]))
            lo, hi = min(i, j), max(i, j)
            runs[lo : hi + 1] = [
                {
                    "first": runs[lo]["first"],
                    "last": runs[hi]["last"],
                    "cls": runs[j]["cls"],
                    "ranked": runs[lo]["ranked"] + runs[hi]["ranked"],
                }
            ]
            changed = True
            break
    merged: List[dict] = []
    for r in runs:
        if merged and merged[-1]["cls"] == r["cls"]:
            merged[-1]["last"] = r["last"]
            merged[-1]["ranked"] += r["ranked"]
        else:
            merged.append(dict(r))
    # The smallest magnitude any swept lane has: an edge at or below it has nothing
    # between it and zero, and reads better as 0.
    smallest = min(
        (
            abs(edge)
            for index, (present, _, _) in bins.items()
            if index != 0 and present > 0
            for edge in binade_edges(index)
        ),
        default=0.0,
    )
    out: List[Tuple[float, float, bool]] = []
    for k, r in enumerate(merged):
        low = -_INF if k == 0 else binade_edges(r["first"])[0]
        high = _INF if k == len(merged) - 1 else binade_edges(r["last"])[1]
        if 0 < abs(low) <= smallest:
            low = 0.0
        if 0 < abs(high) <= smallest:
            high = 0.0
        ranked = r["cls"] != "unranked"
        if out and not low < high:
            # A run of nothing but the zero lane, snapped to [0, 0): dropped rather than
            # written empty. The next entry's `low` is pulled down to 0 below, so under
            # `low <= x < high` the zero lane lands in the domain after it.
            continue
        if out and out[-1][1] != low:
            low = out[-1][1]
        out.append((low, high, ranked))
    return tuple(out)


def entries_of(
    spans: Tuple[Tuple[float, float, bool], ...],
    worst: Callable[[float, float], Optional[int]],
) -> Tuple[Domain, ...]:
    """The entries the table writes for :func:`partition`'s *spans*, given *worst(low,
    high)*: the worst ranked lane in ``[low, high)`` outside the floor (0 where the floor
    covers every one), ``None`` where none is ranked.

    A ranked span carries its worst lane. An unranked one is ``null``, but it can still
    hold a few ranked lanes, and the sweep holds those to the row's largest figure; so
    if any are worse than every figure in the list, the span whose strays are worst
    carries its own figure instead, and the next run on the same output passes the list
    it was written with. ``()`` -- no list, a plain row -- when no ranked span measured
    anything: ``check_domains`` refuses a list without a figure."""
    figures = [worst(low, high) if ranked else None for low, high, ranked in spans]
    if all(f is None for f in figures):
        return ()
    largest = max(f for f in figures if f is not None)
    strays = [
        (stray, i)
        for i, (low, high, ranked) in enumerate(spans)
        if not ranked
        for stray in (worst(low, high),)
        if stray is not None
    ]
    if strays:
        stray, i = max(strays)
        if stray > largest:
            figures[i] = stray
    return tuple(Domain(low, high, f) for (low, high, _), f in zip(spans, figures))


def _is_power_of_two(value: float) -> bool:
    return value != 0 and math.isfinite(value) and math.frexp(abs(value))[0] == 0.5


def check_domains(where: str, value: Any) -> Tuple[Domain, ...]:
    """*value*, a row's ``domains`` as PyYAML loaded it, as :class:`Domain` entries --
    or a ``ValueError`` naming what is wrong. The registry's loader and the PR guard
    both call this, so the guard cannot pass a table the registry refuses to load."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{where}: `{DOMAINS_FIELD}` must be a non-empty list of entries"
        )
    domains: List[Domain] = []
    for i, entry in enumerate(value):
        name = f"{where}: `{DOMAINS_FIELD}` entry {i + 1}"
        if not isinstance(entry, dict) or tuple(entry) != ENTRY_KEYS:
            raise ValueError(
                f"{name} must be a mapping with exactly the keys "
                f"{', '.join(ENTRY_KEYS)}, in that order; got {entry!r}"
            )
        try:
            domains.append(Domain(entry["low"], entry["high"], entry["ulp"]))
        except ValueError as bad:
            raise ValueError(f"{name}: {bad}") from None
    if domains[0].low != -_INF or domains[-1].high != _INF:
        raise ValueError(
            f"{where}: the list must cover (-inf, inf): the first `low` is -.inf and "
            "the last `high` is .inf"
        )
    for before, after in zip(domains, domains[1:]):
        if before.high != after.low:
            raise ValueError(
                f"{where}: the list must be contiguous; an entry ends at "
                f"{number_text(before.high)} and the next starts at "
                f"{number_text(after.low)}"
            )
        if before.high != 0 and not _is_power_of_two(before.high):
            raise ValueError(
                f"{where}: an interior edge is 0 or a power of two, got "
                f"{number_text(before.high)}"
            )
    if all(d.ulp is None for d in domains):
        raise ValueError(
            f"{where}: no entry carries a figure; nothing to hold the cell to"
        )
    return tuple(domains)


def contract_of(
    domains: Tuple[Domain, ...], ceiling: float
) -> Tuple[str, Optional[int]]:
    """The whole-cell contract a list implies: ``("ulp", figure)`` -- gated at the largest
    figure -- when every ranked entry is within *ceiling*, else ``("tolerance",
    figure)``. What a caller that does not look at domains is handed. A list with no
    figure at all (which the loader refuses) reads as tolerance with none."""
    figures = [d.ulp for d in domains if d.ulp is not None]
    figure = max(figures) if figures else None
    return ("ulp" if figure is not None and figure <= ceiling else "tolerance"), figure


def number_text(value: Any) -> str:
    """The one spelling of an edge or a figure: ``.inf``/``-.inf`` as YAML reads them,
    an integral value without a decimal point, anything else as ``repr``, which is the
    shortest text that reads back as the same float."""
    if value is None:
        return "null"
    if value == _INF:
        return ".inf"
    if value == -_INF:
        return "-.inf"
    if _is_int(value):
        return str(value)
    if float(value).is_integer() and abs(value) < 2**53:
        return str(int(value))
    return repr(float(value))


def entry_text(entry: Dict[str, Any]) -> str:
    return "{" + ", ".join(f"{k}: {number_text(entry[k])}" for k in ENTRY_KEYS) + "}"


def domains_text(value: Any) -> str:
    """A ``domains`` value -- :class:`Domain` entries or the mappings PyYAML loads -- as
    it is written on the row."""
    entries = [d.as_entry() if isinstance(d, Domain) else d for d in value]
    return "[" + ", ".join(entry_text(e) for e in entries) + "]"
