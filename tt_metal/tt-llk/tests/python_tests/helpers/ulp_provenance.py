# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SFPU accuracy table as rows, and each row's provenance as fields.

Every row of ``sfpu_accuracy_budget.yaml`` that carries a budget says which measurement
it came from in fields of its own -- ``measured`` and ``nonfinite`` -- which PyYAML
loads and the registry loader validates like any other field. On a ULP row
``measured`` is the gate itself: there is no second number. A row the exhaustive sweep
wrote carries its ``domains`` list instead (``ulp_domains``), and no cell-level figure
at all. The comment beside a row is prose for people; no code reads it.
:class:`Provenance` is the typed view of the fields, and :class:`BudgetTable` is the one
walk over the file, locating rows through PyYAML's own event stream rather than by the
shape of a line.

Standalone on purpose -- ``yaml`` and the standard library -- because the PR budget
guard (``ulp_budget_diff.py``) runs on a slim runner and imports it by path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

if __package__:
    from .ulp_domains import CEILINGS, DOMAINS_FIELD, Domain, contract_of, domains_text
else:  # run by path on the slim runner, as ulp_budget_diff.py imports this module
    from ulp_domains import CEILINGS, DOMAINS_FIELD, Domain, contract_of, domains_text

#: The key dimensions of a row, in the order a cell is named.
KEY_FIELDS = ("in", "out", "approx", "dest", "arch")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


#: libyaml's loader when PyYAML was built with it: the table is parsed twice per load,
#: and the pure-Python loader takes over a second for each.
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

#: The tag PyYAML gives a ``<<`` merge key, which each walker below treats apart.
_MERGE_TAG = "tag:yaml.org,2002:merge"


def refuse_duplicate_keys(node: yaml.Node, where: str) -> None:
    """PyYAML keeps the last of two identical mapping keys without a word, so a
    copy-pasted op name would drop the earlier op's whole table. Checked on the composed
    node tree, before anything is constructed. ``<<`` merge keys are skipped: overriding
    a merged field is what they are for."""
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if key.tag == _MERGE_TAG:
                continue
            if key.value in seen:
                raise ValueError(
                    f"{where} line {key.start_mark.line + 1}: duplicate entry for "
                    f"{key.value!r}. YAML keeps only the "
                    "last, so the earlier one would vanish with nothing to catch it."
                )
            seen.add(key.value)
            refuse_duplicate_keys(value, where)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            refuse_duplicate_keys(item, where)


def _merged_keys(node: yaml.Node) -> set:
    """The keys a ``<<`` value hands a mapping: those of the mapping it names, or of each
    in a sequence of them, their own merges included."""
    if isinstance(node, yaml.SequenceNode):
        return set().union(*(_merged_keys(item) for item in node.value))
    if not isinstance(node, yaml.MappingNode):
        return set()
    keys = set()
    for key, value in node.value:
        if key.tag == _MERGE_TAG:
            keys |= _merged_keys(value)
        else:
            keys.add(key.value)
    return keys


def refuse_merged_provenance(node: yaml.Node, where: str) -> None:
    """A ``<<`` merge row is a cell of its own, so its measurement must be its own too:
    a provenance field that reaches it only through the merge would credit it with the
    anchor's figure. A plain ``*alias`` row *is* the anchor's row, and shares
    its measurement by design. Checked on the composed node tree, before construction
    resolves the merges away."""
    if isinstance(node, yaml.MappingNode):
        own, merged = set(), set()
        for key, value in node.value:
            if key.tag == _MERGE_TAG:
                merged |= _merged_keys(value)
            else:
                own.add(key.value)
                refuse_merged_provenance(value, where)
        borrowed = sorted((merged - own) & set(PROVENANCE_FIELDS))
        if borrowed:
            raise ValueError(
                f"{where} line {node.start_mark.line + 1}: "
                f"{', '.join(f'`{k}`' for k in borrowed)} reaches a row only "
                "through a `<<` merge key. A merged row is a cell of its own and needs "
                "its own measurement fields; a plain `*alias` shares the anchor's."
            )
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            refuse_merged_provenance(item, where)


def load_yaml(text: str, where: str) -> Any:
    """*text* as PyYAML's safe loader reads it, refusing a repeated mapping key, a
    top-level ``<<`` merge, and provenance borrowed through a merge key."""
    loader = _LOADER(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        if isinstance(node, yaml.MappingNode) and any(
            key.tag == _MERGE_TAG for key, _ in node.value
        ):
            # Its ops would load with no key line of their own: every row walk (the
            # guard's, the emitter's) would miss them, and the borrowed-provenance check
            # below never looks inside a merge value.
            raise ValueError(
                f"{where}: a top-level `<<` merge key. Give each op its own key line."
            )
        refuse_duplicate_keys(node, where)
        refuse_merged_provenance(node, where)
        return loader.construct_document(node)
    finally:
        loader.dispose()


@dataclass(frozen=True)
class Provenance:
    """The measurement behind a row, as the row's own fields.

    * ``measured`` -- the worst lane, in steps: over every ranked lane, or for a row with
      a ``near_zero_atol`` floor, over the lanes outside it. On a ULP row it is the gate;
      there is no headroom over it. On a row the sweep could not measure, the worst of
      the lanes it could.
    * ``nonfinite`` -- lanes disagreeing with the golden about being finite, which hold
      the row's cell on tolerance; the headroom report fails a run in which it grows.
    """

    measured: Optional[int] = None
    nonfinite: Optional[int] = None

    @classmethod
    def of(cls, values: Dict[str, Any]) -> "Provenance":
        """The provenance fields of a loaded row, unchecked: a reader of an arbitrary
        revision takes what is there, and :meth:`problems` is the loader's check."""
        return cls(**{k: values[k] for k in PROVENANCE_FIELDS if k in values})

    @property
    def unmeasurable(self) -> bool:
        """Whether lanes disagreeing with the golden about being finite held the row's
        cell on tolerance."""
        return self.nonfinite is not None

    def as_fields(self) -> Dict[str, Any]:
        """The fields to write, in table order, without the unset ones."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def problems(self) -> List[str]:
        """What is wrong with these fields on their own, as messages; ``[]`` if
        nothing. Combinations with the contract are the loader's to check."""
        return [
            f"`{name}` must be an integer >= {least}, got {value!r}"
            for name, least in (("measured", 0), ("nonfinite", 1))
            for value in (getattr(self, name),)
            if value is not None and not (_is_int(value) and value >= least)
        ]


#: The provenance fields of a row, in the order the emitter writes them: the
#: :class:`Provenance` fields, so the two cannot drift.
PROVENANCE_FIELDS = tuple(f.name for f in fields(Provenance))


def check_provenance(where: str, row: Dict[str, Any], *, tolerance: bool) -> None:
    """A row's provenance fields, against themselves and its contract; *tolerance* is
    whether the row's resolved metric is the tolerance one. A ULP row's gate *is* its
    ``measured`` figure, so a ULP row without one is a guess, and the table's rule is
    that no number in it is. The registry's loader and the PR guard both call this, so
    the two cannot disagree about provenance."""
    provenance = Provenance.of(row)
    problems = provenance.problems()
    if not tolerance and provenance.measured is None:
        problems.append("a ulp row's gate is its `measured` figure; the row has none")
    if provenance.nonfinite is not None and not tolerance:
        problems.append(
            "`nonfinite` lanes hold a cell on tolerance, so it belongs on a "
            "`metric: tolerance` row"
        )
    if problems:
        raise ValueError(f"{where} {row}: " + "; ".join(problems))


def key_value(value: Any) -> str:
    """A key field as the registry reads it. YAML 1.1 reads a bare ``Yes``/``No`` as a
    boolean, and the registry maps it back to the enum member."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return str(value)


def key_field(field: str, value: Any) -> str:
    """:func:`key_value` for *field*; an arch is accepted by enum name or value and named
    by the enum name, as the registry and a measurement name it."""
    text = key_value(value)
    return text.upper() if field == "arch" else text


@dataclass(frozen=True)
class Row:
    """One row of the table: its fields as PyYAML loads them, and where it sits.

    *first_line*/*last_line* are 0-based indices into the file, for rewriting it and
    for messages."""

    op: str
    fields: Tuple[Tuple[str, Any], ...]
    first_line: int
    last_line: int
    #: The prose after the row's ``#``, for people. No code reads it.
    comment: str = ""
    #: A ``- *anchor`` row, or one of an op whose whole row list is an alias (``Neg:
    #: *low``, located on the key line): its fields are not inline, so it pins nothing a
    #: reader of the line could see.
    alias: bool = False

    @property
    def values(self) -> Dict[str, Any]:
        return dict(self.fields)

    @property
    def provenance(self) -> Provenance:
        return Provenance.of(self.values)

    @property
    def key(self) -> Tuple[Tuple[str, str], ...]:
        """The key dimensions the row pins, as ``(field, value)`` in KEY_FIELDS order."""
        values = self.values
        return tuple(
            (k, key_field(k, values[k]))
            for k in KEY_FIELDS
            if values.get(k) is not None
        )

    @property
    def pinned(self) -> Dict[str, str]:
        return dict(self.key)

    @property
    def domains(self) -> Tuple[Domain, ...]:
        """The row's ``domains`` as entries, or ``()`` on a row without a list -- or
        with one this reader cannot make sense of; like ``provenance`` it is unchecked,
        and ``ulp_domains.check_domains`` is the loader's refusal."""
        value = self.values.get(DOMAINS_FIELD)
        if not isinstance(value, list):
            return ()
        try:
            return tuple(Domain(e["low"], e["high"], e["ulp"]) for e in value)
        except (TypeError, KeyError, ValueError):
            return ()

    @property
    def ceiling(self) -> Optional[float]:
        """The usable ceiling of the row's output format, or ``None`` when the row does
        not pin one with a per-element ULP."""
        return CEILINGS.get(self.pinned.get("out", ""))

    @property
    def metric(self) -> str:
        """``ulp`` or ``tolerance``: the field, or on a row with a list what the list
        implies (``ulp_domains.contract_of``)."""
        domains = self.domains
        if domains:
            ceiling = self.ceiling
            return "tolerance" if ceiling is None else contract_of(domains, ceiling)[0]
        return str(self.values.get("metric", "ulp"))

    @property
    def max_ulp(self) -> Optional[int]:
        """The step budget -- the row's ``measured`` figure, which is the gate, or on a
        row with a list the largest figure in it when every entry is within the ceiling
        -- or ``None`` on the tolerance metric."""
        domains = self.domains
        if domains:
            ceiling = self.ceiling
            if ceiling is None:
                return None
            metric, figure = contract_of(domains, ceiling)
            return figure if metric == "ulp" else None
        value = self.values.get("measured")
        if self.metric == "tolerance" or not _is_int(value):
            return None
        return value

    @property
    def unmeasurable(self) -> bool:
        """Whether the sweep could not measure the row's cell: ``Provenance.unmeasurable``,
        except that a row with a list measured its cell and says so entry by entry."""
        return not self.domains and self.provenance.unmeasurable

    @property
    def near_zero_atol(self) -> Optional[float]:
        value = self.values.get("near_zero_atol")
        return value if isinstance(value, (int, float)) else None

    def describe(self) -> str:
        pinned = ", ".join(f"{k}: {v}" for k, v in self.key)
        return f"{self.op} {{{pinned}}}" if pinned else f"{self.op} (default)"


@dataclass(frozen=True)
class OpBlock:
    """Where an op sits in the file: its key line, the prose after it, and its rows."""

    op: str
    line: int
    rows: Tuple[Row, ...]
    header: str = ""


class BudgetTable:
    """The table as rows, located through PyYAML's event stream: values mean what the
    loader says they mean (anchors, ``<<`` merges, YAML booleans), and a
    row's comment is the text after ``#`` past the end of its node. Tolerant of a
    top-level entry that is not a list of rows -- the guard parses arbitrary base
    revisions; ``_load_table`` is what refuses them."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.blocks: Dict[str, OpBlock] = {}
        #: The file as :func:`load_yaml` reads it, for a reader that checks the values.
        self.loaded: Any = None
        self._parse()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BudgetTable":
        return cls(Path(path).read_text(encoding="utf-8"))

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def rows(self) -> List[Row]:
        return [row for block in self.blocks.values() for row in block.rows]

    def rows_of(self, op: str) -> List[Row]:
        block = self.blocks.get(op)
        return list(block.rows) if block else []

    def row(self, op: str, **key: str) -> Row:
        """The one row of *op* pinning exactly *key*. Key fields are spelt as in the
        table, with ``in_`` for ``in``: ``row("Gelu", in_="Float16", out="Float16")``.
        """
        names = ("in_", "out", "approx", "dest", "arch")
        unknown = set(key) - set(names)
        if unknown:
            raise TypeError(f"not key fields: {sorted(unknown)}")
        wanted = tuple(
            (k, key[name]) for k, name in zip(KEY_FIELDS, names) if name in key
        )
        found = [r for r in self.rows_of(op) if r.key == wanted]
        if len(found) != 1:
            raise LookupError(f"{op}: {len(found)} rows pin {dict(wanted)}, not one")
        return found[0]

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _comment_after(self, mark) -> str:
        """The comment on *mark*'s line past *mark*, without the ``#``; ``""`` if none."""
        line = self.lines[mark.line] if mark.line < len(self.lines) else ""
        _, sep, comment = line[mark.column :].partition("#")
        return comment.strip() if sep else ""

    def _parse(self) -> None:
        loaded = self.loaded = load_yaml(self.text, "budget table")
        if not isinstance(loaded, dict):
            return
        events = list(yaml.parse(self.text, Loader=_LOADER))
        i = 0
        # Down to the top-level mapping.
        while i < len(events) and not isinstance(events[i], yaml.MappingStartEvent):
            i += 1
        i += 1
        while i < len(events) and not isinstance(events[i], yaml.MappingEndEvent):
            key_event = events[i]
            i += 1
            if not isinstance(key_event, yaml.ScalarEvent):
                i = _skip_node(events, _skip_node(events, i - 1))
                continue
            op = key_event.value
            positions: List[Tuple[Any, Any, bool]] = []
            if isinstance(events[i], yaml.SequenceStartEvent):
                i += 1
                while not isinstance(events[i], yaml.SequenceEndEvent):
                    start = events[i]
                    end_index = _skip_node(events, i)
                    end = events[end_index - 1]
                    if (
                        isinstance(start, yaml.MappingStartEvent)
                        and not start.flow_style
                        and end_index - 2 > i
                    ):
                        # A block-style row's MappingEnd sits at the next token, the
                        # next row's line; its last value is where the row ends.
                        end = events[end_index - 2]
                    positions.append(
                        (
                            start.start_mark,
                            end.end_mark,
                            isinstance(start, yaml.AliasEvent),
                        )
                    )
                    i = end_index
                i += 1
            else:
                start = events[i]
                i = _skip_node(events, i)
                if isinstance(start, yaml.AliasEvent) and isinstance(
                    loaded.get(op), list
                ):
                    # `Neg: *low`: the whole row list is an anchor's. Each row is there
                    # all the same, with the budget the registry enforces, so it is read
                    # as an alias row on the key line rather than left out -- which hid
                    # `*low` -> `*high` from the guard.
                    positions = [
                        (start.start_mark, start.end_mark, True) for _ in loaded[op]
                    ]
            entries = loaded.get(op)
            rows: List[Row] = []
            if isinstance(entries, list) and len(entries) == len(positions):
                for values, (start, end, alias) in zip(entries, positions):
                    if not isinstance(values, dict):
                        continue
                    rows.append(
                        Row(
                            op=op,
                            fields=tuple(values.items()),
                            first_line=start.line,
                            last_line=end.line,
                            comment=self._comment_after(end),
                            alias=alias,
                        )
                    )
            self.blocks[op] = OpBlock(
                op=op,
                line=key_event.start_mark.line,
                rows=tuple(rows),
                header=self._comment_after(key_event.end_mark),
            )
        # Fail closed: an op the loader resolves but the walk cannot place would read as
        # no rows on both sides of a diff, and a raised budget in it would compare clean.
        missing = sorted(str(op) for op in loaded if str(op) not in self.blocks)
        if missing:
            raise ValueError(f"budget table: cannot locate op(s) {missing}")


def _skip_node(events: list, i: int) -> int:
    """The index just past the node starting at *events[i]*."""
    depth = 0
    while True:
        event = events[i]
        i += 1
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
        if depth == 0:
            return i


def _field_text(key: str, value: Any) -> str:
    if key in ("approx", "dest"):
        return f'{key}: "{value}"'  # quoted so YAML keeps them strings
    if key == DOMAINS_FIELD:
        # ``repr`` would spell an infinity ``inf``, which YAML reads as a string.
        return f"{key}: {domains_text(value)}"
    return f"{key}: {value}"


def render_row(row_fields: Dict[str, Any], comment: str = "") -> str:
    """One inline row as the emitter writes it, with its newline. ``approx``/``dest``
    quoted so YAML keeps them strings; a ``domains`` list in ``ulp_domains``' one
    spelling."""
    body = ", ".join(_field_text(k, v) for k, v in row_fields.items())
    return f"  - {{{body}}}" + (f"  # {comment}" if comment else "") + "\n"
