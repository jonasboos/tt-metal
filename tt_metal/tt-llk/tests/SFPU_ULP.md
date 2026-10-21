# SFPU accuracy: the ULP sweep, and how to enrol an op in it

Every SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`. For most ops that declaration is a
tolerance. For the ops enrolled here it is a **step budget**: "every element is within
N representable values of the reference". Most unary budgets come from the exhaustive
sweep, which checks an op against *every normal value of a 16-bit input format* and its
zero -- `±inf`, NaN and the subnormals are never fed (the unpack flushes a subnormal, so a
lane there would measure the flush rather than the op), and `-0.0` is the same value as
`+0.0` -- or, for `Float32`, a strided sample of 65,024 across the whole range
(`ulp_sweep.is_exhaustive` tells the two apart). The exceptions are the `Signbit`, isinf/isnan and threshold-family
rows, measured on their functional drivers' hand-built stimuli (`MEASURED_ON_SWEEP`, after
the format table below). The isinf/isnan sweep feeds `±inf` and NaN on purpose.

This document is how you add an op to that second group.

## Quick start

```bash
cd tests/python_tests

# What the hardware measures for your op, over the whole format (a stride of Float32).
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit \
  --compile-producer -n 8
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --ulp-emit \
  --compile-consumer

# ...which rewrites your op's block in the table. Read the diff, then gate on it.
git diff helpers/sfpu_accuracy_budget.yaml
CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py --op MyOp --compile-consumer
```

`--op` matches the op name exactly. `-k MyOp` is a substring match: `-k Exp` also runs
`Exp2`, `ExpWithBase`, `Expm1` and `Expm1Cw`, and rewrites their blocks too.

`--ulp-emit` **writes the checked-in table**, once, at the end of the session; under
`-n` the controller merges every worker's measurements first. It refuses to write unless
you are on Wormhole, the session ran to the end (an interrupt, `pytest.exit()` or an
internal error writes nothing, however many grids were complete), no test in it failed
and every touched `(in, out)` grid is complete — but it is still a deliberate act, so
read the diff before committing it. Off Wormhole it refuses at startup, before
anything compiles. The producer half only compiles, and `--collect-only` runs nothing,
so both write nothing and say so. An op whose block it cannot regenerate (an op-wide
`atol`/`rtol` row on a measured cell, or an alias) is kept verbatim and named in the
summary with why; the session still passes, so a whole-table emit can end green. Each
refusal is its own exception type (`WrongArch`, `SessionNotClean`, `IncompleteGrid`,
`AmbiguousCollapse`, `UnplacedMeasurements`), and a successful emit returns a
`WriteReport` that the one-line summary is rendered from.

## Why a step budget rather than a tolerance

`atol=0.05` is about 6 bfloat16 steps at 1.0 and about 0.01 at 512. A tolerance loose
enough to pass an op's tail is blind in the middle of its range, and PCC stays above
0.99 through error levels no consumer would accept. A step count is the same strictness
everywhere.

And the sweep measures a different number than a functional driver does. The drivers
sample a few thousand points from an op's *safe* domain; a budget measured that way
describes the sample, and cannot see a tail the sample never reaches. The sweep's number
is the format's.

## What the sweep feeds

| format | as input | as output | how |
|---|---|---|---|
| `Float16_b` | yes | yes | all 65,025 values: every normal one, and zero |
| `Float16` | yes | yes | all 61,441 values: every normal one, and zero |
| `Bfp8_b` | yes | yes | swept in bfloat16, packed on the way in |
| `Bfp4_b` | yes | **no** | swept in bfloat16, packed on the way in |
| `Float32` | yes | yes | **strided**, not exhaustive — see below |

`Bfp4_b` is input-only: it keeps 2 fractional bits, so a bfloat16 step count would read
every legal quantization of a `Bfp4_b` *output* as a 32-step error.

Binary, ternary and scalar rows are measured over those drivers' own sweeps, and those
drivers gate on the whole contract, `Float32` included. So do the unary signbit, isinf/isnan
and threshold sweeps, whose hand-built stimuli are what the predicates' rows were
measured on (`MEASURED_ON_SWEEP` in `test_sfpu_accuracy_budget.py`). The isinf/isnan
and threshold predicates have no registered domain, so those sweeps are their only gate;
`Signbit`, `ReluMin` and `ReluMax` are driven by the exhaustive sweep as well.

On Wormhole and Blackhole no functional driver sweeps a `Float16` input or output, so
nothing in CI gates the `Float16` rows there: the binary ones, and the unary rows on
the tolerance metric. They stay in the table as the last measurement, and a stale one
fails no test; re-measure before relying on one.

`Float32` has 2^32 values and one device run holds 2^16, so it cannot be enumerated. The
sweep strides the format's total order instead. Every binade holds the same number of
representable values, so each gets an equal share: one run reaches 261 binades from 0 to
3.4e38. Ask `ulp_sweep.is_exhaustive(input_format)` if you need to know which you got.

On Wormhole and Blackhole an exponent-B input (`Float16_b`, `Bfp8_b`, `Bfp4_b`) packed to
`Float16` needs a 32-bit Dest, so the sweep runs those cells with `dest_acc=Yes` only:
asked for `No`, the harness would run the `Yes` kernel anyway.

The domain is deliberately **not** clipped to the op's safe range. Undefined inputs are
swept and then masked out of the statistics, so they still reach hardware.

## Lanes the sweep does not count

`measurable_mask` drops five kinds, none of them a budget question, and the driver then
keeps only the lanes the op *claims* (below):

- **either side NaN** — an op undefined at an input lands here on its own;
- **the two sides disagreeing about being non-finite** — a reciprocal overflowing where
  the golden is still finite; one such lane ranks at ~48,000 steps;
- **inputs the unpack does not deliver as fed** — below the smallest normal of the input
  format *and* of the format the unpacker writes, or past the latter's largest value
  where its exponent range is narrower: a Float32 input into a `Float16` output at
  `dest_acc=No` lands in a `Float16` Dest, which flushes below 2**-14 and saturates past
  65504 (about 44% of the strided lanes). The sweep feeds no subnormal of its own --
  the hardware flushes them and the golden does not, so `ceil(5.69e-39)` read 1 in the
  model and 0 on silicon, 16,129 bfloat16 steps, once the whole of `Ceil`'s, `Floor`'s
  and `Sqrt`'s apparent error -- but a `Bfp8_b` block's shared exponent can still make
  a swept normal subnormal on the way in, so the input is judged as the block-float
  quantizer hands it to the golden as well as as generated;
- **answers the Dest lost** (`dest_holds`) — a `Float16` input runs on an fp16 Dest,
  which tops out at 65504 and flushes below 2^-14 whatever the output. Under a `Float32`
  or `Float16_b` output the hardware hands back 65536 (or NaN) where the answer was
  70000, while the golden holds the answer (or substitutes a finite 131008 for an fp16
  infinity): 8 million fp32 steps for the Dest's overflow, not the op's. Judged on the
  pair: an overflow is excused only where the answer is NaN or has the golden's sign in
  the Dest's top binade (2^15 or more -- past 65504 the golden is a stand-in, so an
  answer near the edge has nothing true to be ranked against: `I0`'s 65440 for 65772
  is its usual 0.5%), and a flush only across the 2^-14 boundary (one side 0, the other
  at most 2^-14). A wrong-signed or small finite answer is the op's -- `Expm1Cw`'s
  #54770 answers -1 or -131008 where the golden is 131008. Only where the Dest
  is narrower than the output; under an fp16 output the two overflow together and that
  is judged as an output overflow;
- **the sweep's own zero padding** — the tensor is 65,536 lanes and the bfloat16 sweep feeds
  65,025 values (every normal one, and zero), so the last 511 are padding rather than data.

The second kind is a *failure*, not a non-question: it is dropped from the ranking only
because `nonfinite_failures` reports it, over the same claimed lanes.
`ulp_sweep.ranked_mask` ANDs `claimed_lanes` into the mask, so the budget is measured over
everywhere the op claims an answer: the whole format, less the undefined side of each
singularity `sfpu_domains._OP_SINGULARITIES` registers (`Log` below zero, `Reciprocal`
at zero), every non-positive integer for `Digamma`, `Lgamma` and `Polygamma`
(`_NONPOSITIVE_INTEGER_POLES`: the poles are a lattice, and every fp32 value of
magnitude 2^23 or more is an integer) and a per-op argument-reduction limit (`Sin` and
`Cos` and `Tan` past 65504, where the reduction breaks, on the formats that reach it:
bfloat16 and Float32, not float16, which ends there). The same limit bounds the lanes the
sweep ranks, so a range-reduction regression below it fails the gate. Not the functional driver's
sampling window, which is where points are drawn rather than where an op stops being
defined, and not `_SFPU_UNDEFINED_RANGES`, whose holes are guard bands around those
points rather than the points themselves. In `nonfinite_failures`, a golden past the
output format's range is excused only where the store saturated -- NaN, an infinity of the golden's sign, or on a
`Float16` output the pack's clamp to ±65504 of that sign (an out-of-range value from a
wider Dest packs to the format's largest magnitude). Any other finite answer to an
infinite golden is a failure.

One more exclusion is a defect already on the books rather than the sweep's doing.
`ulp_sweep._KNOWN_NONFINITE_LANES` names, per op and per cell, the inputs on which the
hardware is known to answer on the wrong side of infinity, each entry pointing at its
issue (`Celu` returns `inf` for `x` in 65408..65504 on a 16-bit `Float16` Dest, #58607).
Without it one such lane parked the whole ~64,000-lane cell as `not measurable`, which
the gate skips outright. Only the non-finite disagreement is excused: a named lane that
agrees is ranked like any other, and the gate fails the day no named lane of a cell
disagrees any more, so an entry cannot outlive its fix.

Subnormal *outputs* are ranked with the band flushed, on `Float16` as well: the golden
keeps IEEE fp16 subnormals that the pack path does not reproduce, and an exact op read
512 steps on `Float16_b -> Float16` from that band alone.

## Enrolling an op, step by step

### 1. Check it can be gated at all

A step count needs a per-element ULP, which means a float output. `ulp.has_ulp_gate(fmt)`
is the authority. Integer formats want bit equality; `Bfp4_b`, `Bfp2_b` and the MX
formats keep their block-aware lattice compares.

Your op also has to be in `sfpu_domains.sfpu_unary_ops()` and have a registered domain,
or the sweep has nothing to feed it.

### 2. Give it a block in the table

The emitter keeps an op's key line -- its name and any header comment -- as it is; it
never writes a key line itself. So a *new* op needs one by hand first. Add the name and
one placeholder row (a key with no rows fails to load, and so does a ULP row without
the `measured` figure that is its gate):

```yaml
MyOp:
  - {in: Float16_b, out: Float16_b, metric: tolerance}  # placeholder, replaced by the emit run
```

Key the placeholder on a swept `in`/`out` pair, or the emit run keeps it instead of
replacing it.

### 3. Measure it

Run the two commands under **Quick start**. The emitter rewrites your op's block with
what the hardware reported, and preserves anything it did not measure — a sampled row
for a cell the sweep does not drive, or an `arch:`-keyed entry. It writes only whole `(in, out)` grids: a
run narrowed inside one (`-k` on a single approx mode, `--maxfail`, an interrupt)
writes nothing.

### 4. Read what it wrote

```yaml
MyOp:
  - {in: Float16_b, out: Float16_b, run: 2026-09-23, domains: [{low: -.inf, high: .inf, ulp: 1}]}
  - {in: Float16, out: Float16_b, run: 2026-09-23, domains: [{low: -.inf, high: -4, ulp: 14337}, {low: -4, high: .inf, ulp: 1}]}
  - {in: Float16_b, out: Bfp8_b, metric: tolerance, measured: 393, run: 2026-09-23}
```

Every row records `run`, the day the sweep measured it (a functional driver's row says
`sampled:` instead), and what it measured: a `domains` list on every cell with a
per-element ULP, or a `measured` figure on the rest. A figure *is* the gate: there is
no headroom over it. The comment is for people. Five verdicts:

- **`domains: [{low: L, high: H, ulp: U}, …]`** — every cell with a per-element ULP,
  one list over the whole swept range ("Where an op holds its figure", below). An entry
  within the output's usable ceiling is gated at its figure: the sweep holds its lanes to
  exactly `U` steps, a reading one step worse fails it, and a re-emit that records the
  worse figure is a raised gate the PR guard reports. An entry past the ceiling is
  measured, not gated, and the headroom report holds it to its figure. `ulp: null` is
  an entry with too few ranked lanes to draw an edge on, none of them worse than the
  row's largest figure. A one-domain op gets a one-entry list, so every
  such row reads the same way. A measured 0 stays 0, on the strided `Float32` input
  too: the stride is a fixed set of values, and the figure is a measurement of it.
- **`near_zero_atol: A`** beside the list — a floor: the step count is a difference of
  bit-pattern ranks, so a lane where the hardware answers 0 for a 2e-9 the golden still
  holds reads as 2^29 fp32 steps, and one such lane demoted the whole cell. Lanes under
  1 % of the cell's largest golden (and under `A/0.01`) are judged on absolute error
  instead; `A` is exactly the largest absolute error among the band lanes past the
  ceiling, written as measured, and every entry's figure is over the lanes outside it.
  The emitter grants a floor up to `ulp_sweep.EMIT_MAX_NEAR_ZERO_ATOL` (1e-3); a cell
  that would need more keeps its list with the floor capped there, and the entries show
  where it still falls past the ceiling.
- **`measured: N`**, no `metric` — a row a functional driver sampled (the binary,
  ternary and scalar ops, and the hand-built unary sweeps). The driver holds the cell to
  exactly `N` steps.
- **`metric: tolerance, measured: N`** on a step-gateable output — a row the drivers
  sampled past `usable_budget_ceiling`, where a step budget would no longer be *tighter*
  than the tolerance it replaces. The op keeps tolerance + PCC on that cell, and
  `measured` is what the headroom report holds it to. A list whose largest figure is
  past the ceiling implies the same for its whole cell, while its entries within the
  ceiling stay gated.
- **`metric: tolerance, measured: N`** on a block-float output — block-quantized. The sweep
  enumerates a format in value order, so sixteen adjacent values share a `Bfp8_b` block
  and the exponent fits all of them: the best case for quantization, not a
  representative one. `Abs` reads **15,616 steps** there from random mixed-magnitude
  blocks (the table's `Bfp8_b` note) and **393** from the sorted sweep of a bfloat16
  input, so neither number is enrollable.
- **`metric: tolerance, measured: N, nonfinite: L`**, "not measurable: …" — `L` lanes
  disagreed with the golden about being finite where the op claims an answer (inf/NaN
  against a finite golden, or a finite answer to an infinite one), or the cell had no
  lane a step count could describe (no `measured`). No budget buys that; the row says
  why instead of leaving a hole the next emit would paper over, and `measured` is the
  worst of the lanes that *were* measurable (absent when none was), so the rest of the
  cell is not thrown away with the demotion. On a
  gateable output such a row must also be acknowledged, with its cause, in
  `test_sfpu_accuracy_budget._UNMEASURABLE_CELLS_ACKNOWLEDGED` — and if the cause is a
  tracked defect on a handful of inputs, it belongs in `_KNOWN_NONFINITE_LANES` instead,
  so the rest of the cell keeps its gate.

### 5. Gate on it

Re-run without `--ulp-emit`. Green means the budgets in the table hold over every value
of a 16-bit input and over the `Float32` stride. Then run the host guards, which check things the sweep cannot:

```bash
pytest test_sfpu_accuracy_budget.py test_ulp_sweep.py test_ulp_provenance.py -q
```

## The rules the table enforces

- **No number is a guess, and the measurement is the number.** A ULP row's gate is its
  `measured` figure, with the `run`/`sampled` date it came from; the loader refuses one
  without them. There is no headroom: a worse measurement is a raised gate, recorded by
  a re-emit in a change of its own. The comment beside a row is prose; no code reads it.
  `python3 helpers/ulp_budget_diff.py diff --base <old.yaml> --head <new.yaml>` lists
  every cell a change loosens and whether its date moved with it.
- **Most specific key wins.** A row's key fields are `in`, `out`, `approx`, `dest` and
  `arch`, all optional. Two rows matching one variant equally specifically are an
  authoring error, not a tie-break, and the loader refuses them.
- **A row holds for the configuration it was measured in.** Every exhaustive row was
  measured with `FAST_MODE(No)` and `CLAMP_NEGATIVE(True)` compiled in, and the key has
  no axis for either. The unary functional driver builds `Sqrt`/`Rsqrt` at
  `FastMode.Yes` and takes only a row's tolerance arm, which those flags do not move.
- **A gate is a measurement, swept or sampled.** The drivers seed their stimuli, so a
  sampled row's figure is reproducible and gates like the sweep's; neither carries an
  envelope.
- **The list is the statement.** A row with `domains` carries no `measured`, `metric`
  or `sampled` beside it, pins the `in` and `out` it was measured on, and names its
  `run`; the loader refuses anything else. An entry never widens or narrows what the op
  claims (`claimed_lanes`): it reports where the claim holds, and at what figure.
- **A gated cell is not parked quietly.** A `not measurable` verdict on a gateable output
  has to be acknowledged in `_UNMEASURABLE_CELLS_ACKNOWLEDGED` with its cause, or the
  lanes behind it named in `_KNOWN_NONFINITE_LANES` with their issue.
- **Budgets do not transfer between architectures.** Every unkeyed number was measured
  on Wormhole. Off it, an op falls back to tolerance unless a row names that `arch`
  itself. Re-measure before trusting any of it on Blackhole.
- **An exact op may not carry a wide budget.** Sign-bit ops, copies, integer results,
  predicates and constant fills are the flakiness canaries: if one fails, the golden or
  the datapath moved. `test_an_exact_op_never_carries_a_wide_budget` holds their
  budgets, and `test_every_swept_cell_of_an_exact_op_is_gated_or_waived` holds the cells
  demoted to tolerance to an explicit list, each with its measurement.
- **`Bfp8_b` charges for block quantization.** A budget there is denominated in bfloat16
  steps, two to one `Bfp8_b` step, and does not forgive the shared exponent. No cell
  with a `Bfp8_b` output is gated by steps: the sweep records its measurement and the
  cell stays on tolerance, whose lattice compare is the stronger criterion there.

## Not enrolling an op

Two ways, and they mean different things:

- **No row in the table** — the op resolves to today's tolerance and the sweep does not
  collect it.
  Fine as an interim state; from #57520, `test_every_unary_op_is_enrolled_or_excused`
  fails until you pick one of these two deliberately.
- **`sfpu_domains._UNARY_OPS_NOT_SWEPT`** — the op cannot be swept at all. Say why in a
  comment next to it.

## A budget that is nearly right

If an op is exact over most of its range and catastrophic in a narrow band near zero,
the measurement will be dominated by the band and the cell will demote to tolerance.
`near_zero_atol` is the floor under the budget for exactly that: the lanes where the
reference crosses zero, judged on absolute error instead of steps. The emitter derives
it from the measurement (`ulp_sweep.near_zero_floor`) and writes it on the row, so a
re-emit regenerates a floor row like any other; the only blocks still kept verbatim are
the op-wide `atol`/`rtol` anchors of `GeluAppx` and `SigmoidAppx`, which the summary
names. A floor past `EMIT_MAX_NEAR_ZERO_ATOL` is a kernel's cut-off rather than its
rounding -- `Softplus` answers 0 below x = -5 where the answer is up to 0.0065, the
approximate `Gelu` is 0.024 off -- and stays a hand decision: the demoted row records
the floor it would have needed.

## Where an op holds its figure: domains

One line per format combination says how the op behaves over the whole swept range:

```yaml
Tanh:
  - {in: Float16_b, out: Float16_b, approx: "Yes", dest: "No", run: 2026-10-20, domains: [{low: -.inf, high: -1, ulp: 3}, {low: -1, high: 1, ulp: 10}, {low: 1, high: .inf, ulp: 3}]}
Log:
  - {in: Float16_b, out: Float16_b, dest: "No", run: 2026-10-20, domains: [{low: -.inf, high: 0, ulp: null}, {low: 0, high: .inf, ulp: 1}]}
Abs:
  - {in: Float16_b, out: Float16_b, dest: "No", run: 2026-10-20, domains: [{low: -.inf, high: .inf, ulp: 0}]}
```

Read: approximate `tanh` is 3 steps everywhere except on `[-1, 1)`, where it is 10,
past bfloat16's ceiling of 6 — that one entry is why the cell as a whole is not
gated, and the two others are gated at 3. `log` answers nothing below 0 and is 1 step
from 0 up. `abs` is exact over the whole range.

- **An entry** `{low: L, high: H, ulp: U}` holds the lanes whose input, *as the kernel
  received it* (`ulp_sweep.received_inputs`: block-quantized for `Bfp8_b`/`Bfp4_b`,
  truncated to the 16-bit Dest for a `Float32` input at `dest: "No"`), satisfies
  `L <= x < H`. `U` is the worst ranked lane there, outside the row's near-zero floor
  if it has one; `null` means too few lanes there are ranked to draw an edge on (fewer
  than `DOMAIN_MIN_BIN_LANES` in each of its binades): past the op's claim (`cos`
  beyond 65504), undefined (`log` below 0, every value below -128 for `lgamma`), or an
  answer the Dest cannot hold. The few ranked lanes a `null` entry can still hold are
  gated at the row's largest figure, so the emitter writes `null` only while they fit
  it; otherwise the entry whose strays are worst carries its own figure. A cell where
  no entry would carry a figure gets a plain row, not a list. Which lanes an op claims stays in code
  (`_CLAIM_LIMIT`, `_OP_SINGULARITIES`, `_NONPOSITIVE_INTEGER_POLES`); the list reports
  it.
- **Gated or not follows the figure.** An entry within `usable_budget_ceiling` of the
  output (6 for `Float16_b`, 51 for `Float16`, 419,430 for `Float32`) is gated at it;
  one past the ceiling is measured and held to its figure by the headroom report. The
  whole-cell contract `accuracy_contract` hands to callers that do not look at domains
  (the functional drivers) is derived: gated at the largest figure when every entry is
  within the ceiling, tolerance otherwise.
- **The list is contiguous and complete**: the first `low` is `-.inf`, the last `high`
  is `.inf`, each `high` is the next `low`, and every interior edge is `0` or a power
  of two, written in its shortest exact form (`0.00390625`, `65536`,
  `8.507059173023462e+37`).
- **The emitter draws it** from the sweep's per-lane distances (`ulp_domains`): the
  worst ranked lane per signed binade of the received input (`[2^k, 2^(k+1))`, or
  `[-2^(k+1), -2^k)`, so -1 opens `[-1, -0.5)` and no binade straddles an edge), split
  where it crosses the ceiling (`DOMAIN_CLASS_BITS` would split the gateable range into step classes as
  well; the binary rule is the question asked, and factor-2 classes gave eight times
  the entries for the pack path's 0-vs-1 flicker between binades), a binade with fewer
  than `DOMAIN_MIN_BIN_LANES` lanes folded into its neighbour, a gateable domain under
  `DOMAIN_MIN_LANES` ranked lanes folded into the neighbour at least as loose (a bad
  region is never hidden that way), an edge with no swept lane between it and zero
  written as `0`. A lane the row's floor covers counts as ranked at 0 steps when the
  list is drawn -- it is held, by the floor -- so the near-zero band draws no entry of
  its own, and an entry the floor covers entirely reads `0`. Every entry's figure is
  then measured over exactly its lanes outside the floor, so what is written is what
  the gate holds them to.
- **The sweep gates entry by entry.** Each gateable entry's lanes are held to its
  figure with the row's floor; the ranked lanes a `null` entry leaves (strays in
  otherwise unranked binades) are held to the row's largest figure, so no ranked lane
  goes unjudged. A failure names the entry. Where that figure is past the ceiling the
  sweep gates no fallback, and the headroom report holds those lanes to it instead.
- **The headroom report judges entry by entry too**: the `--ulp-measure` row carries
  the reading over each entry of the table's list (`domains`) and the per-binade profile
  it was drawn from (`bins`), and `python3 helpers/ulp_budget_diff.py domains --measured
  ulp.jsonl [--op Cos] [--table helpers/sfpu_accuracy_budget.yaml]` prints, from that
  artifact and no device, the list the rule would draw today beside the one the table
  holds.
- **The PR guard judges segment by segment**: over the union of both revisions' edges,
  an entry raised, a list replaced by a looser one, or a plain row turned into a list
  are each reported for exactly the lanes they change, with the range on the line, and
  a list redrawn with the same figures is no change at all.

The hand-written `atol`/`rtol` anchors (`GeluAppx`, `SigmoidAppx`) are kept verbatim
by the emitter and so carry no list until those anchors move onto rows.

## Gotchas

- **Any host pytest run in `tests/` wipes `/tmp/tt-llk-build`.** Chain
  `--compile-producer` and `--compile-consumer` in one go; if you run a host suite in
  between, recompile.
- **The sweep is marked `accuracy`**, which every LLK workflow deselects. Run it by name
  or by `-m accuracy`. `nightly` would *not* have kept it out of `llk-e2e`.
- **Both runs take the same op set**: every unary op with a key line in the table,
  whatever its rows say. The key line is the enrolment, so an op on tolerance everywhere
  is measured on every run and held to the figures its rows record by the headroom
  report, and an op with no block is not swept at all (step 2 adds the block first).
  `--ulp-emit` changes what is written, not what is collected, so the producer and the
  consumer build the same set with or without it.
- **A plain gate run does not judge tolerance cells.** It fails a gated cell over its
  budget and *skips* a tolerance cell, with the measured maximum in the skip reason. To
  hold those cells to the figures their rows record, measure and compare:

  ```bash
  CHIP_ARCH=wormhole pytest test_unary_sfpu_ulp.py -n auto --ulp-measure=ulp.jsonl
  python3 helpers/ulp_budget_diff.py headroom \
    --table helpers/sfpu_accuracy_budget.yaml --measured ulp.jsonl
  ```
- **`CHIP_ARCH` must be set**, and `--ulp-emit` refuses to run on anything but
  Wormhole, because `_render` does not emit `arch` and the rows would be badged wrongly.

## Where things live

| file | what |
|---|---|
| `python_tests/test_unary_sfpu_ulp.py` | the harness: one device run per variant |
| `python_tests/helpers/ulp.py` | the metric: `ulp_distance`, `within_ulp`, the near-zero floor |
| `python_tests/helpers/ulp_sweep.py` | what the sweep feeds, what it masks, and the emitter |
| `python_tests/helpers/sfpu_accuracy_budget.yaml` | the table |
| `python_tests/helpers/sfpu_accuracy_budget.py` | how a row is resolved |
| `python_tests/helpers/ulp_provenance.py` | the table read as rows, and each row's provenance fields |
| `python_tests/helpers/ulp_domains.py` | the `domains` list: its meaning, the rule that draws it, its one spelling |
| `python_tests/helpers/ulp_budget_diff.py` | the CI comparison, and the headroom report (from #57527) |
