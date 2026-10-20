# Wan2.2 VAE decode regression on BH galaxy: investigation notes

Issue: https://github.com/tenstorrent/tt-metal/issues/60670

Status: **Not a tt-metal code regression (confirmed 2026-10-19 by CI probe A).** The last good
nightly's exact commit and Docker image now reproduce the slow VAE. The cause is an environment
change on the BH single-galaxy CI machines around the morning of 2026-09-15. See "Probe results"
directly below. The earlier sections are kept for history; their code-window narrowing is
superseded.

## Probe results (2026-10-17, dispatched `models-t1-e2e-tests.yaml`, `wan-2.2-t2v`, `bh_sc1`)

| Probe | Ref | Docker image | 480p VAE | 720p VAE | Run |
|---|---|---|---|---|---|
| 09-14 nightly (original) | `f8c4ce59dd0` | `2be5f2ab…` | 0.324 s | 0.676 s | 34799396959 |
| **A** | `f8c4ce59dd0` (same commit) | `2be5f2ab…` (same image) | **0.752 s** | **1.021 s** | 38959412196 |
| B | `656f87fcdb4` | n/a | 0.726 s | 1.015 s | 38959415458 |
| C | `656f87fcdb4` + UMD `f70cc57f7f8` | n/a | infra failure, no test ran | n/a | 38959418717 |

- Same code and same image as the good run, yet slow now. The tt-dit weight cache is cold
  ("Cache does not exist ... Writing cache") in every run, old and new, so a stale persisted cache
  is ruled out. The Hugging Face fetch counts are identical.
- **Machine change on 2026-09-15:** in the 09-15 nightly (run 34921216342, created 02:27 UTC), every
  `bh_sc1` job waited about 10.5 h and started between 12:56 and 13:18 UTC. That suggests the pool
  was down or under maintenance that morning. The first jobs after it came back are the first slow
  ones. Their pytest timeouts are consistent with fresh machines recompiling every kernel.
- KMD/firmware alone don't explain it. 09-14 (good) and 09-16 (bad) both report KMD 2.10.0 and
  firmware bundle 19.12.0. The probes ran on today's machines (pool `multihost-ci-sc1-r4rn9`,
  KMD 2.11.0, firmware 19.16.0) and are slow there too. The change is something else on the hosts
  or devices that persisted across this later upgrade: host OS or kernel, CPU frequency or power
  settings, PCIe/IOMMU or hugepage configuration, board power or thermal settings, or the
  hardware itself.
- The probe A denoise time (39.5 s at 480p vs about 57 s on 09-14) also differs by host, matching
  the bimodal runner behaviour noted in the issue.

### Probe A-HP: same code on the 14 kW (`bh_sc1_high_power`) pool, 2026-10-20

Branch `jameslee/probe-60670-A-hp-f8c4ce59dd0` (`f8c4ce59dd0` plus two lines that route only the
Wan T2V sc1 job to `bh_sc1_high_power`). Run 39341458236, job 118369560488, host `bh-glx-120-d07u08`
(runner `multihost-ci-sc1-hp-*`), KMD 2.11.0, firmware 19.16.0, AICLK settled at 1343 MHz.

| Test | 09-14 good (`bh_sc1`) | Probe A (`bh_sc1`, 10-17) | **Probe A-HP (14 kW, 10-20)** |
|---|---|---|---|
| perf 480p VAE | 0.324 s | 0.752 s | timeout (1770 s, first-run compile on a fresh host) |
| perf 720p VAE | 0.676 s | 1.021 s | **0.972 s** |
| pipeline 720p `[>>] vae` section | 0.655 s | n/a | **0.984 s** |
| perf 720p denoise | 157.2 s | 125.9 s | 123.9 s |

**The high-power pool does not restore the old VAE time.** It is at most about 5% better than the
regular pool and still about 45% slower than 09-14. Power class (8 kW vs 14 kW) is therefore not
the explanation. tt-train's MFU gap between the pools (#56782) is real, but it is a different
effect.

## Update 2026-10-20 (overnight investigation)

### Prior work: issue #58457 (mtairum, 2026-09-29)

#58457 reached the same conclusion three weeks earlier. Old code (`f8c4ce59dd0`) with the new
SFPI was slow, and new code with the old SFPI and old Docker image was slow, so the cause is
outside the source tree and the image. The UMD bump was also reverted on 09-18 (`b99aa035f39`),
and nightlies 09-19 to 09-25 ran with the pre-window UMD and were still slow. Their proposed next
step, a per-stage profile of one decode, was never posted. Facts from that issue used here:

- About 215 device ops per decode: 35 `conv3d`, 33 `neighbor_pad_async`, 23 `to_layout` each way,
  30 `dit_rms_norm_unary_fused`, 4 `all_gather_async`.
- Unchanged on the same nights and machines: Flux.2 VAE decode (traced, 0.154–0.158 s), Wan DiT
  denoise (eager, about 160k ops, 56.9 s), Wan text encoder (0.08 s).
- I2V: VAE decode +0.34 s and VAE **encode** +0.28 s, so the cost is in the VAE's ops, not the
  large readback.

### The same physical hosts got slower (fleet-wide change)

The job logs print the host (`Detected Hosts: bh-glx-120-…`). Across all 196 sc1 Wan T2V/I2V jobs
from 09-05 to 10-20:

| Host | Before | After |
|---|---|---|
| `bh-glx-120-d06u02` | 09-08: 480p 0.288 s, 720p 0.579 s; 09-10 I2V 0.673 s | 09-21: 0.731/1.022 s; 10-02: 0.751/1.000 s; 10-17 (probe A): 0.752/1.021 s |
| `bh-glx-120-d03u08` | 09-09 I2V 0.676 s | 09-15: 720p 1.148 s |

Every run up to **2026-09-14 03:33 UTC** is fast (480p 0.29–0.43 s). Every run from
**2026-09-15 13:18 UTC** onward is slow (0.68–0.85 s), on every host in the pool and on both
denoise speed classes (≈40 s and ≈57 s). Machines were not swapped; all of them changed.

The CI setup flow (`ttop-recovery`: IPMI reset of all cards, then cluster validation and link
traffic test, "All Detected Links are healthy") is identical in the 09-14 and 09-16 logs. KMD,
firmware and the set of UMD/Metal/Fabric log messages are also identical once numbers are
normalised.

### Candidate: host CPU power management

This dev host (`bh-glx-120-b10u02`, same fleet, reproduces 0.689 s) runs `acpi-cpufreq` with the
`schedutil` governor (1.5–3.25 GHz, cores idling at 1.5 GHz). The deepest idle state is **C2 with
an 800 µs exit latency**, enabled, and the kernel command line is only `iommu=pt` (no
`processor.max_cstate` or `idle=poll`). The KMD has `power_policy=Y`,
`idle_power_down_grace_ms=5000`. If the 09-15 maintenance changed a BIOS or OS power profile, a
host thread that sleeps between device operations pays up to about 0.8 ms per wake-up. A few
hundred such waits in an eager, device-bound VAE would add about 0.4 s, while leaving the
throughput-bound denoise and the short text encoder unchanged.

Probe D1 tests this directly, along with other candidates. **D1 ruled C-states out** (see the next
section).

## Root cause (2026-10-20): torch OpenMP threads collide with Weka's busy-polling cores

### Probe D1: the extra time is in the readback, not the decoder

Branch `jameslee/probe-60670-D1-diag`, run 39385870365, host `bh-glx-120-d04u14`. The VAE runs
alone (no transformer, no denoise) on `f8c4ce59dd0`.

| Stage | 480p | 720p |
|---|---|---|
| Host prep + upload of latents | 0.05 s | 0.07–0.11 s |
| **Decoder forward** (35 `conv3d`, 33 `neighbor_pad_async`, …) | **0.245 s** | 0.515 s |
| **`float_to_uint8` + `.cpu()` + host reassembly** | **0.40–0.47 s** | **0.41–0.43 s** |
| Total (same as the perf test's `vae` metric) | 0.70–0.77 s | 1.00–1.04 s |

- The decoder forward alone (0.245 s) is less than the whole good-era VAE (0.324 s). The device
  ops and collectives are fine.
- `neighbor_pad` forced to 1 link: no gain (0.79 s). A heat soak before decoding: no change.
- 64 `SCHED_IDLE` spinner processes (one per CPU, which removes C-states): **worse**, 1.43 s. The
  slow part is CPU-contention-sensitive.
- Note: the `HOSTENV` shell dump in D1 ran on the GitHub runner pod, not on the galaxy host. In
  the multihost flow, the job command runs on the pod and `mpirun` starts the test on the host.

### Probe D2: it is `_reassemble_2d`, a host-side torch loop

Branch `jameslee/probe-60670-D2-d2h`, run 39435646808, host `bh-glx-120-d04u08` (AMD EPYC 9354P,
32 cores / 64 threads, kernel 6.8.0-106, `iommu=pt`, KMD 2.11.0, PCIe Gen5, 28 chips x1 and 4 chips
x8).

| Readback step | 480p | 720p |
|---|---|---|
| `float_to_uint8` (device) | 2 ms | 2 ms |
| `.cpu(blocking=False)` call + sync (the actual D2H) | 6–7 ms | 8–10 ms |
| `_to_torch_zero_copy` × 32 | 0.5 ms | 0.4 ms |
| **`_reassemble_2d`** (32 strided `permute().contiguous()` copies into the output) | **424–449 ms** | **396–468 ms** |

The D2H transfer itself is fast: 537 MB from 32 chips in 25 ms. A single torch permute of the whole
230 MB video takes 35 ms on the same host in the same process. The reassembly is about 13 ms per
shard regardless of size.

### Local reproduction, pure torch, no device (`bh-glx-120-b10u02`, same host type)

Replicating `_reassemble_2d` with 32 random uint8 shards (480p shapes):

| torch threads / CPUs | reassemble | whole-tensor permute |
|---|---|---|
| 1 / 64 | 109 ms | 101 ms |
| 16 / 64 | 15 ms | 13 ms |
| 32 / 64 | **16 ms** | 12 ms |
| **64 / 64** | **433 ms** (720p: 483 ms) | 18 ms |
| 60 / 60 (CPUs 1, 33, 2, 34 excluded) | **16.7 ms** | 11 ms |

On this host, `wekanode` (the Weka storage client) has two threads pinned to **CPU 1 and CPU 2 at
100%**. Everything else is idle (mean busy 4%). CPUs 33 and 34 are their SMT siblings.
`/opt/weka` and `/etc/wekaio` were created here on 2026-09-21.

### Mechanism

1. `models/tt_dit/tests/conftest.py` sets `torch.set_num_threads(os.cpu_count())`, which is 64
   on these hosts. That line has not changed since March 2026, and both good and bad CI logs print
   `Setting torch num_threads to 64`.
2. The galaxy hosts now run the Weka client with two polling threads pinned at 100% on CPUs 1 and
   2, which also slows their SMT siblings.
3. Each of `_reassemble_2d`'s 32 per-shard copies is an OpenMP parallel region. Each region waits
   at its barrier for the threads that share a core with a Weka poller, about 13 ms per shard, so
   **+0.4 s per VAE pass**. One large op (the whole-tensor permute, or anything in denoise) pays
   that wait once and barely notices, which is why only the Wan VAE regressed. The I2V VAE encode
   path does similar host-side torch work.

This fits every earlier observation: the same code, same image and same KMD/firmware are slow on
every host after the change; it is a fixed cost independent of resolution; it is Blackhole-galaxy
specific (WH galaxy hosts are a different fleet); it is not C-states (spinners make it worse).

### Probe E: confirmed on a CI host

Branch `jameslee/probe-60670-E-torch-threads`, run 39467552240, host `bh-glx-120-d04u08`,
code `f8c4ce59dd0` plus a `TT_DIT_TORCH_THREADS` override in the conftest.

| | 32 torch threads | 64 torch threads (default), same host |
|---|---|---|
| Per-CPU load (`/proc/stat`) | **CPU 1 and CPU 2 at 100%**, mean 3.9% | same |
| 720p perf test VAE | **0.607 s, PASS** (target 0.85) | 1.02–1.05 s in every run since 09-15 |
| 480p perf test VAE | none (first test on a cold host hit the 1800 s timeout) | **0.777 s, FAIL** (target 0.42) |

Denoise (157 s at 720p) and text encoding (0.08 s) are unaffected. The CI host has the same two
always-busy cores as the dev host. The container cannot see `/opt/weka`, so the Weka install date
on the CI hosts is still unknown (the `/mnt/weka` mount point is dated 10-05 on both hosts, which is
not the install date).

### Fix

Branch `jameslee/fix-60670-tt-dit-torch-threads` (off `main`, commit `221e1c81915`):
`models/tt_dit/tests/conftest.py` sets torch to one thread per **physical** core in the process
affinity (counting SMT siblings once) instead of `os.cpu_count()`. On these hosts that is 32.
Validation run on bh_sc1: 39484292256 (full Wan T2V sc1 job: perf 480p/720p + 720p pipeline).

**Open question for infra:** when the Weka client (busy-polling CPUs 1–2) was deployed to the
`multihost-ci-sc1-*` galaxy hosts. The regression starts between 2026-09-14 03:33 and
2026-09-15 13:18 UTC, across the 09-15 morning outage.

**Asks:**

1. CI/infra: what changed on the BH sc1 (`multihost-ci-sc1-*`) machines on 2026-09-14/15?
2. Re-baseline the `vae` targets in `t2v_metrics()` for BH 4x8 to the current machines (about
   0.75 s at 480p and 1.05 s at 720p, plus tolerance), so the job goes green and the 720p
   pipeline quality test runs again. Do this unless infra can restore the old configuration. These findings come from static analysis
of the commit window and from CI job logs. Nothing was run on hardware (the local galaxy was
unavailable).

## Summary so far

- The regression is **Blackhole-specific**. WH galaxy 4x8 runs the same test on the same untraced,
  no-warmup path, and its VAE time did not move across the boundary.
- A CI run on a branch based on mid-window commit `656f87fcdb4` already shows the regression.
  That narrows the window from 53 commits to the ~24 in `f8c4ce59dd0..656f87fcdb4` and **clears**
  the suspects listed in the issue: the tilize Metal 2.0 port (`93f77385b40`), the llk-helper
  copy/typecast migration (`8cf12808550`), the all_gather scatter-write fix (`e44536782e1`) and
  SFPI 7.77.0 (`e4a93d6371c`).
- In the narrowed window, the only change that is BH-specific and could plausibly affect this
  path is the **UMD bump `0d472904340`** (`tt_metal/third_party/umd` `f70cc57f7f8 -> b2a5f606c87`).
  It is the current lead, but I have not yet found the mechanism.
- The missing 480p number on 09-15 is a side effect. On 09-15, the T2V 480p and I2V 720p sc1
  tests both hit pytest timeouts. That fits `0eb79bc47b4` (runtime binary reload), whose
  `dev_msgs.h` layout change invalidates every cached kernel, so the first run recompiles all of
  them. This is not the VAE regression.

## Evidence

### VAE decode times from nightly logs (`test_performance_wan.py`, `t2v`)

| Job | 09-14 `f8c4ce59dd0` | 09-15 `004157dd545` | 09-16 `df15dfd17d5` |
|---|---|---|---|
| BH sc1 `ring_bh_4x8` 480p | 0.324 s | timeout (recompile) | 0.761 s |
| BH sc1 `ring_bh_4x8` 720p | 0.676 s | 1.148 s | 1.046 s |
| WH galaxy `wh_4x8` 480p | 1.138 s | (setup error) | 1.128 s |
| BH sc4 `bh_4x32` 720p, ranks 0-3 | 0.77 / 0.46 / 0.78 / 0.59 s | 0.80 / 0.77 / 0.81 / 0.89 s | 0.82 / 0.81 / 0.85 / 0.87 s |
| BH quietbox_2 `2x2` 480p | 4.24 s | 2.98 s | 4.27 s (noisy, no clear change) |

The BH 4x8 increase is roughly a constant +0.4 s at both 480p and 720p. BH sc4 (4x32) also moved
up somewhat.

The I2V sc1 job flipped from pass on 09-14 (VAE 0.69 s at 720p) to fail on 09-15 (timeout).

### Mid-window data point

`nkapre/matmul-registry-productized` (head `1b992daddf2`, merge base `656f87fcdb4`, 09-14). Its
diff only adds an opt-in matmul/AGMM config registry plus CI changes:

| Job | 480p VAE |
|---|---|
| 105012202248 | 0.753 s |
| 105013804062 | 0.818 s |

Both are already regressed. So the culprit is in `f8c4ce59dd0..656f87fcdb4`, unless that
branch's own changes are responsible, which is unlikely.

Other branch runs in the following days have merge bases after `004157dd545` and add no
information.

### How the test measures VAE time

- `traced = mesh_shape == (4, 32)` in `test_performance_wan.py`. On 4x8 there is no tracing and
  the test's warmup iteration is skipped. The pipeline constructor already warms up the VAE with
  `uint8` output: the `BuildKernels | compiled ...` log lines for tilize, `eltwise_binary_scalar`,
  clamp, untilize and `typecast_rm_chunked` all appear before "Running warmup".
- No `BuildKernels | compiled` lines appear inside the timed run, in either the good or the bad
  log. So the time is not JIT compile in the timed section.
- The sets of compiled kernels before the timed run are essentially the same in the good and bad
  runs (only small differences in tilize / `tilize_metal2` / `cq_dispatch` variant counts).
- On BH the VAE is not coresident-excluded, so `reload_weights()` inside `decode()` does nothing,
  and no weight upload falls in the timed section.
- The timed section includes `fast_device_to_host` (`float_to_uint8`, then `.cpu()`, then host
  reassembly). Host↔device transfer behaviour is therefore in scope.

### Commits in `f8c4ce59dd0..656f87fcdb4` and why each is ruled out or still open

| Commit | Change | Verdict |
|---|---|---|
| `668c2907575` | Sharded concat rank fix | Only changes factory selection for **sharded** inputs. The VAE's concats take the interleaved path. Unlikely. |
| `5dfe136d485` | msda fixed point | Not used by the VAE. |
| `ce240a3229e` | high_bw_all_gather open path | Not used by the VAE. |
| `294bd9a0436` | gather reader clamp | Not used by the VAE. |
| `9a1b4e565bb` | sparse SDPA attention sink | Only touches the attention-sink path in `compute_streaming.hpp`; the VAE has no sinks. |
| `0d472904340` | **UMD bump** | **Open, lead suspect.** Only BH-specific candidate. See below. |
| `668752a4056` | Comparison mode for mesh tensors (`decorators.py`) | Only touches comparison-mode code; `FastOperation.__call__` is unchanged. Unlikely. |
| `70f840f46ad` | SDPA decode correction in FP32 half-sync | Decode-only and FP32-half-sync-only. Unlikely. |
| `a8fe40d9439`, `fdfad84bffc`, `bd5b7a7a7aa` | LLK sanitizer hooks (pack untilize, unpack AB, WH fast tilize) | `SAN_HOOK` reduces to an empty `do {} while (false)` when the sanitizer is disabled, and WH got the same hooks without regressing. Unlikely, but codegen was not checked. |
| `656f87fcdb4` | Scalar-LHS binary | The tensor-scalar path is unchanged apart from the extra `scalar_is_lhs=false` hash field. Unlikely. |
| others | Models / CI / targets / Quasar only | Not relevant. |

### UMD bump contents (`f70cc57f7f8..b2a5f606c87`)

- tt-umd#3399: adds an `IoOrdering` parameter and flips `LocalChip::write_to_device` /
  `read_from_device`'s reconfigured-window **fallback** from Relaxed to **Strict**. The PR states
  the cost: "11x to 16x on the bulk write path on wormhole_b0, 28x to 30x on blackhole. Reads are
  unaffected". It only applies to targets not covered by a static TLB. Metal configures static
  TLBs for Tensix, ETH and DRAM (`include_dram_tlbs = true` on silicon, `tt_cluster.cpp`), so the
  fast-dispatch hot path should not reach the fallback. **Still need to confirm** which path, if
  any, hits it during VAE decode / D2H on BH.
- tt-umd#3406: routes `SiliconTlbWindow::configure` through `execute_safe` when IoSafety is
  enabled. Matters only if windows are reconfigured often.
- tt-umd#3210, #3380, #3381: telemetry accessors moved to `DeviceFirmware`, ARC APB accessors
  retired, `configure_iatu_region` deleted. Metal's `dispatch_telemetry.cpp` calls
  `tt_device.read_from_device` / `write_to_device`, so check how often that runs.
- tt-umd#3349, #3414, #3416, #3417: cluster_id, test renames, CI. Not relevant.

## Update: environment ruled out, more commits cleared

- **Not an environment change.** The mid-window branch runs executed on 09-16/17, after the
  regression date, so on their own they can't separate a code change from a runner change.
  However, the good run, the bad nightly and the branch run all report the same
  `KMD version: 2.10.0`, `Established firmware bundle version: 19.12.0` and `IOMMU: enabled`,
  and the runners come from the same `multihost-ci-sc1-mn9pv` pool.
- **Docker image is not the cause.** The nightly image changed between 09-14 (`2be5f2ab…`) and
  09-16 (`8c782511…`), but the regressed branch run (105012202248) used the *old* image
  `2be5f2ab…`.
- **The UMD submodule commit tracks good/bad** in all three runs (good `f70cc57f`, bad and branch
  `b2a5f606`). That is consistent with the UMD bump, but not proof.
- **LLK sanitizer hooks are cleared.** The hooks are live only with `-DLLK_SAN_ENABLE`
  (`TT_METAL_LLK_SANITIZER`), and that flag appears in none of the CI kernel compile lines.
- **No mechanism found yet for the UMD bump.** Metal never calls `create_io_window`, so the bump
  affects Metal only through `LocalChip::write_to_device` / `read_from_device` falling back to a
  reconfigured window, which is now Strict. That fallback is used only for targets with no static
  TLB. Metal sets up static TLBs (`tt_metal/llrt/tlb_config.cpp`) for all Tensix, ETH, DISPATCH
  and L2CPU cores. On BH it also sets one 4 GiB DRAM window per channel, on the *last* NOC port
  only and only when the chip has 8 channels. Fast-dispatch buffer reads and writes go through the
  CQ and hugepages, and the prefetcher uses a static TLB. The direct `Cluster::write_core` /
  `read_core` / barrier calls I found are init-time, sockets, global CBs or slow dispatch, so none
  of them is on the VAE's per-op path. Dispatch telemetry is read on demand only.
- The `test_pipeline_wan` 720p leg shows the same jump in its `[>>] vae` / `[<<] vae` events:
  0.655 s (09-14) vs 1.090 s (09-16).

## CI bisect plan (no local hardware needed)

`models-t1-e2e-tests.yaml` can be dispatched with `model=wan-2.2-t2v`, `sku=bh_sc1 (BH Single
Galaxy)`. The workflow file at both `f8c4ce59dd0` and `656f87fcdb4` accepts these inputs. Each run
takes about 45 min on one sc1 galaxy. Push a branch per probe point and dispatch it.

| Probe | Ref | Expected | Purpose |
|---|---|---|---|
| A | `f8c4ce59dd0` | ~0.32 s | Confirms old code is still fast *today*, which rules out any date-based drift. |
| B | `656f87fcdb4` | ~0.76 s | Confirms the narrowed window without the matmul-registry branch confound. |
| C | `656f87fcdb4` with `tt_metal/third_party/umd` set back to `f70cc57f7f8` | ? | Isolates the UMD bump in a single run. |

If C is fast, the culprit is the UMD bump; split tt-umd#3399 from #3406. If C is slow, bisect
`f8c4ce59dd0..656f87fcdb4` (24 commits, about 5 runs). The VAE-relevant commits to test first are
`668c2907575` (concat), `668752a4056` (decorators), `656f87fcdb4` (scalar-LHS binary).

## Next steps

1. **On hardware:** build `656f87fcdb4` and `0d472904340^` (or check out `f8c4ce59dd0` with only
   the UMD submodule moved to `b2a5f606c87`) and compare
   `test_performance_wan.py -k "ring_bh_4x8_sp1tp0 and t2v and resolution_480p"`.
   This confirms or clears the UMD bump in one A/B run.
2. If the UMD bump is the cause, split it between tt-umd#3399 (Strict fallback) and #3406 /
   telemetry, and find the Metal call site that hits the slow path. Use `perf` or Tracy on the VAE
   decode and `fast_device_to_host`.
3. If the UMD bump is not the cause, bisect the remaining VAE-relevant commits in the narrowed
   window (`668c2907575`, `656f87fcdb4`, the LLK sanitizer commits).
4. Independently: add a warmup or tracing for 4x8 in `test_performance_wan.py` so VAE timing is
   less sensitive to first-run effects.

## Data sources

- Nightly runs: 34799396959 (09-14), 34921216342 (09-15), 35047979653 (09-16).
- BH sc1 T2V jobs: 103840008423, 104231133586, 104643382568.
- WH galaxy jobs: 103840008560, 104643382856. BH sc4 jobs: 103840008385, 104231133439, 104643382682.
- Mid-window branch jobs: 105012202248, 105013804062. Later branch jobs: 104847443485 (VAE 0.81 s).
