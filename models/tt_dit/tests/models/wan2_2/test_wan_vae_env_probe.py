# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Diagnostic probe for #60670 (Wan VAE decode ~2x slower on BH sc1 since 2026-09-15).

Runs the Wan VAE decoder on its own (no transformer, no denoise) and reports:
  1. plain decode wall time (the same thing the perf test's ``vae`` metric measures),
  2. a stage split: host prep + H2D, decoder forward, float_to_uint8 + D2H + host reassembly,
  3. a per-op breakdown where every ttnn op is bracketed by synchronize_device,
  4. decode time with neighbor_pad forced to 1 link,
  5. tt-smi snapshots (clocks, temperatures, power limits) before/after load,
  6. decode time right after a heat soak.
Every result line starts with "PROBE" so it can be grepped from the job log.
"""

import collections
import json
import shutil
import statistics
import subprocess
import time

import pytest
import torch

import ttnn
from models.tt_dit.models.vae import vae_wan2_1 as vae_mod
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig

from ....utils.test import ring_params_8k_req_exact_devices


def _log(msg):
    print(f"PROBE {msg}", flush=True)


# --------------------------------------------------------------------------- tt-smi snapshots


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)[:120]
    else:
        out[prefix] = obj


def tt_smi_snapshot(tag, dump_first_device=False):
    exe = shutil.which("tt-smi")
    if exe is None:
        _log(f"SMI[{tag}] tt-smi not found")
        return
    try:
        res = subprocess.run([exe, "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        _log(f"SMI[{tag}] failed: {e!r}")
        return
    if res.returncode != 0:
        _log(f"SMI[{tag}] rc={res.returncode} stderr={res.stderr[-500:]!r}")
        return
    try:
        data = json.loads(res.stdout)
    except Exception as e:  # noqa: BLE001
        _log(f"SMI[{tag}] bad json: {e!r} head={res.stdout[:300]!r}")
        return
    devices = data.get("device_info") or []
    _log(f"SMI[{tag}] devices={len(devices)} top_keys={sorted(data.keys())}")
    if dump_first_device and devices:
        flat = {}
        _flatten("", devices[0], flat)
        for k in sorted(flat):
            _log(f"SMI[{tag}] dev0 {k} = {flat[k]}")
    per_key = collections.defaultdict(list)
    for dev in devices:
        flat = {}
        _flatten("", dev, flat)
        for k, v in flat.items():
            per_key[k].append(v)
    for k in sorted(per_key):
        vals = per_key[k]
        uniq = collections.Counter(str(v) for v in vals)
        if len(uniq) == 1:
            _log(f"SMI[{tag}] all {k} = {vals[0]}")
        else:
            _log(f"SMI[{tag}] var {k} = {[str(v) for v in vals]}")


# --------------------------------------------------------------------------- host CPU power state


def cpuidle_totals():
    """Sum over CPUs of (usage, time_us) per idle state name, from sysfs (readable unprivileged)."""
    import glob
    import os

    tot = collections.defaultdict(lambda: [0, 0])
    for sdir in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*"):
        try:
            name = open(os.path.join(sdir, "name")).read().strip()
            tot[name][0] += int(open(os.path.join(sdir, "usage")).read())
            tot[name][1] += int(open(os.path.join(sdir, "time")).read())
        except Exception:  # noqa: BLE001
            pass
    return dict(tot)


def cpuidle_delta(before, after):
    return {k: (after[k][0] - before.get(k, [0, 0])[0], after[k][1] - before.get(k, [0, 0])[1]) for k in after}


class IdleSpinners:
    """One SCHED_IDLE busy-loop process per CPU: keeps cores out of C-states without competing with real work."""

    CODE = "import os\ntry:\n    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))\nexcept Exception as e:\n    print('PROBE SPIN sched_idle failed', e, flush=True)\nwhile True:\n    pass\n"

    def __enter__(self):
        import os
        import sys

        cpus = sorted(os.sched_getaffinity(0))
        self.procs = []
        for c in cpus:
            proc = subprocess.Popen([sys.executable, "-c", self.CODE])
            try:
                os.sched_setaffinity(proc.pid, {c})
            except Exception:  # noqa: BLE001
                pass
            self.procs.append(proc)
        time.sleep(2.0)
        _log(f"SPIN started {len(self.procs)} SCHED_IDLE spinners")
        return self

    def __exit__(self, *exc):
        for proc in self.procs:
            proc.kill()
        for proc in self.procs:
            proc.wait()
        _log("SPIN stopped")
        return False


# --------------------------------------------------------------------------- per-op timing


class OpTimer:
    """Wraps every ttnn op so each outermost call is bracketed by synchronize_device."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.enabled = False
        self.depth = 0
        self.stats = collections.defaultdict(lambda: [0, 0.0, 0.0])  # calls, total, max
        self.calls = []
        self._patched = []

    @staticmethod
    def _detail(name, args, kwargs):
        try:
            if name.endswith("neighbor_pad_async"):
                axes = args[5] if len(args) > 5 else kwargs.get("cluster_axes")
                return f"[axes={list(axes)},links={kwargs.get('num_links')}]"
            if name.endswith("all_gather_async"):
                return f"[dim={kwargs.get('dim', args[1] if len(args) > 1 else '?')},axis={kwargs.get('cluster_axis')}]"
        except Exception:  # noqa: BLE001
            pass
        return ""

    @staticmethod
    def _shape(args):
        for a in args:
            if isinstance(a, ttnn.Tensor):
                try:
                    return tuple(a.shape)
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _wrap(self, name, fn):
        timer = self

        def wrapper(*args, **kwargs):
            if not timer.enabled or timer.depth > 0:
                return fn(*args, **kwargs)
            timer.depth += 1
            try:
                ttnn.synchronize_device(timer.mesh)
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                ttnn.synchronize_device(timer.mesh)
                dt = time.perf_counter() - t0
            finally:
                timer.depth -= 1
            key = name + timer._detail(name, args, kwargs)
            s = timer.stats[key]
            s[0] += 1
            s[1] += dt
            s[2] = max(s[2], dt)
            timer.calls.append((dt, key, timer._shape(args)))
            return out

        return wrapper

    def install(self):
        skip = {"synchronize_device", "to_torch", "get_device_tensors"}
        for ns, prefix in ((ttnn, "ttnn"), (ttnn.experimental, "ttnn.experimental"), (ttnn.transformer, "ttnn.transformer")):
            for attr in list(vars(ns)):
                if attr in skip or attr.startswith("_"):
                    continue
                try:
                    obj = getattr(ns, attr, None)
                except Exception:  # noqa: BLE001
                    continue
                if obj is None or not hasattr(obj, "python_fully_qualified_name") or not callable(obj):
                    continue
                self._patched.append((ns, attr, obj))
                setattr(ns, attr, self._wrap(f"{prefix}.{attr}", obj))
        _log(f"OPTIMER installed wrappers on {len(self._patched)} ops")

    def uninstall(self):
        for ns, attr, obj in self._patched:
            setattr(ns, attr, obj)
        self._patched = []

    def report(self, tag, sync_s):
        total = sum(s[1] for s in self.stats.values())
        ncalls = sum(s[0] for s in self.stats.values())
        _log(
            f"OPS[{tag}] total_synced={total:.4f}s calls={ncalls} idle_sync={sync_s * 1e6:.1f}us "
            f"(approx sync overhead {ncalls * sync_s:.4f}s)"
        )
        for key, (n, tot, mx) in sorted(self.stats.items(), key=lambda kv: -kv[1][1]):
            _log(
                f"OPS[{tag}] {tot:8.4f}s {100 * tot / max(total, 1e-9):5.1f}% n={n:4d} "
                f"mean={1e6 * tot / n:9.1f}us max={1e6 * mx:9.1f}us {key}"
            )
        for dt, key, shape in sorted(self.calls, key=lambda c: -c[0])[:25]:
            _log(f"OPS[{tag}] slowest {1e6 * dt:9.1f}us {key} shape={shape}")
        self.stats.clear()
        self.calls.clear()


# --------------------------------------------------------------------------- helpers


def idle_sync_seconds(mesh, n=200):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n


def small_op_bench(mesh):
    x = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )
    for _ in range(20):
        ttnn.deallocate(ttnn.add(x, 1.0))
    ttnn.synchronize_device(mesh)
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        y = ttnn.add(x, 1.0)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(y)
    rt = (time.perf_counter() - t0) / n
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(ttnn.add(x, 1.0))
    ttnn.synchronize_device(mesh)
    thr = (time.perf_counter() - t0) / n
    _log(f"MICRO small add+sync round trip = {rt * 1e6:.1f}us; async small add throughput = {thr * 1e6:.1f}us/op")
    ttnn.deallocate(x)


def heat_soak(mesh, seconds):
    a = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )
    t_end = time.perf_counter() + seconds
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        for _ in range(20):
            ttnn.deallocate(ttnn.matmul(a, a))
        ttnn.synchronize_device(mesh)
        iters += 20
    el = time.perf_counter() - t0
    _log(f"SOAK {iters} matmuls 4096^3 in {el:.1f}s -> {el / iters * 1e3:.2f} ms/matmul")
    ttnn.deallocate(a)


def decode_staged(adapter, latents):
    """Same steps as WanVAEDecoderAdapter.decode(output_type='uint8'), with a timer per stage."""
    mesh = adapter.device
    pc = adapter._parallel_config
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    x = latents.to(adapter._torch_vae.dtype)
    x = x * adapter._latents_std + adapter._latents_mean
    tt_x, logical_h, logical_w = adapter._decoder.prepare_input(x)
    tt_x = vae_mod.typed_tensor_2dshard(
        tt_x,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        dtype=adapter._decoder.dtype,
    )
    adapter.reload_weights()
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    video, new_h, new_w = adapter._decoder(tt_x, logical_h, t_chunk_size=adapter._t_chunk_size, logical_w=logical_w)
    ttnn.synchronize_device(mesh)
    t2 = time.perf_counter()
    concat_dims = [None, None]
    concat_dims[pc.height_parallel.mesh_axis] = 3
    concat_dims[pc.width_parallel.mesh_axis] = 4
    out = vae_mod.fast_device_to_host(
        video,
        mesh,
        concat_dims,
        ccl_manager=adapter._ccl_manager,
        pre_transfer_fn=vae_mod.float_to_uint8,
        permute=(0, 2, 3, 4, 1),
    )
    out = out[:, :, :new_h, :new_w, :]
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0, tuple(out.shape)


def plain_decodes(adapter, latents, n, tag):
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        out = adapter.decode(latents, output_type="uint8")
        times.append(time.perf_counter() - t0)
    _log(
        f"DECODE[{tag}] n={n} mean={statistics.mean(times):.4f}s min={min(times):.4f}s "
        f"max={max(times):.4f}s all={[round(t, 4) for t in times]} out={tuple(out.shape)}"
    )
    return times


# --------------------------------------------------------------------------- test


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), ring_params_8k_req_exact_devices]],
    ids=["ring_bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(832, 480), (1280, 720)], ids=["480p", "720p"])
def test_wan_vae_env_probe(mesh_device, width, height):
    res = f"{height}p"
    mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 8))
    num_frames = 81

    tt_smi_snapshot(f"{res}-start", dump_first_device=(res == "480p"))
    sync_s = idle_sync_seconds(mesh)
    _log(f"MICRO idle synchronize_device = {sync_s * 1e6:.1f}us")
    small_op_bench(mesh)

    cfg = WanPipelineConfig.default(mesh_shape=mesh.shape, height=height, width=width, num_frames=num_frames)
    _log(
        f"CONFIG {res}: num_links={cfg.num_links} vae_t_chunk_size={cfg.vae_t_chunk_size} "
        f"vae_dtype={cfg.vae_dtype} vae_parallel={cfg.vae_parallel_config}"
    )
    ccl = CCLManager(mesh_device=mesh, num_links=cfg.num_links, topology=ttnn.Topology.Linear)
    adapter = vae_mod.WanVAEDecoderAdapter(
        checkpoint_name=cfg.checkpoint_name,
        parallel_config=cfg.vae_parallel_config,
        ccl_manager=ccl,
        height=height,
        width=width,
        num_frames=num_frames,
        vae_t_chunk_size=cfg.vae_t_chunk_size,
        vae_dtype=cfg.vae_dtype,
        sdpa_t_fracture_w_only=cfg.sdpa_t_fracture_w_only,
    )
    t0 = time.perf_counter()
    adapter.reload_weights()
    _log(f"LOAD {res} weights in {time.perf_counter() - t0:.1f}s")

    torch.manual_seed(0)
    latent_t = (num_frames - 1) // 4 + 1
    latents = torch.randn(1, adapter.config.z_dim, latent_t, height // 8, width // 8)

    t0 = time.perf_counter()
    adapter.decode(latents, output_type="uint8")
    _log(f"DECODE[{res}-first] {time.perf_counter() - t0:.2f}s (includes compile on a cold host)")
    plain_decodes(adapter, latents, 2, f"{res}-warm2")

    # 1. plain decode, as the perf test measures it (+ C-state entries during it)
    ci0 = cpuidle_totals()
    plain_decodes(adapter, latents, 5, f"{res}-plain")
    _log(f"CPUIDLE[{res}-plain x5] (entries, us) per state = {cpuidle_delta(ci0, cpuidle_totals())}")

    # 1b. same, with every core kept busy by SCHED_IDLE spinners (no C-states, high clocks)
    with IdleSpinners():
        ci0 = cpuidle_totals()
        plain_decodes(adapter, latents, 5, f"{res}-spinners")
        _log(f"CPUIDLE[{res}-spinners x5] (entries, us) per state = {cpuidle_delta(ci0, cpuidle_totals())}")
        small_op_bench(mesh)
    plain_decodes(adapter, latents, 3, f"{res}-plain-again")

    # 2. stage split
    for i in range(3):
        prep, fwd, d2h, tot, shape = decode_staged(adapter, latents)
        _log(
            f"STAGES[{res}] iter{i} prep+h2d={prep:.4f}s decoder_fwd={fwd:.4f}s "
            f"to_uint8+d2h+host={d2h:.4f}s total={tot:.4f}s out={shape}"
        )

    # 3. per-op breakdown (each op bracketed by synchronize_device)
    timer = OpTimer(mesh)
    timer.install()
    try:
        timer.enabled = True
        decode_staged(adapter, latents)
        timer.enabled = False
        timer.report(f"{res}-warmup", sync_s)
        timer.enabled = True
        prep, fwd, d2h, tot, _ = decode_staged(adapter, latents)
        timer.enabled = False
        _log(f"STAGES[{res}-instrumented] prep+h2d={prep:.4f}s decoder_fwd={fwd:.4f}s to_uint8+d2h+host={d2h:.4f}s")
        timer.report(f"{res}", sync_s)
    finally:
        timer.enabled = False
        timer.uninstall()

    # 4. neighbor_pad forced to 1 link
    orig = vae_mod.get_neighbor_pad_num_links
    vae_mod.get_neighbor_pad_num_links = lambda ccl_manager, input_tensor, dim: 1
    try:
        plain_decodes(adapter, latents, 1, f"{res}-np1link-warm")
        plain_decodes(adapter, latents, 3, f"{res}-np1link")
    finally:
        vae_mod.get_neighbor_pad_num_links = orig
    plain_decodes(adapter, latents, 2, f"{res}-np-default-again")

    # 5./6. telemetry around load, decode right after a heat soak
    tt_smi_snapshot(f"{res}-after-decodes")
    heat_soak(mesh, 45)
    tt_smi_snapshot(f"{res}-after-soak")
    plain_decodes(adapter, latents, 2, f"{res}-after-soak")
    time.sleep(20)
    plain_decodes(adapter, latents, 2, f"{res}-after-20s-idle")
