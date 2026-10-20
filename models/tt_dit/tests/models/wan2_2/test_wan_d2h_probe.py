# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Probe D2 for #60670: where does the fixed ~0.4 s in the Wan VAE readback go?

Probe D1 showed the decoder forward is fine (0.245 s at 480p) and the readback stage
(float_to_uint8 + .cpu() + host reassembly) costs ~0.4 s at both 480p and 720p.
This probe splits that stage, measures D2H transfers of several sizes, measures pure host
memory work, and dumps the galaxy host's configuration (this test runs on the host itself).
Every result line starts with "PROBE".
"""

import glob
import os
import statistics
import subprocess
import time

import pytest
import torch

import ttnn
from models.tt_dit.models.vae import vae_wan2_1 as vae_mod
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.utils import tensor as tensor_mod

from ....utils.test import ring_params_8k_req_exact_devices


def _log(msg):
    print(f"PROBE {msg}", flush=True)


def _read(path, limit=400):
    try:
        with open(path) as f:
            return f.read().strip()[:limit]
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def host_env():
    cmds = [
        "hostname; uname -r; cat /proc/cmdline; nproc; uptime -s",
        "lscpu | grep -E 'Model name|Thread|Core|Socket|NUMA node|MHz' | cut -c1-120",
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor "
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq /sys/devices/system/cpu/cpufreq/boost 2>&1",
        "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n | awk '{a[NR]=$1} END{print \"cur_freq kHz min\",a[1],\"med\",a[int(NR/2)+1],\"max\",a[NR],\"n\",NR}'",
        "for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do echo $(basename $s) $(cat $s/name) lat=$(cat $s/latency) dis=$(cat $s/disable); done; cat /sys/devices/system/cpu/cpuidle/current_governor 2>&1",
        "cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; grep -E 'MemTotal|MemAvailable|HugePages_|Hugepagesize' /proc/meminfo",
        "cat /proc/self/cgroup; for f in cpu.max cpu.weight cpuset.cpus.effective memory.max memory.high memory.current memory.events memory.swap.max; do echo \"$f: $(cat /sys/fs/cgroup/$f 2>&1 | tr '\\n' ' ')\"; done",
        "grep -E '^(anon|file|kernel|shmem|pgfault|pgmajfault|thp_fault_alloc|thp_fault_fallback|workingset_refault_anon|pgscan|pgsteal) ' /sys/fs/cgroup/memory.stat 2>&1 | head -20",
        "cat /sys/module/pcie_aspm/parameters/policy 2>&1; for d in /sys/bus/pci/devices/*; do [ \"$(cat $d/vendor 2>/dev/null)\" = 0x1e52 ] || continue; echo \"$(cat $d/current_link_speed) x$(cat $d/current_link_width) iommu=$(cat $d/iommu_group/type 2>&1) numa=$(cat $d/numa_node)\"; done | sort | uniq -c",
        "cat /sys/module/tenstorrent/version; grep -H . /sys/module/tenstorrent/parameters/* 2>&1",
        "ls /dev/hugepages* 2>&1 | head -5; mount | grep -E 'huge|/dev/shm' | head -5",
        "ulimit -a | grep -E 'locked|open files|max user'",
        "ps -eo pid,pcpu,pmem,rss,comm --sort=-pcpu | head -8",
    ]
    for c in cmds:
        try:
            out = subprocess.run(["bash", "-c", c], capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            out = repr(e)
        for line in out.splitlines():
            _log(f"HOST {line}")


def mem_events():
    return _read("/sys/fs/cgroup/memory.events")


def med(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


# --------------------------------------------------------------------------- pure host work


def host_memory_bench(nbytes):
    def alloc_touch():
        t = torch.empty(nbytes, dtype=torch.uint8)
        t.fill_(1)
        return t

    m_alloc = med(alloc_touch, 3)
    src = torch.randint(0, 255, (1, 3, 81, nbytes // (3 * 81 * 832) + 1, 832), dtype=torch.uint8)
    m_perm = med(lambda: src.permute(0, 2, 3, 4, 1).contiguous(), 3)
    m_copy = med(lambda: src.clone(), 3)
    _log(
        f"HOSTMEM {nbytes / 1e6:.0f}MB alloc+touch={m_alloc[0] * 1e3:.1f}ms clone={m_copy[0] * 1e3:.1f}ms "
        f"permute_C_last={m_perm[0] * 1e3:.1f}ms torch_threads={torch.get_num_threads()}"
    )


# --------------------------------------------------------------------------- D2H microbench


def d2h_bench(mesh, per_dev_bytes, dtype):
    elem = 1 if dtype == ttnn.uint8 else 2
    n_elems = per_dev_bytes // elem
    w = 1024
    h = max(1, n_elems // w)
    torch_dtype = torch.uint8 if dtype == ttnn.uint8 else torch.bfloat16
    rows, cols = tuple(mesh.shape)
    host = torch.zeros(rows, cols, h, w, dtype=torch_dtype)
    t = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    ttnn.synchronize_device(mesh)

    def all_dev_blocking():
        t.cpu(blocking=True)

    def all_dev_nonblocking_sync():
        h_ = t.cpu(blocking=False)
        ttnn.synchronize_device(mesh)
        return h_

    def one_dev():
        ttnn.get_device_tensors(t)[0].cpu(blocking=True)

    a = med(all_dev_blocking)
    b = med(all_dev_nonblocking_sync)
    c = med(one_dev)
    _log(
        f"D2H per_dev={per_dev_bytes / 1e6:.3f}MB dtype={dtype} total={per_dev_bytes * rows * cols / 1e6:.0f}MB: "
        f"all32_blocking med={a[0] * 1e3:.1f}ms (min {a[1] * 1e3:.1f} max {a[2] * 1e3:.1f}) | "
        f"all32_nonblocking+sync med={b[0] * 1e3:.1f}ms | one_device med={c[0] * 1e3:.1f}ms"
    )
    ttnn.deallocate(t)


# --------------------------------------------------------------------------- VAE readback split


def readback_split(adapter, latents, tag):
    mesh = adapter.device
    pc = adapter._parallel_config
    x = latents.to(adapter._torch_vae.dtype) * adapter._latents_std + adapter._latents_mean
    tt_x, lh, lw = adapter._decoder.prepare_input(x)
    tt_x = vae_mod.typed_tensor_2dshard(
        tt_x,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        dtype=adapter._decoder.dtype,
    )
    video, nh, nw = adapter._decoder(tt_x, lh, t_chunk_size=adapter._t_chunk_size, logical_w=lw)
    ttnn.synchronize_device(mesh)

    ev0 = mem_events()
    t0 = time.perf_counter()
    u8 = vae_mod.float_to_uint8(video)
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    mesh_coords = list(u8.tensor_topology().mesh_coords())
    host_tensor = u8.cpu(blocking=False)
    t2 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    t3 = time.perf_counter()
    shards_tt = ttnn.get_device_tensors(host_tensor)
    logical_shape = list(shards_tt[0].shape)
    trim = tuple(slice(0, d) for d in logical_shape)
    shards = [tensor_mod._to_torch_zero_copy(s)[trim] for s in shards_tt]
    t4 = time.perf_counter()
    concat_dims = [None, None]
    concat_dims[pc.height_parallel.mesh_axis] = 3
    concat_dims[pc.width_parallel.mesh_axis] = 4
    out = tensor_mod._reassemble_2d(
        mesh_coords, shards, logical_shape, tuple(mesh.shape), concat_dims, (0, 2, 3, 4, 1), None
    )
    t5 = time.perf_counter()
    ev1 = mem_events()
    _log(
        f"READBACK[{tag}] to_uint8={1e3 * (t1 - t0):.1f}ms cpu_call={1e3 * (t2 - t1):.1f}ms sync={1e3 * (t3 - t2):.1f}ms "
        f"to_torch={1e3 * (t4 - t3):.1f}ms reassemble={1e3 * (t5 - t4):.1f}ms total={1e3 * (t5 - t0):.1f}ms "
        f"shard={tuple(logical_shape)} dtype={u8.dtype} out={tuple(out.shape)}"
    )
    if ev0 != ev1:
        _log(f"READBACK[{tag}] memory.events changed: {ev0!r} -> {ev1!r}")

    # same tensor, blocking read
    t0 = time.perf_counter()
    u8.cpu(blocking=True)
    t1 = time.perf_counter()
    _log(f"READBACK[{tag}] cpu(blocking=True) of the uint8 video = {1e3 * (t1 - t0):.1f}ms")
    # bf16 video (no to_uint8) for comparison
    t0 = time.perf_counter()
    video.cpu(blocking=True)
    t1 = time.perf_counter()
    _log(f"READBACK[{tag}] cpu(blocking=True) of the bf16 video = {1e3 * (t1 - t0):.1f}ms shape={tuple(video.shape)}")
    for tt in (u8, video):
        ttnn.deallocate(tt)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), ring_params_8k_req_exact_devices]],
    ids=["ring_bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_wan_d2h_probe(mesh_device):
    mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 8))
    host_env()

    for nbytes in (100_000_000, 230_000_000):
        host_memory_bench(nbytes)

    for per_dev in (65_536, 1_048_576, 3_145_728, 8_388_608, 16_777_216):
        d2h_bench(mesh, per_dev, ttnn.uint8)
    for per_dev in (1_048_576, 8_388_608):
        d2h_bench(mesh, per_dev, ttnn.bfloat16)

    for width, height in ((832, 480), (1280, 720)):
        res = f"{height}p"
        cfg = WanPipelineConfig.default(mesh_shape=mesh.shape, height=height, width=width, num_frames=81)
        ccl = CCLManager(mesh_device=mesh, num_links=cfg.num_links, topology=ttnn.Topology.Linear)
        adapter = vae_mod.WanVAEDecoderAdapter(
            checkpoint_name=cfg.checkpoint_name,
            parallel_config=cfg.vae_parallel_config,
            ccl_manager=ccl,
            height=height,
            width=width,
            num_frames=81,
            vae_t_chunk_size=cfg.vae_t_chunk_size,
            vae_dtype=cfg.vae_dtype,
            sdpa_t_fracture_w_only=cfg.sdpa_t_fracture_w_only,
        )
        adapter.reload_weights()
        torch.manual_seed(0)
        latents = torch.randn(1, adapter.config.z_dim, 21, height // 8, width // 8)
        adapter.decode(latents, output_type="uint8")
        adapter.decode(latents, output_type="uint8")
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            adapter.decode(latents, output_type="uint8")
            ts.append(time.perf_counter() - t0)
        _log(f"DECODE[{res}] plain = {[round(t, 4) for t in ts]}")
        for i in range(3):
            readback_split(adapter, latents, f"{res}-{i}")
        adapter.deallocate_weights()
        del adapter, ccl
        ttnn.synchronize_device(mesh)

    for nbytes in (100_000_000, 230_000_000):
        host_memory_bench(nbytes)
    _log(f"memory.events at end: {mem_events()!r}")
