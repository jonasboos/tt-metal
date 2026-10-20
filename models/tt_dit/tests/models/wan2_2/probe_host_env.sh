#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Host / board state dump for the #60670 probe. Read-only; every line starts with "HOSTENV".
# Run before the mesh is opened.

p() { sed 's/^/HOSTENV /'; }
r() { echo "--- $*" | p; ( eval "$@" ) 2>&1 | head -${HOSTENV_MAXLINES:-60} | p; }

r 'hostname; date -u; uptime; uptime -s; cat /proc/loadavg'
r 'uname -a; cat /proc/cmdline'
r 'nproc; lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|MHz|Flags" | cut -c1-200'
r 'cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.weight'
r 'cat /sys/fs/cgroup/cpu.stat'
r 'for f in scaling_driver scaling_governor energy_performance_preference scaling_min_freq scaling_max_freq cpuinfo_max_freq; do echo "$f: $(cat /sys/devices/system/cpu/cpu0/cpufreq/$f)"; done'
r 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c'
r 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort -n | awk "{a[NR]=\$1} END{print \"cur_freq kHz min\",a[1],\"median\",a[int(NR/2)+1],\"max\",a[NR],\"n\",NR}"'
r 'grep -H . /sys/devices/system/cpu/amd_pstate/status /sys/devices/system/cpu/intel_pstate/status /sys/devices/system/cpu/intel_pstate/no_turbo /sys/devices/system/cpu/cpufreq/boost /sys/module/intel_idle/parameters/max_cstate /sys/module/processor/parameters/max_cstate'
r 'for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do echo "$(basename $s) $(cat $s/name) latency=$(cat $s/latency)us disable=$(cat $s/disable) usage=$(cat $s/usage)"; done'
r 'grep -iE "MemTotal|MemAvailable|Huge" /proc/meminfo; cat /sys/kernel/mm/transparent_hugepage/enabled'
r 'cat /sys/module/pcie_aspm/parameters/policy'
r 'for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor 2>/dev/null)" = "0x1e52" ] || continue; echo "speed=$(cat $d/current_link_speed) width=x$(cat $d/current_link_width) max=$(cat $d/max_link_speed)/x$(cat $d/max_link_width) numa=$(cat $d/numa_node) iommu=$(cat $d/iommu_group/type 2>/dev/null) l1_aspm=$(cat $d/link/l1_aspm 2>/dev/null) l0s=$(cat $d/link/l0s_aspm 2>/dev/null)"; done | sort | uniq -c'
r 'for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor 2>/dev/null)" = "0x1e52" ] || continue; echo "$(basename $d) $(cat $d/current_link_speed) x$(cat $d/current_link_width) numa=$(cat $d/numa_node)"; done'
r 'ls /sys/class/iommu 2>/dev/null; for g in $(ls /sys/kernel/iommu_groups 2>/dev/null | head -3); do cat /sys/kernel/iommu_groups/$g/type; done'
r 'cat /sys/module/tenstorrent/version; grep -H . /sys/module/tenstorrent/parameters/* 2>/dev/null'
r 'ls /dev/tenstorrent | wc -l; ls /dev/hugepages* 2>/dev/null | head'
r 'ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -12'
r 'env | grep -E "^TT_|^ARCH|^OMP|^MKL|^TORCH" | sort'
r 'python3 -c "import torch; print(\"torch\", torch.__version__, \"threads\", torch.get_num_threads())"'

echo "--- tt-smi snapshot (before mesh open)" | p
if command -v tt-smi >/dev/null 2>&1; then
    timeout 90 tt-smi -s --snapshot_no_tty > /tmp/probe_smi_before.json 2>/tmp/probe_smi_before.err
    echo "rc=$? bytes=$(wc -c < /tmp/probe_smi_before.json)" | p
    python3 - <<'EOF' 2>&1 | p
import collections, json
d = json.load(open("/tmp/probe_smi_before.json"))
devs = d.get("device_info") or []
print("devices", len(devs), "top keys", sorted(d))
def flat(prefix, o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            flat(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = o
per = collections.defaultdict(list)
for dv in devs:
    f = {}
    flat("", dv, f)
    for k, v in f.items():
        per[k].append(str(v))
for k in sorted(per):
    c = collections.Counter(per[k])
    print(("all " if len(c) == 1 else "var ") + k, "=", per[k][0] if len(c) == 1 else dict(c))
EOF
else
    echo "tt-smi not found" | p
fi

echo "--- ethernet links / DRAM harvesting (UMD only)" | p
for mode in "" "--map" "--dram"; do
    echo "check_links $mode" | p
    timeout 120 python3 -I "$(dirname "$0")/probe_check_links.py" $mode 2>&1 | tail -40 | p
done
