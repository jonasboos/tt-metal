# SPDX-FileCopyrightText: \u00a9 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe E for #60670: per-CPU background load on the galaxy host (/proc/stat is not namespaced)."""
import os
import subprocess
import time


def snap():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split()
            v = list(map(int, p[1:9]))
            d[p[0]] = (sum(v), v[3] + v[4])
    return d


a = snap()
time.sleep(3)
b = snap()
busy = {k: 100 * (1 - (b[k][1] - a[k][1]) / max(1, b[k][0] - a[k][0])) for k in a}
print("PROBE CPULOAD host:", os.uname().nodename, "cpus:", len(busy), "os.cpu_count:", os.cpu_count(), flush=True)
print("PROBE CPULOAD per-CPU busy > 20%:", {k: round(v) for k, v in busy.items() if v > 20}, flush=True)
print("PROBE CPULOAD mean busy: %.1f%%" % (sum(busy.values()) / len(busy)), flush=True)
for c in ("cpu1", "cpu2"):
    n = c[3:]
    print(f"PROBE CPULOAD {c} siblings:", open(f"/sys/devices/system/cpu/cpu{n}/topology/thread_siblings_list").read().strip(), flush=True)
for cmd in ("cat /proc/pressure/cpu", "ls -ld --time-style=full-iso /opt/weka /etc/wekaio /mnt/weka", "uptime"):
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout + subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stderr
    for line in out.splitlines()[:6]:
        print(f"PROBE CPULOAD [{cmd.split()[0]}] {line}", flush=True)
