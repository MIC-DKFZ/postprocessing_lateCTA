# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Process and system resource checks (memory, child processes, open files), read
from /proc so that leaks across cases can be spotted and OOM kills avoided
"""

import os

import numpy as np

from utils.stage_timer import StageTimer


def count_children() -> int:
    """
    Number of live child processes of this process

    nnU-Net spawns a preprocessing worker pool per data iterator. If those workers
    are not reaped between cases they accumulate, and each one holds its own copy of
    the volume it was handed, so host memory grows case by case.
    """
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/task")) and len(
            [
                p
                for p in os.listdir("/proc")
                if p.isdigit() and _ppid_of(int(p)) == os.getpid()
            ]
        )
    except Exception:
        return -1


def _ppid_of(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            # field 4 is ppid; the comm field may contain spaces so split after ')'
            return int(f.read().rpartition(")")[2].split()[1])
    except Exception:
        return -1


def count_open_fds() -> int:
    """Number of open file descriptors, another thing that leaks across iterations"""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


def system_memory_gb() -> tuple:
    """Return (total, available) system memory in GiB, or (nan, nan)"""
    try:
        info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 2**20  # kB -> GiB
        return info.get("MemTotal", float("nan")), info.get(
            "MemAvailable", float("nan")
        )
    except Exception:
        return float("nan"), float("nan")


def check_memory_headroom(limit_gb: float, cid: str = "-") -> None:
    """
    Abort with a clear error if this process is close to a memory ceiling

    An OOM kill arrives as SIGKILL: no traceback, no cleanup, nothing written. This
    raises a normal MemoryError slightly before that point, so the failure is
    recorded and (with --keep_going) the run continues with the next case.
    """
    if limit_gb is None or limit_gb <= 0:
        return
    rss = StageTimer.rss_gb()
    if np.isfinite(rss) and rss > limit_gb:
        raise MemoryError(
            f"RSS {rss:.2f} GB exceeded the --mem_limit_gb ceiling of {limit_gb:.2f} GB "
            f"while processing '{cid}'. Lower --np, or process this case separately"
        )
