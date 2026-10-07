# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Per-stage wall-clock and memory timing, streamed to a CSV file
"""

import csv
import os
import resource
import shutil
import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import torch


# Column layout of the timings CSV
FIELDS = [
    "run_id",
    "ts",
    "cid",
    "stage",
    "seconds",
    "rss_gb",
    "peak_rss_gb",
    "status",
    "error",
]


def fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.mmm"""
    h, rem = divmod(float(seconds), 3600.0)
    m, s = divmod(rem, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


class StageTimer:
    """
    Collect wall-clock timings and resident memory per processing stage, per case

    Rows are appended to the CSV and flushed to disk as soon as each stage ends, so
    the record survives a hard kill (OOM killer, SIGKILL, node failure) where no
    traceback or atexit handler would ever run.


    Usage
    -----
    timer = StageTimer()
    timer.attach_csv("out/timings.csv")
    with timer("inference", cid):
        ...
    timer.summary()

    """

    def __init__(self, sync_cuda: bool = True):
        self.records = []  # list of dicts keyed by FIELDS
        self.sync_cuda = sync_cuda
        self._fh = None
        self._writer = None
        # Identifies this invocation. When several short-lived runs append to the
        # same CSV (e.g. with --max_cases), this is what separates them
        self.run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"

    def attach_csv(self, path: os.PathLike, resume: bool = True) -> None:
        """
        Stream every recorded row to 'path' as it happens

        With resume=True an existing file is appended to rather than truncated, so a
        series of short-lived runs (--max_cases) accumulates into a single timeline
        and the evidence from a run that crashed is preserved.
        """
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if (exists and resume) else "w"

        if mode == "a":
            # Appending to a file written by an older version would silently misalign
            # every column, so verify the header matches before trusting it
            try:
                with open(path, "r", newline="") as f:
                    header = next(csv.reader(f), [])
            except Exception:
                header = []

            if header != FIELDS:
                backup = f"{path}.{datetime.now().strftime('%Y%m%dT%H%M%S')}.bak"
                shutil.move(path, backup)
                print(
                    f"NOTE: existing timings have a different column layout and were "
                    f"moved to '{os.path.basename(backup)}'; starting a new file",
                    flush=True,
                )
                exists, mode = False, "w"
            else:
                # A run killed mid-write can leave a partial final line. Append a
                # newline if needed so the next row does not fuse onto it
                try:
                    with open(path, "rb") as f:
                        f.seek(-1, os.SEEK_END)
                        needs_nl = f.read(1) not in (b"\n", b"\r")
                    if needs_nl:
                        with open(path, "a", newline="") as f:
                            f.write("\n")
                        print(
                            "NOTE: repaired a truncated final line in the timings file",
                            flush=True,
                        )
                except Exception:
                    pass

        # line buffered; every write is followed by an explicit flush + fsync
        self._fh = open(path, mode, newline="", buffering=1)
        self._writer = csv.DictWriter(
            self._fh, fieldnames=FIELDS, extrasaction="ignore"
        )
        if mode == "w" or not exists:
            self._writer.writeheader()
            self._fh.flush()

        # Emit anything recorded before the CSV was attached
        for rec in self.records:
            self._write_row(rec)

    def _write_row(self, rec: dict) -> None:
        if self._writer is None:
            return
        try:
            self._writer.writerow(rec)
            self._fh.flush()
            os.fsync(self._fh.fileno())  # push past the OS page cache
        except Exception as exc:  # never let logging kill the run
            print(f"WARNING: could not write timing row: {exc}", flush=True)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
            except Exception:
                pass
            self._fh, self._writer = None, None

    @staticmethod
    def rss_gb() -> float:
        """Current resident set size of this process, in GiB"""
        try:
            with open("/proc/self/statm", "r") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 2**30
        except Exception:
            return float("nan")

    @staticmethod
    def peak_rss_gb() -> float:
        """Peak RSS of this process since it started, in GiB (monotonic)"""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        except Exception:
            return float("nan")

    def _sync(self):
        # CUDA kernels are launched asynchronously: without a sync, a GPU stage
        # would report only the time spent queueing the work
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def __call__(self, stage: str, cid: str = "-"):
        self._sync()
        t0 = time.perf_counter()
        status, err = "ok", ""
        try:
            yield
        except BaseException as exc:
            # Record the partial timing before the exception propagates, so a stage
            # that dies still leaves a row behind
            status, err = "error", f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            try:
                self._sync()
            except Exception:
                pass
            self.record(
                stage=stage,
                cid=cid,
                seconds=time.perf_counter() - t0,
                status=status,
                error=err,
            )

    def record(
        self,
        stage: str,
        cid: str,
        seconds: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        """Add a timing and immediately persist it"""
        rec = {
            "run_id": self.run_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cid": cid,
            "stage": stage,
            "seconds": seconds,
            "rss_gb": self.rss_gb(),
            "peak_rss_gb": self.peak_rss_gb(),
            "status": status,
            "error": error,
        }
        self.records.append(rec)
        self._write_row(rec)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=FIELDS)

    def summary(self, total_seconds: float = None, n_cases: int = None) -> pd.DataFrame:
        """Aggregate per stage and print a table. Returns the aggregated frame."""
        df = self.to_frame()
        if df.empty:
            print("No timings recorded")
            return df

        agg = (
            df.groupby("stage")["seconds"]
            .agg(calls="count", total="sum", mean="mean", median="median", max="max")
            .sort_values("total", ascending=False)
        )
        agg["percent"] = 100.0 * agg["total"] / agg["total"].sum()

        print("\n" + "=" * 72)
        print("EXECUTION TIME BREAKDOWN")
        print("=" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:10.2f}"):
            print(agg.to_string())
        print("-" * 72)
        if total_seconds is not None:
            print(f"{'Total wall clock':<28}{fmt_hms(total_seconds)}")
            if n_cases:
                print(
                    f"{'Per case':<28}{fmt_hms(total_seconds / n_cases)}"
                    f"  ({n_cases} cases)"
                )
        print("=" * 72 + "\n", flush=True)
        return agg
