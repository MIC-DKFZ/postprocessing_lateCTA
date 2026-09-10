# SPDX-FileCopyrightText: Copyright 2026 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

"""
Summarize the per-case timings written by postprocess_end2end.py

Reads the long-format timings.csv (cid, stage, seconds), pivots it to one row per
case, groups the raw stages into the three pipeline components, and reports
mean (95% CI) across cases for each.

Usage
-----
python summarize_timings.py timings.csv
python summarize_timings.py timings.csv --out-prefix run_late --boot 10000
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# Raw stage -> pipeline component. Stages not listed here are reported on their own
# but excluded from the component totals (e.g. predictor_init, which is a one-off)
COMPONENTS = {
    "totalsegmentator": "brain_segmentation",
    "totalsegmentator_cached": "brain_segmentation",
    "read_image": "io",
    "read_brain_mask": "io",
    "prepare_input": "generator",
    "inference": "generator",
    "postprocess_preds": "postprocessing",
    "process_case": "postprocessing",
}

# Recorded once per run rather than once per case
RUN_LEVEL_STAGES = ["predictor_init"]

# Rows that mark a failure rather than a measured pipeline stage
FAILURE_STAGES = ["case_failed"]


def mean_ci(x: np.ndarray, alpha: float = 0.05, n_boot: int = 0) -> dict:
    """
    Mean with a 95% confidence interval

    Uses the Student-t interval by default. With n_boot > 0 a BCa-free percentile
    bootstrap is added, which does not assume normality and is the safer choice
    for the small, right-skewed samples typical of runtime measurements.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    out = {
        "n": n,
        "mean": np.nan,
        "sd": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "median": np.nan,
        "min": np.nan,
        "max": np.nan,
        "total": np.nan,
    }
    if n == 0:
        return out

    out.update(
        mean=x.mean(),
        sd=x.std(ddof=1) if n > 1 else np.nan,
        median=np.median(x),
        min=x.min(),
        max=x.max(),
        total=x.sum(),
    )

    if n > 1:
        # Student-t interval: half-width = t_{1-alpha/2, n-1} * s / sqrt(n)
        sem = stats.sem(x)
        half = sem * stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        out["ci_low"], out["ci_high"] = out["mean"] - half, out["mean"] + half

        if n_boot > 0:
            rng = np.random.default_rng(0)
            means = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
            lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out["boot_ci_low"], out["boot_ci_high"] = lo, hi

    return out


def fmt_ci(row: pd.Series, decimals: int = 2) -> str:
    """Format a stats row as 'mean (low-high)'"""
    if not np.isfinite(row.get("ci_low", np.nan)):
        return f"{row['mean']:.{decimals}f} (CI undefined, n={int(row['n'])})"
    return (
        f"{row['mean']:.{decimals}f} "
        f"({row['ci_low']:.{decimals}f}-{row['ci_high']:.{decimals}f})"
    )


def main(args):
    # A run killed mid-write can leave a truncated final line; skip it rather than
    # refusing to parse the whole file
    df = pd.read_csv(args.csv, on_bad_lines="skip")
    expected = {"cid", "stage", "seconds"}
    assert expected.issubset(df.columns), f"CSV must contain columns {expected}"

    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df = df.dropna(subset=["seconds"])

    # Report and exclude failed stages: their timings are partial by definition
    if "status" in df.columns:
        failed = df[df["status"].astype(str).eq("error")]
        if len(failed) > 0:
            print(f"{len(failed)} failed stage row(s) found, excluded from statistics:")
            for _, r in failed.iterrows():
                print(f"  {r['cid']:<16}{r['stage']:<22}{str(r.get('error',''))[:70]}")
            print()
            df = df[~df["status"].astype(str).eq("error")]

    df = df[~df["stage"].isin(FAILURE_STAGES)]

    # A CSV accumulated across several invocations (--max_cases, or a rerun after a
    # crash) can hold more than one row for the same (cid, stage): a case that was
    # interrupted and later redone. Keep the LAST attempt, which is the one that
    # actually produced the output, otherwise a retried case is counted twice and
    # its partial first attempt drags the mean down
    n_runs = df["run_id"].nunique() if "run_id" in df.columns else 1
    subset = ["cid", "stage"]
    n_dupes = int(df.duplicated(subset=subset, keep="last").sum())
    if n_dupes > 0:
        print(
            f"{n_dupes} duplicate (case, stage) row(s) across {n_runs} run(s); "
            "keeping the most recent attempt of each\n"
        )
        df = df.drop_duplicates(subset=subset, keep="last")
    elif n_runs > 1:
        print(f"Timings accumulated across {n_runs} run(s)\n")

    # Keep only cases that completed every stage they started; a run cut short by an
    # OOM kill leaves its last case with a partial set of rows that would bias means
    if args.complete_only:
        counts = (
            df[~df["stage"].isin(RUN_LEVEL_STAGES)].groupby("cid")["stage"].nunique()
        )
        full = counts.max() if len(counts) else 0
        partial = counts[counts < full].index.tolist()
        if partial:
            print(f"Dropping {len(partial)} incomplete case(s): {partial}\n")
            df = df[~df["cid"].isin(partial)]

    # Separate run-level stages from per-case ones
    run_level = df[df["stage"].isin(RUN_LEVEL_STAGES)]
    per_case = df[~df["stage"].isin(RUN_LEVEL_STAGES) & (df["cid"] != "-")].copy()

    # Warn about cached brain masks: they are not a real segmentation cost
    n_cached = (per_case["stage"] == "totalsegmentator_cached").sum()
    if n_cached > 0:
        print(
            f"NOTE: {n_cached} case(s) reused a cached brain mask. Those rows reflect "
            "a cache hit, not segmentation time.\n"
        )

    # One row per case, one column per raw stage
    wide = per_case.pivot_table(
        index="cid", columns="stage", values="seconds", aggfunc="sum"
    )

    # Stages absent for a case (e.g. process_case when no _boxes.pkl existed) become
    # NaN in the pivot. Treat them as zero contribution to that case's total, but keep
    # a record of which cases were incomplete
    incomplete = wide[wide.isna().any(axis=1)].index.tolist()
    wide = wide.fillna(0.0)

    # Group raw stages into pipeline components
    comp = pd.DataFrame(index=wide.index)
    for stage, component in COMPONENTS.items():
        if stage in wide.columns:
            comp[component] = comp.get(component, 0.0) + wide[stage]
    comp["case_total"] = comp.sum(axis=1)

    table = pd.concat([wide, comp], axis=1).sort_values("case_total", ascending=False)

    # Statistics across cases, for every raw stage and every component
    rows = {}
    for col in table.columns:
        rows[col] = mean_ci(table[col].values, alpha=args.alpha, n_boot=args.boot)
    summary = pd.DataFrame(rows).T
    summary["mean_95ci"] = summary.apply(fmt_ci, axis=1)

    order = [c for c in COMPONENTS.values() if c in summary.index]
    order = list(dict.fromkeys(order)) + ["case_total"]
    order += [c for c in summary.index if c not in order]
    summary = summary.loc[order]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=" * 78)
    print("PER-CASE TIMINGS (seconds)")
    print("=" * 78)
    print(table.round(2).to_string())
    if incomplete:
        print(f"\nCases missing at least one stage (counted as 0): {incomplete}")

    print("\n" + "=" * 78)
    print(f"ACROSS-CASE STATISTICS (n = {len(table)} cases)")
    print("=" * 78)
    cols = ["n", "mean", "sd", "median", "min", "max", "total", "mean_95ci"]
    if args.boot > 0:
        cols = cols[:-1] + ["boot_ci_low", "boot_ci_high", "mean_95ci"]
    print(summary[cols].round(2).to_string())

    if len(run_level) > 0:
        print("\nRun-level stages (once per invocation, excluded from per-case stats):")
        for stage, grp in run_level.groupby("stage"):
            secs = pd.to_numeric(grp["seconds"], errors="coerce").dropna()
            if len(secs) == 1:
                print(f"  {stage:<24}{secs.iloc[0]:.2f}s")
            else:
                print(
                    f"  {stage:<24}{secs.sum():.2f}s total over {len(secs)} run(s) "
                    f"(mean {secs.mean():.2f}s)"
                )

    if "peak_rss_gb" in df.columns:
        peak = pd.to_numeric(df["peak_rss_gb"], errors="coerce").max()
        rss = pd.to_numeric(df["rss_gb"], errors="coerce")
        print(
            f"\nMemory: peak RSS {peak:.2f} GB | max RSS observed at a stage "
            f"boundary {rss.max():.2f} GB"
        )
        worst = df.loc[rss.idxmax()] if rss.notna().any() else None
        if worst is not None:
            print(f"  highest at stage '{worst['stage']}' (case {worst['cid']})")

    grand = table["case_total"].sum() + run_level["seconds"].sum()
    print(f"\nSum of all recorded time: {grand:.2f}s ({grand / 60:.2f} min)")

    if len(table) < 10:
        print(
            f"\nWARNING: n = {len(table)} is small. The t-interval assumes roughly "
            "normal case times and is wide and unstable at this sample size; report "
            "the median and range alongside it, or use --boot for a percentile "
            "bootstrap that does not assume normality."
        )

    if args.out_prefix:
        table.round(4).to_csv(f"{args.out_prefix}_per_case.csv")
        summary.round(4).to_csv(f"{args.out_prefix}_summary.csv")
        print(
            f"\nWritten: {args.out_prefix}_per_case.csv, {args.out_prefix}_summary.csv"
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="timings.csv produced by postprocess_end2end.py")
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Write per-case and summary CSVs with this prefix",
    )
    parser.add_argument(
        "--alpha", default=0.05, type=float, help="1 - confidence level (default 0.05)"
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Drop cases missing stages (e.g. the case interrupted by a crash)",
    )
    parser.add_argument(
        "--boot",
        default=0,
        type=int,
        help="Bootstrap resamples for a non-parametric CI (e.g. 10000; 0 disables)",
    )
    args = parser.parse_args()
    assert os.path.exists(args.csv), f"File '{args.csv}' does not exist"
    return args


if __name__ == "__main__":
    main(get_args())
