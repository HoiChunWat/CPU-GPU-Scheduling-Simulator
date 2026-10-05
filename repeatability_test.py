#!/usr/bin/env python3

import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CPU-GPU Scheduling Repeatability Test
#
# Per workload:
#   5 warm-up runs
#   10 measured runs
#   median timing
#
# Whole benchmark:
#   11 complete repetitions
#
# Final decision boundaries:
#   median(lowest GPU-winning Work)
#   median(highest CPU-winning Work)
#   median(lowest GPU-winning K)
#   median(highest CPU-winning K)
#
# Existing completed run_XX.txt files are reused automatically.
# ============================================================

REPETITIONS = 11
COOLDOWN_SECONDS = 60

EXECUTABLE = Path("./build/matrix_benchmark")
BENCHMARK_CSV = Path("matrix_random_work_results.csv")
OUTPUT_DIR = Path("repeatability")
SUMMARY_CSV = OUTPUT_DIR / "repeatability_summary.csv"

# ------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------

WORK_OVERLAP_RE = re.compile(
    r"WORK OVERLAP\s*:\s*([\d,]+)\s*~\s*([\d,]+)",
    re.IGNORECASE,
)

HIGHEST_CPU_WORK_RE = re.compile(
    r"Highest CPU Work\s*:\s*([\d,]+)",
    re.IGNORECASE,
)

LOWEST_GPU_WORK_RE = re.compile(
    r"Lowest GPU Work\s*:\s*([\d,]+)",
    re.IGNORECASE,
)

# These two lines are printed whether K overlaps or separates cleanly.
LOWEST_GPU_K_RE = re.compile(
    r"Lowest GPU-winning K\s*:\s*(\d+)",
    re.IGNORECASE,
)

HIGHEST_CPU_K_RE = re.compile(
    r"Highest CPU-winning K\s*:\s*(\d+)",
    re.IGNORECASE,
)

CPU_WINS_RE = re.compile(
    r"CPU Wins\s*:\s*(\d+)",
    re.IGNORECASE,
)

GPU_WINS_RE = re.compile(
    r"GPU Wins\s*:\s*(\d+)",
    re.IGNORECASE,
)

CORRECTNESS_RE = re.compile(
    r"Correctness\s*:\s*(PASS|FAIL)",
    re.IGNORECASE,
)


def parse_integer(text: str) -> int:
    return int(text.replace(",", ""))


def discrete_median(values):
    """For 11 values, return the actual 6th observed value after sorting."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def relation(lowest_gpu, highest_cpu):
    if lowest_gpu <= highest_cpu:
        return "OVERLAP"
    return "CLEAN"


def parse_benchmark_output(output: str, run_number: int):
    # --------------------------------------------------------
    # Work boundaries
    # --------------------------------------------------------

    work_overlap_match = WORK_OVERLAP_RE.search(output)

    if work_overlap_match is not None:
        lowest_gpu_work = parse_integer(work_overlap_match.group(1))
        highest_cpu_work = parse_integer(work_overlap_match.group(2))
    else:
        highest_cpu_match = HIGHEST_CPU_WORK_RE.search(output)
        lowest_gpu_match = LOWEST_GPU_WORK_RE.search(output)

        if highest_cpu_match is None or lowest_gpu_match is None:
            raise RuntimeError(
                f"Run {run_number}: could not determine Work boundaries."
            )

        highest_cpu_work = parse_integer(highest_cpu_match.group(1))
        lowest_gpu_work = parse_integer(lowest_gpu_match.group(1))

    # --------------------------------------------------------
    # K boundaries
    #
    # Do NOT search only for "K OVERLAP".
    # A valid run may have CLEAN K separation, in which case
    # the program intentionally does not print a K OVERLAP line.
    # --------------------------------------------------------

    lowest_gpu_k_match = LOWEST_GPU_K_RE.search(output)
    highest_cpu_k_match = HIGHEST_CPU_K_RE.search(output)

    if lowest_gpu_k_match is None or highest_cpu_k_match is None:
        raise RuntimeError(
            f"Run {run_number}: could not determine K decision boundaries."
        )

    lowest_gpu_k = int(lowest_gpu_k_match.group(1))
    highest_cpu_k = int(highest_cpu_k_match.group(1))

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    cpu_match = CPU_WINS_RE.search(output)
    gpu_match = GPU_WINS_RE.search(output)
    correctness_match = CORRECTNESS_RE.search(output)

    if correctness_match is None:
        raise RuntimeError(
            f"Run {run_number}: could not find correctness status."
        )

    cpu_wins = int(cpu_match.group(1)) if cpu_match else -1
    gpu_wins = int(gpu_match.group(1)) if gpu_match else -1
    correctness = correctness_match.group(1).upper()

    if correctness != "PASS":
        raise RuntimeError(
            f"Run {run_number}: correctness failed."
        )

    return {
        "run": run_number,
        "work_low": lowest_gpu_work,
        "work_high": highest_cpu_work,
        "work_relation": relation(lowest_gpu_work, highest_cpu_work),
        "k_low": lowest_gpu_k,
        "k_high": highest_cpu_k,
        "k_relation": relation(lowest_gpu_k, highest_cpu_k),
        "cpu_wins": cpu_wins,
        "gpu_wins": gpu_wins,
        "correctness": correctness,
    }


def print_run_result(record, reused=False):
    prefix = "Reused existing result" if reused else "Completed"

    print(
        f"{prefix}\n"
        f"Work boundaries : {record['work_low']:,} ~ {record['work_high']:,} "
        f"({record['work_relation']})\n"
        f"K boundaries    : {record['k_low']} ~ {record['k_high']} "
        f"({record['k_relation']})\n"
        f"CPU/GPU wins    : {record['cpu_wins']} / {record['gpu_wins']}\n"
        f"Correctness     : {record['correctness']}"
    )


def load_existing_run(run_number: int):
    log_path = OUTPUT_DIR / f"run_{run_number:02d}.txt"

    if not log_path.exists():
        return None

    output = log_path.read_text(encoding="utf-8")

    try:
        return parse_benchmark_output(output, run_number)
    except RuntimeError:
        # Existing file is incomplete or from an incompatible format.
        return None


def run_one_benchmark(run_number: int):
    print(f"\n========== COMPLETE RUN {run_number}/{REPETITIONS} ==========")
    print("Running benchmark...")

    completed = subprocess.run(
        [str(EXECUTABLE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = completed.stdout

    log_path = OUTPUT_DIR / f"run_{run_number:02d}.txt"
    log_path.write_text(output, encoding="utf-8")

    if completed.returncode != 0:
        print(output)
        raise RuntimeError(
            f"Run {run_number} failed with exit code {completed.returncode}. "
            f"Full output saved to {log_path}"
        )

    try:
        record = parse_benchmark_output(output, run_number)
    except RuntimeError as error:
        raise RuntimeError(
            f"{error} See {log_path}"
        ) from error

    # Preserve the full per-run benchmark CSV.
    if BENCHMARK_CSV.exists():
        csv_copy = OUTPUT_DIR / f"run_{run_number:02d}.csv"
        shutil.copy2(BENCHMARK_CSV, csv_copy)

    print_run_result(record)
    return record


def main():
    if not EXECUTABLE.exists():
        print(
            f"ERROR: {EXECUTABLE} does not exist.\n"
            "Compile matrix_benchmark.cu first.",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for run_number in range(1, REPETITIONS + 1):
        # --------------------------------------------------------
        # Resume support
        #
        # If a previous attempt already completed this run,
        # reuse the saved TXT instead of wasting another benchmark.
        # --------------------------------------------------------

        existing = load_existing_run(run_number)

        if existing is not None:
            print(
                f"\n========== COMPLETE RUN {run_number}/{REPETITIONS} =========="
            )
            print_run_result(existing, reused=True)
            records.append(existing)
            continue

        record = run_one_benchmark(run_number)
        records.append(record)

        # Only cool down after a newly executed benchmark.
        if run_number < REPETITIONS:
            print(
                f"\nCooling down for {COOLDOWN_SECONDS} seconds "
                "before the next complete run..."
            )
            time.sleep(COOLDOWN_SECONDS)

    # ========================================================
    # FINAL MEDIAN BOUNDARIES
    # ========================================================

    work_lows = [r["work_low"] for r in records]
    work_highs = [r["work_high"] for r in records]
    k_lows = [r["k_low"] for r in records]
    k_highs = [r["k_high"] for r in records]

    median_work_low = discrete_median(work_lows)
    median_work_high = discrete_median(work_highs)
    median_k_low = discrete_median(k_lows)
    median_k_high = discrete_median(k_highs)

    median_work_relation = relation(
        median_work_low,
        median_work_high,
    )

    median_k_relation = relation(
        median_k_low,
        median_k_high,
    )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Run",
                "LowestGPUWork",
                "HighestCPUWork",
                "WorkRelation",
                "LowestGPUK",
                "HighestCPUK",
                "KRelation",
                "CPUWins",
                "GPUWins",
                "Correctness",
            ]
        )

        for r in records:
            writer.writerow(
                [
                    r["run"],
                    r["work_low"],
                    r["work_high"],
                    r["work_relation"],
                    r["k_low"],
                    r["k_high"],
                    r["k_relation"],
                    r["cpu_wins"],
                    r["gpu_wins"],
                    r["correctness"],
                ]
            )

        writer.writerow([])

        writer.writerow(
            [
                "Median",
                median_work_low,
                median_work_high,
                median_work_relation,
                median_k_low,
                median_k_high,
                median_k_relation,
                "",
                "",
                "",
            ]
        )

        writer.writerow(
            [
                "ObservedMin",
                min(work_lows),
                min(work_highs),
                "",
                min(k_lows),
                min(k_highs),
                "",
                "",
                "",
                "",
            ]
        )

        writer.writerow(
            [
                "ObservedMax",
                max(work_lows),
                max(work_highs),
                "",
                max(k_lows),
                max(k_highs),
                "",
                "",
                "",
                "",
            ]
        )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print(
        "\n"
        "============================================================\n"
        "              11-RUN REPEATABILITY SUMMARY\n"
        "============================================================\n"
    )

    print(
        f"{'Run':>3} | "
        f"{'GPU Work':>12} | "
        f"{'CPU Work':>12} | "
        f"{'Work':>7} | "
        f"{'GPU K':>5} | "
        f"{'CPU K':>5} | "
        f"{'K':>7} | "
        f"{'CPU':>4} | "
        f"{'GPU':>4}"
    )

    print(
        "----+--------------+--------------+---------+-------+-------+---------+------+------+"
    )

    for r in records:
        print(
            f"{r['run']:>3} | "
            f"{r['work_low']:>12,} | "
            f"{r['work_high']:>12,} | "
            f"{r['work_relation']:>7} | "
            f"{r['k_low']:>5} | "
            f"{r['k_high']:>5} | "
            f"{r['k_relation']:>7} | "
            f"{r['cpu_wins']:>4} | "
            f"{r['gpu_wins']:>4}"
        )

    print(
        "\n"
        "--------------------- MEDIAN BOUNDARIES ---------------------\n"
        f"Lowest GPU Work    : {median_work_low:,}\n"
        f"Highest CPU Work   : {median_work_high:,}\n"
        f"Work relation      : {median_work_relation}\n"
        "\n"
        f"Lowest GPU K       : {median_k_low}\n"
        f"Highest CPU K      : {median_k_high}\n"
        f"K relation         : {median_k_relation}\n"
    )

    if median_work_relation == "OVERLAP":
        print(
            f"Median Work overlap: "
            f"{median_work_low:,} ~ {median_work_high:,}"
        )
    else:
        print(
            f"Median Work gap    : "
            f"{median_work_high:,} < Work < {median_work_low:,}"
        )

    if median_k_relation == "OVERLAP":
        print(
            f"Median K overlap   : "
            f"{median_k_low} ~ {median_k_high}"
        )
    else:
        print(
            f"Median K gap       : "
            f"{median_k_high} < K < {median_k_low}"
        )

    print(
        "\n"
        "---------------------- OBSERVED RANGE -----------------------\n"
        f"Lowest GPU Work range  : {min(work_lows):,} ~ {max(work_lows):,}\n"
        f"Highest CPU Work range : {min(work_highs):,} ~ {max(work_highs):,}\n"
        f"Lowest GPU K range     : {min(k_lows)} ~ {max(k_lows)}\n"
        f"Highest CPU K range    : {min(k_highs)} ~ {max(k_highs)}\n"
        "\n"
        f"Summary CSV            : {SUMMARY_CSV}\n"
        f"Per-run logs/CSVs      : {OUTPUT_DIR}/run_01 ... run_11\n"
        "============================================================"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
