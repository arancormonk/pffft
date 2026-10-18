#!/usr/bin/env python3
"""Compile the standalone driver and measure existing CMake builds serially.

Example: python3 benchmarks/perf_analysis/run_matrix.py \
  --build release=build_simd_analysis/release --build native=build_simd_analysis/native \
  --output build_simd_analysis/results/matrix.csv --cpu 2
No CMake settings or production sources are changed by this script.
"""
import argparse
import csv
import itertools
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["build", "precision", "backend", "kind", "operation", "n", "repeat",
           "iterations", "seconds", "ns_per_call", "checksum"]


def cpu_snapshot():
    """Linux aggregate CPU accounting; guest time is already in user time."""
    fields = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return sum(fields), sum(fields) - fields[3] - fields[4]


def busy_cpus(before, after):
    return (os.cpu_count() or 1) * (after[1] - before[1]) / max(1, after[0] - before[0])


def wait_for_idle(limit):
    last_message = 0
    while True:
        before = cpu_snapshot()
        time.sleep(0.1)
        busy = busy_cpus(before, cpu_snapshot())
        # Leave one CPU of the limit available for the benchmark itself.
        if busy <= limit - 1:
            return
        if time.monotonic() - last_message >= 10:
            print(f"Waiting for host load: {busy:.1f} CPUs busy", file=sys.stderr, flush=True)
            last_message = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True, help="LABEL=CMAKE_BUILD_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.05)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384, 65536])
    parser.add_argument("--ops", nargs="+", default=["forward", "backward", "ordered-forward",
                                                  "ordered-backward", "reorder", "zconv", "pipeline"])
    parser.add_argument("--kinds", nargs="+", default=["real", "complex"])
    parser.add_argument("--precisions", nargs="+", default=["float", "double"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--interleave", action="store_true",
                        help="Alternate builds between samples using fresh processes")
    parser.add_argument("--max-busy-cpus", type=float,
                        help="Linux: wait for quiet host, reject samples exceeding this total CPU load")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interleaved CSV with unchanged binaries and parameters")
    args = parser.parse_args()
    if args.resume and not args.interleave:
        parser.error("--resume requires --interleave")
    if args.max_busy_cpus is not None and args.max_busy_cpus <= 1:
        parser.error("--max-busy-cpus must exceed 1")
    builds = []
    for spec in args.build:
        label, path = spec.split("=", 1)
        path = Path(path).resolve()
        builds.append((label, path))
        if not args.no_compile:
            for precision in args.precisions:
                command = [args.compiler, "-O3", "-DNDEBUG", "-std=c++11", "-Wall", "-Wextra",
                           "-pedantic", "-Werror", "-DPFFFT_STATIC_DEFINE", "-DPFFASTCONV_STATIC_DEFINE",
                           "-I" + str(ROOT / "include"), str(Path(__file__).with_name("bench_hotloops.cpp"))]
                if precision == "double":
                    command += ["-DPERF_DOUBLE"]
                else:
                    command += [str(path / "libpffastconv.a")]
                command += [str(path / "libpffft.a"), "-lm", "-o", str(path / ("hotloops_" + precision))]
                subprocess.run(command, check=True)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = list(itertools.product(args.precisions, args.kinds, args.ops, args.sizes))
    cases = [(p, k, op, n) for p, k, op, n in cases
             if not (op == "zp" and k == "complex")
             and not (op.startswith("fastconv") and p == "double")
             and not (op == "fastconv-single" and k == "real")]
    rng = random.Random(1729)
    rng.shuffle(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume:
        with args.output.open(newline="") as previous:
            reader = csv.DictReader(previous)
            if reader.fieldnames != COLUMNS:
                raise RuntimeError("Unexpected CSV header")
            for row in reader:
                if any(row.get(field) is None for field in COLUMNS):
                    raise RuntimeError("Incomplete CSV row")
                completed.add(tuple(row[field] for field in
                                    ["build", "precision", "kind", "operation", "n", "repeat"]))
    # Exclusive creation prevents an accidental rerun from replacing evidence.
    mode = "a" if args.resume else "x"
    with args.output.open(mode, newline="") as out, args.output.with_suffix(".log").open(mode) as log:
        writer = csv.writer(out, lineterminator="\n")
        if not args.resume:
            writer.writerow(COLUMNS)
        for index, (precision, kind, operation, n) in enumerate(cases):
            runs = []
            for repeat in range(args.repeats if args.interleave else 1):
                order = list(builds)
                rng.shuffle(order)
                runs.extend((repeat, label, path) for label, path in order)
            for repeat, label, path in runs:
                if (label, precision, kind, operation, str(n), str(repeat)) in completed:
                    continue
                command = [str(path / ("hotloops_" + precision)), kind, operation, str(n),
                           str(1 if args.interleave else args.repeats), str(args.seconds)]
                while True:
                    if args.max_busy_cpus is not None:
                        wait_for_idle(args.max_busy_cpus)
                        before = cpu_snapshot()
                    result = subprocess.run(command, text=True, capture_output=True, check=True)
                    if args.max_busy_cpus is None:
                        break
                    busy = busy_cpus(before, cpu_snapshot())
                    if busy <= args.max_busy_cpus:
                        break
                    log.write(f"Rejected for host load ({busy:.2f} busy CPUs): {command}\n")
                    log.flush()
                log.write(f"{label} {command}\n{result.stderr}")
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) != len(COLUMNS) - 1:
                        raise RuntimeError(f"malformed output: {row}")
                    if args.interleave:
                        row[5] = str(repeat)
                    writer.writerow([label] + row)
                out.flush()
                log.flush()
            if index % 10 == 0 or index == len(cases) - 1:
                print(f"{index + 1}/{len(cases)} cases complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
