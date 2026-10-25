#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Derive the sizes at which pffft_new_setup() picks the 8-lane AVX2 backend.

Reads run_matrix.py CSVs, one per independent sweep, each comparing a build
with PFFFT_ENABLE_WIDE_DISPATCH=OFF (label 'native4') with one that uses
the 8-lane backend for every size it supports (label 'wide'). A transform
type and size is enabled only if, in every sweep:

  * each of the eight transform modes and the FFT-multiply-IFFT pipeline is
    at least 1.10x as fast in the median, with the interquartile ranges of
    the two builds' samples not overlapping;
  * zreorder and zconvolve are not slower beyond noise (median >= 0.97x).

That is the table for the measured CPUs. The table for the inferred CPUs
(see src/pffft_cpu_x86.h) uses 1.25x instead of 1.10x, to leave a margin
for microarchitectural differences, in every sweep of every group.

Every sample must be a finite positive time, every 'native4' sample must
come from the 4-lane backend and all 'wide' samples of a case from the same
backend. Sizes the 8-lane backend does not support (samples labelled -x4
in the wide build), sizes with fewer than 3 samples per build and case and
sizes missing from any sweep are not enabled.

Each measured CPU group (see pffft_x86_wide_class()) has its own table,
from all of its sweeps (different models and systems of one
microarchitecture: a size must pass on each).

usage: wide_size_table.py --output src/pffft_wide_sizes.h
                          --group zen3 SWEEP.csv[.gz]... [--group zen4 SWEEP...]...
       wide_size_table.py --report SWEEP.csv[.gz]...
       wide_size_table.py --self-test
"""
import argparse
import collections
import csv
import gzip
import math
import statistics
import textwrap

GATED = ["forward", "backward", "ordered-forward", "ordered-backward",
         "inplace-forward", "inplace-backward", "inplace-ordered-forward",
         "inplace-ordered-backward", "pipeline"]
AUXILIARY = ["reorder", "zconv"]
MIN_SPEEDUP = 1.10
MIN_SPEEDUP_INFERRED = 1.25
MIN_AUXILIARY = 0.97


def read_sweep(path):
    """{(kind, n, operation, build): [(ns_per_call, backend), ...]}"""
    opener = gzip.open if path.endswith(".gz") else open
    samples = collections.defaultdict(list)
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if row["precision"] != "float":
                continue
            key = (row["kind"], int(row["n"]), row["operation"], row["build"])
            samples[key].append((float(row["ns_per_call"]), row["backend"]))
    return samples


def quartiles(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[2]


def judge(samples, kind, n, min_speedup=MIN_SPEEDUP):
    """(passed, reason) for one sweep"""
    worst = None
    for op in GATED + AUXILIARY:
        narrow = samples.get((kind, n, op, "native4"), [])
        wide = samples.get((kind, n, op, "wide"), [])
        if len(narrow) < 3 or len(wide) < 3:
            return False, f"{op} not measured (at least 3 samples per build)"
        times = [t for t, _ in narrow + wide]
        if not all(math.isfinite(t) and t > 0 for t in times):
            return False, f"{op} invalid timing"
        if not all(b.endswith("-x4") for _, b in narrow):
            return False, f"{op} baseline not 4-lane"
        lanes = {b for _, b in wide}
        if len(lanes) != 1:
            return False, f"{op} mixed backends in the 8-lane build"
        if not lanes.pop().endswith("-x8"):
            return False, "not supported by the 8-lane backend"
        narrow, wide = [t for t, _ in narrow], [t for t, _ in wide]
        ratio = statistics.median(narrow) / statistics.median(wide)
        if op in GATED:
            if not ratio >= min_speedup:
                return False, f"{op} {ratio:.3f}x"
            if not quartiles(wide)[1] < quartiles(narrow)[0]:
                return False, f"{op} interquartile ranges overlap"
            worst = ratio if worst is None else min(worst, ratio)
        elif not ratio >= MIN_AUXILIARY:
            return False, f"{op} {ratio:.3f}x"
    return True, f"slowest mode {worst:.3f}x"


def tables(paths, sweeps=None):
    """(enabled, inferred, report lines) from one group's sweeps"""
    sweeps = sweeps if sweeps is not None else [read_sweep(p) for p in paths]
    sizes = sorted({(k[0], k[1]) for s in sweeps for k in s})
    enabled = {"complex": [], "real": []}
    inferred = {"complex": [], "real": []}
    lines = []
    for kind, n in sizes:
        verdicts = [judge(s, kind, n) for s in sweeps]
        ok = all(v[0] for v in verdicts)
        ok_inferred = all(judge(s, kind, n, MIN_SPEEDUP_INFERRED)[0] for s in sweeps)
        if ok:
            enabled[kind].append(n)
        if ok_inferred:
            inferred[kind].append(n)
        tag = ("on " if ok else "off") + ("+inf" if ok_inferred else "    ")
        lines.append(f"{kind:7} {n:8} {tag}  " + "; ".join(v[1] for v in verdicts))
    return enabled, inferred, lines


def write_header(path, groups, inferred):
    """groups: [(name, source paths, enabled)]"""
    with open(path, "w") as out:
        out.write("/* SPDX-License-Identifier: BSD-3-Clause\n"
                  " * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>\n"
                  " */\n\n")
        out.write("/* Generated by benchmarks/perf_analysis/wide_size_table.py from\n")
        for name, sources, _ in groups:
            text = f"{name}: " + ", ".join(p.rsplit('/', 1)[-1] for p in sources)
            out.write("\n".join(textwrap.wrap(text, 74, initial_indent="   ", subsequent_indent="     ",
                                             break_on_hyphens=False))
                      + "\n")
        out.write("   (see benchmarks/perf_analysis/README.md); do not edit by hand.\n\n"
                  "   Sizes at which pffft_new_setup() uses the 8-lane AVX2 backend on\n"
                  "   each measured CPU group of pffft_x86_wide_class() (rule: 1.10x in\n"
                  "   each of the group's sweeps) and on the inferred CPUs (1.25x in every\n"
                  "   sweep, within the L2 limit of pffft_wide_select.h), in increasing\n"
                  "   order; the arrays hold one unused element when their count is 0. */\n\n"
                  "#ifndef PFFFT_WIDE_SIZES_H\n#define PFFFT_WIDE_SIZES_H\n")
        for table, prefix in [(enabled, f"{name}_") for name, _, enabled in groups] + [(inferred, "inferred_")]:
            for kind in ("complex", "real"):
                values = table[kind]
                out.write(f"\n#define PFFFT_WIDE_{prefix.upper()}{kind.upper()}_COUNT {len(values)}\n")
                out.write(f"static const int pffft_wide_{prefix}{kind}_sizes[] = {{")
                for i, n in enumerate(values or [0]):
                    out.write(("\n  " if i % 10 == 0 else " ") + f"{n},")
                out.write("\n};\n")
        out.write("\n#endif /* PFFFT_WIDE_SIZES_H */\n")


def self_test():
    """negative controls for judge() and the empty-table output"""
    import os
    import tempfile

    def sweep(narrow, wide, narrow_lanes="SSE1-x4", wide_lanes=("SSE1-x8",) * 3):
        s = {}
        for op in GATED + AUXILIARY:
            s[("complex", 64, op, "native4")] = [(t, narrow_lanes) for t in narrow]
            s[("complex", 64, op, "wide")] = list(zip(wide, wide_lanes))
        return s

    good = sweep([200, 201, 202], [100, 101, 102])
    assert judge(good, "complex", 64)[0]
    assert judge(good, "complex", 64, MIN_SPEEDUP_INFERRED)[0]
    # 1.20x: enough for the measured table, not for the inferred one
    modest = sweep([120, 121, 122], [100, 100.5, 101])
    assert judge(modest, "complex", 64)[0]
    assert not judge(modest, "complex", 64, MIN_SPEEDUP_INFERRED)[0], "inferred margin"
    cases = {
        "NaN timing": sweep([200, 201, 202], [100, float("nan"), 102]),
        "infinite baseline": sweep([200, float("inf"), 202], [100, 101, 102]),
        "zero timing": sweep([200, 201, 202], [0, 101, 102]),
        "mixed backends": sweep([200, 201, 202], [100, 101, 102],
                                wide_lanes=("SSE1-x4", "SSE1-x4", "SSE1-x8")),
        "8-lane baseline": sweep([200, 201, 202], [100, 101, 102], narrow_lanes="SSE1-x8"),
        "unsupported size": sweep([200, 201, 202], [100, 101, 102], wide_lanes=("SSE1-x4",) * 3),
        "too slow": sweep([105, 106, 107], [100, 101, 102]),
        "overlap": sweep([150, 160, 170], [100, 140, 175]),   # medians 1.14x apart
        "too few samples": sweep([200, 201], [100, 101]),
        "missing": {},
    }
    for name, s in cases.items():
        assert not judge(s, "complex", 64)[0], name
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sizes.h")
        write_header(path, [("zen3", ["none.csv"], {"complex": [64, 128], "real": []}),
                            ("zen4", ["other.csv"], {"complex": [], "real": [128]})],
                     {"complex": [64], "real": []})
        text = open(path).read()
        assert "#define PFFFT_WIDE_ZEN3_COMPLEX_COUNT 2" in text and "#define PFFFT_WIDE_ZEN3_REAL_COUNT 0" in text
        assert "#define PFFFT_WIDE_ZEN4_COMPLEX_COUNT 0" in text and "#define PFFFT_WIDE_ZEN4_REAL_COUNT 1" in text
        assert "#define PFFFT_WIDE_INFERRED_COMPLEX_COUNT 1" in text
        assert "#define PFFFT_WIDE_INFERRED_REAL_COUNT 0" in text
    print("wide_size_table.py self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--group", action="append", nargs="+", metavar=("NAME", "SWEEP"),
                        help="a measured CPU group and its sweeps (repeatable)")
    parser.add_argument("sweeps", nargs="*")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    groups = [(g[0], g[1:]) for g in (args.group or [])]
    if args.sweeps:
        groups.insert(0, ("zen3", args.sweeps))
    if not groups or any(not paths for _, paths in groups):
        parser.error("no sweep CSVs given")
    out, every = [], []
    if args.report:
        print("# per group and size: 'on' enabled for the group; '+inf' at least 1.25x in the\n"
              "# group's sweeps (the inferred table needs that in every group's); the\n"
              "# verdict of each sweep in order")
    for name, paths in groups:
        sweeps = [read_sweep(p) for p in paths]
        every += sweeps
        enabled, _, lines = tables(paths, sweeps)
        out.append((name, paths, enabled))
        if args.report:
            print(f"# {name}")
            print("\n".join(lines))
        print(f"{name}: enabled {len(enabled['complex'])} complex, {len(enabled['real'])} real sizes")
    _, inferred, _ = tables([], every)
    print(f"inferred (every sweep): {len(inferred['complex'])} complex, {len(inferred['real'])} real sizes")
    if args.output:
        write_header(args.output, out, inferred)


if __name__ == "__main__":
    main()
