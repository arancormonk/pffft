#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Static throughput cross-check of the 8-lane backend's speedup per CPU model.

For each case (transform type, N, mode), each build (the shipping baseline
GCC build with and without the 8-lane backend) and each llvm-mca scheduling
model:

  predicted cycles per call = sum over basic blocks b of
      (executions of b per call, counted by valgrind's callgrind in
       mca_driver.c's mca_region) x (llvm-mca steady-state cycles per
       iteration of b on the model)

This covers every executed instruction of the operation: passes,
preprocess/finalize, copies, reordering. llvm-mca has no memory hierarchy,
so only L1-resident sizes are meaningful.

Decision rule (fixed before the data, see benchmarks/perf_analysis/README.md
"CPU classes"), over the cases whose N is in the inferred size table and
L1-resident (complex N <= 1024, real N <= 2048), all eight modes:

  calibration  znver3's predicted speedup (4-lane / 8-lane cycles) is within
               15% of the speedup measured on the Zen 3 host for the same
               build, in every case; otherwise the analysis is no evidence.
  control      znver1 (256-bit operations split into two 128-bit halves) has
               a bias-corrected speedup (below) of at most 1.05x in the
               median; otherwise the method cannot tell datapath widths apart.
  gate         every inferred model's bias-corrected speedup
                 measured(case) x predicted_model(case) / predicted_znver3(case)
               is at least 1.10x in every case (the measured table's rule).

usage: mca_predict.py --wide DIR --native4 DIR --measured CSV[.gz] --output PREFIX
                      [--models m1,m2,...] [--calls K]
"""
import argparse
import collections
import csv
import gzip
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = ["forward", "backward", "ordered-forward", "ordered-backward", "inplace-forward",
         "inplace-backward", "inplace-ordered-forward", "inplace-ordered-backward"]
# every inferred group of src/pffft_cpu_x86.h -> LLVM -mcpu, plus the controls
MODELS = ["znver3", "znver1", "znver2", "znver4", "znver5", "haswell", "broadwell", "skylake",
          "cannonlake", "icelake-client", "tigerlake", "rocketlake", "alderlake", "raptorlake"]
CONTROL = "znver1"
CALIBRATION = "znver3"
L1_LIMIT = {"complex": 1024, "real": 2048}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def build_driver(libdir):
    exe = libdir / "mca_driver"
    run(["cc", "-O2", "-std=c99", "-no-pie", "-I" + str(ROOT / "include"), str(HERE / "mca_driver.c"),
         str(libdir / "libpffft.a"), "-lm", "-o", str(exe)])
    return exe


def counts(exe, kind, mode, n, calls):
    """{address: executions per call} inside mca_region"""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cg.out")
        subprocess.run(["valgrind", "--tool=callgrind", "--dump-instr=yes", "--compress-pos=no",
                        "--compress-strings=no", "--collect-atstart=no", "--toggle-collect=mca_region",
                        "--callgrind-out-file=" + out, str(exe), kind, mode, str(n), str(calls)],
                       check=True, capture_output=True, text=True)
        acc = collections.Counter()
        skip = False
        for line in open(out):
            if line.startswith("calls="):
                skip = True          # the next cost line is the call's inclusive cost
                continue
            m = re.match(r"^0x([0-9a-f]+) \d+ (\d+)$", line)
            if m:
                if skip:
                    skip = False
                    continue
                acc[int(m.group(1), 16)] += int(m.group(2))
            elif line.strip() and not line[0].isdigit() and not line.startswith(("0x", "+", "-", "*")):
                skip = False
    return {a: c / calls for a, c in acc.items()}


def disassemble(exe):
    """[(function, [(addr, text)])] of the executable"""
    funcs, cur = [], None
    for line in run(["objdump", "-d", "--no-show-raw-insn", "-w", str(exe)]).splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:$", line)
        if m:
            cur = (m.group(2), []); funcs.append(cur); continue
        m = re.match(r"^\s*([0-9a-f]+):\t(.*)$", line)
        if m and cur is not None and m.group(2).strip():
            cur[1].append((int(m.group(1), 16), m.group(2).split("#")[0].strip()))
    return funcs


def blocks(funcs, executed):
    """maximal runs of instructions with the same execution count, cut after
    branches, calls and returns: (id, [text], {address of first})"""
    out = []
    for name, insns in funcs:
        cur, cur_count = [], None
        for addr, text in insns:
            c = executed.get(addr, 0)
            if cur and c != cur_count:
                out.append((cur[0][0], [t for _, t in cur], cur_count)); cur = []
            cur.append((addr, text)); cur_count = c
            mnem = text.split()[0] if text.split() else ""
            if mnem.startswith(("j", "call", "ret")) or mnem in ("notrack", "bnd"):
                out.append((cur[0][0], [t for _, t in cur], cur_count)); cur = []
        if cur:
            out.append((cur[0][0], [t for _, t in cur], cur_count))
    return [b for b in out if b[2]]


def mca_text(block_id, texts):
    lines = [f"# LLVM-MCA-BEGIN b{block_id:x}"]
    for t in texts:
        toks = t.split()
        while toks and toks[0] in ("notrack", "bnd", "cs", "ds", "data16", "rex.W", "rex"):
            toks.pop(0)
        if not toks:
            continue
        m = toks[0]
        if m.startswith(("call", "ret", "nop", "xchg", "endbr", "ud2", "int3", "hlt")):
            continue
        if m.startswith("j"):
            if len(toks) > 1 and toks[1].startswith("*"):
                continue           # indirect jump: no static target
            lines.append(f"{m} .Lb{block_id:x}"); continue
        lines.append(" ".join(toks))
    lines.append("# LLVM-MCA-END")
    return [f".Lb{block_id:x}:"] + lines


def mca_cycles(model, regions):
    """{block id: cycles per iteration}; drops instructions llvm-mca cannot
    parse and reports them"""
    regions = {k: list(v) for k, v in regions.items()}
    dropped = []
    for _ in range(200):
        text = []
        order = sorted(regions)
        line_of = {}
        for k in order:
            for l in mca_text(k, regions[k]):
                text.append(l)
                line_of[len(text)] = (k, l)
        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write("\n".join(text) + "\n")
            path = f.name
        p = subprocess.run(["llvm-mca", "-mcpu=" + model, "-iterations=100", "-skip-unsupported-instructions=any",
                            "-all-stats=false", "-all-views=false", "-summary-view", path],
                           capture_output=True, text=True)
        os.unlink(path)
        if p.returncode == 0:
            res, cur = {}, None
            for line in p.stdout.splitlines():
                m = re.match(r"^\[\d+\] Code Region - b([0-9a-f]+)$", line)
                if m:
                    cur = int(m.group(1), 16); continue
                m = re.match(r"^Total Cycles:\s+(\d+)$", line)
                if m and cur is not None:
                    res[cur] = int(m.group(1)) / 100.0
            return res, dropped
        m = re.search(r":(\d+):\d+: error: (.*)", p.stderr)
        if not m:
            raise RuntimeError(p.stderr[-2000:])
        k, l = line_of[int(m.group(1))]
        dropped.append((model, l, m.group(2)))
        # remove that instruction from its block
        for i, t in enumerate(regions[k]):
            if " ".join(t.split()).endswith(l.split(None, 1)[-1]) or t.strip() == l:
                del regions[k][i]
                break
        else:
            raise RuntimeError(f"cannot locate {l!r}")
    raise RuntimeError("too many unparsable instructions")


def measured_ratios(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    t = collections.defaultdict(list)
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if row["precision"] == "float":
                t[(row["build"], row["kind"], int(row["n"]), row["operation"])].append(float(row["ns_per_call"]))
    out = {}
    for (build, kind, n, op), v in t.items():
        if build != "gcc-wide":
            continue
        base = t.get(("gcc-native4", kind, n, op))
        if base:
            out[(kind, n, op)] = statistics.median(base) / statistics.median(v)
    return out


def inferred_sizes():
    text = (ROOT / "src" / "pffft_wide_sizes.h").read_text()
    out = {}
    for kind in ("complex", "real"):
        m = re.search(r"pffft_wide_" + kind + r"_inferred_sizes\[\] = \{([^}]*)\}", text)
        out[kind] = {int(x) for x in m.group(1).replace(",", " ").split()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide", type=Path, required=True)
    ap.add_argument("--native4", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--calls", type=int, default=20)
    args = ap.parse_args()
    models = args.models.split(",")
    sizes = inferred_sizes()
    measured = measured_ratios(args.measured)
    cases = [(kind, n, mode) for kind in ("complex", "real") for n in sorted(sizes[kind])
             if n <= L1_LIMIT[kind] for mode in MODES if (kind, n, mode) in measured]
    if not cases:
        sys.exit("no measured cases at L1-resident sizes of the inferred table")
    pred = {}            # (build, model, case) -> cycles per call
    dropped_all = []
    for label, libdir in (("wide", args.wide), ("native4", args.native4)):
        exe = build_driver(libdir)
        funcs = disassemble(exe)
        per_case = {c: counts(exe, c[0], c[2], c[1], args.calls) for c in cases}
        blks = {}
        per_case_blocks = {}
        for c, ex in per_case.items():
            bl = blocks(funcs, ex)
            per_case_blocks[c] = [(b[0], b[2]) for b in bl]
            for bid, texts, _ in bl:
                blks[bid] = texts
        for model in models:
            cyc, dropped = mca_cycles(model, blks)
            dropped_all += [(label,) + d for d in dropped]
            for c, bl in per_case_blocks.items():
                pred[(label, model, c)] = sum(cnt * cyc.get(bid, 0.0) for bid, cnt in bl)
        print(f"{label}: {len(blks)} blocks, {len(cases)} cases", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output) + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "n", "mode", "measured", *[f"pred_{m}" for m in models],
                    *[f"corrected_{m}" for m in models]])
        rows = []
        for c in cases:
            r = {m: pred[("native4", m, c)] / pred[("wide", m, c)] for m in models}
            corr = {m: measured[c] * r[m] / r[CALIBRATION] for m in models}
            rows.append((c, r, corr))
            w.writerow([*c, f"{measured[c]:.4f}", *[f"{r[m]:.4f}" for m in models],
                        *[f"{corr[m]:.4f}" for m in models]])
    with open(str(args.output) + ".txt", "w") as f:
        def say(s=""):
            print(s); f.write(s + "\n")
        llvm = run(["llvm-mca", "--version"]).split("\n")
        say(f"llvm-mca: {next((l.strip() for l in llvm if 'version' in l), '?')}")
        say(f"cases: {len(cases)} (kinds, L1-resident inferred-table sizes, 8 modes)")
        cal = [abs(r[CALIBRATION] / measured[c] - 1) for c, r, _ in rows]
        cal_ok = max(cal) <= 0.15
        say(f"calibration ({CALIBRATION} vs measured): max deviation {max(cal):.1%}, "
            f"median {statistics.median(cal):.1%}: {'PASS' if cal_ok else 'FAIL'}")
        ctl = statistics.median(corr[CONTROL] for _, _, corr in rows)
        ctl_ok = ctl <= 1.05
        say(f"control ({CONTROL}): median corrected speedup {ctl:.3f}x: {'PASS' if ctl_ok else 'FAIL'}")
        for m in models:
            if m in (CALIBRATION, CONTROL):
                continue
            v = [corr[m] for _, _, corr in rows]
            ok = min(v) >= 1.10
            say(f"gate {m:15s}: corrected min {min(v):.3f}x median {statistics.median(v):.3f}x: "
                f"{'PASS' if ok else 'FAIL'}")
        say(f"dropped instructions (not parsed by llvm-mca): {len(dropped_all)}")
        for d in sorted(set(dropped_all))[:20]:
            say(f"  {d}")


if __name__ == "__main__":
    main()
