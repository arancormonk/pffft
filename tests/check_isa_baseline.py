#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
"""Check that a baseline x86-64 build keeps AVX code inside the 8-lane backend.

Disassembles the given objects or binaries with objdump and reports every
VEX- or EVEX-encoded instruction (including VEX-128 forms such as vxorps
xmm, 128-bit FMA and vzeroupper) in a function whose name does not start
with one of the 8-lane backend's prefixes. Only meaningful for builds
configured for the baseline ISA (TARGET_C_ARCH=none); a host-tuned build
legitimately uses AVX everywhere. The exemption is by name: code that runs
before the CPU check (size checks, the CPUID probe) must not use these
prefixes.

usage: check_isa_baseline.py [--objdump PATH] FILE...
       (exit status 1 on findings or when a file holds no functions)
"""
import re
import subprocess
import sys

WIDE_PREFIXES = ("pffft_w8_", "w8_")
FUNCTION = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")


def wide(name):
    return name.startswith(WIDE_PREFIXES)


def check(path, objdump):
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", "-w", path],
                         check=True, capture_output=True, text=True).stdout
    findings = []
    current, checked = None, 0
    for line in out.splitlines():
        m = FUNCTION.match(line)
        if m:
            current = m.group(1)
            checked += 1
            continue
        if current is None or wide(current):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip().endswith(":"):
            continue
        mnemonic = parts[1].split()[0] if parts[1].split() else ""
        # every VEX/EVEX mnemonic starts with 'v'; the legacy ones that do
        # (verr, verw) never occur in this code
        if mnemonic.startswith("v") and mnemonic not in ("verr", "verw"):
            findings.append((current, line.strip()))
    return checked, findings


def main():
    args = sys.argv[1:]
    objdump = "objdump"
    if len(args) >= 2 and args[0] == "--objdump":
        objdump, args = args[1], args[2:]
    if not args:
        print(__doc__)
        return 2
    status = 0
    for path in args:
        checked, findings = check(path, objdump)
        if checked == 0:
            # e.g. an archive of LTO bitcode: nothing was actually checked
            status = 1
            print(f"{path}: no functions disassembled")
        elif findings:
            status = 1
            print(f"{path}: {len(findings)} AVX instructions outside the 8-lane backend:")
            for name, line in findings[:40]:
                print(f"  {name}: {line}")
        else:
            print(f"{path}: OK ({checked} functions, no AVX outside the 8-lane backend)")
    return status


if __name__ == "__main__":
    sys.exit(main())
