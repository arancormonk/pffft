#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#
# Runs the tests of a baseline x86-64 build under Intel SDE (CI):
#  - chips without AVX2 (Nehalem, Sandy Bridge): the 8-lane backend must be
#    unavailable and the tests must pass without executing anything the chip
#    lacks in the checked images (the test executable, and pffft.dll in a
#    DLL build);
#  - chips with AVX2: test_pffft_wide must pass and derive the CPU class of
#    src/pffft_cpu_x86.h from the emulated CPUID ("raw class");
#  - an AVX2 chip with SDE's AVX/SSE transition checker: no transitions
#    while test_pffft_wide runs (it forces the 8-lane code);
#  - negative controls: an unguarded AVX2 instruction in an executable (and
#    in a DLL) must be reported under -snb, and the transition checker must
#    report the program in AST_NEGATIVE (if set), so that a pass means something.
#
# usage: [AST_NEGATIVE=EXE] run_sde_checks.sh SDE BUILD_DIR NEG_EXE [NEG_CALLER_EXE NEG_DLL]
set -u
sde=$1 build=$2 neg=$3 negcaller=${4:-} negdll=${5:-}
# the runs change directory: absolute paths
abs() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
sde=$(abs "$sde")
ext=""
[ -f "$build/tests/test_pffft_wide.exe" ] && ext=".exe"
dll=$(find "$build" -maxdepth 1 \( -name pffft.dll -o -name libpffft.dll \) | head -1)
status=0
fail() { echo "FAIL: $*"; status=1; }

scope() {   # SDE chip-check scope for executable $1
  if [ -n "$dll" ]; then
    echo "-chip_check_image $(basename "$1") -chip_check_image $(basename "$dll")"
  else
    echo "-chip_check_exe_only"
  fi
}

run() {     # chip, executable: prints the output, returns SDE's status
  local chip=$1 exe=$2
  # shellcheck disable=SC2046
  (cd "$(dirname "$exe")" && "$sde" -"$chip" $(scope "$exe") -- "./$(basename "$exe")") 2>&1
}

"$sde" -version | head -3

# no AVX2: 4 lanes, every instruction valid for the chip
for chip in nhm snb; do
  for t in test_pffft_wide test_pffft_accuracy_float; do
    exe="$build/tests/$t$ext"
    out=$(run "$chip" "$exe"); rc=$?
    echo "== $chip $t: rc=$rc $(echo "$out" | grep -E "backend|SDE-ERROR|passed|FAIL" | head -3 | tr '\n' '|')"
    [ $rc -eq 0 ] || fail "$chip $t exited with $rc"
    echo "$out" | grep -q "SDE-ERROR" && fail "$chip $t executed an instruction the chip lacks"
    if [ "$t" = test_pffft_wide ]; then
      echo "$out" | grep -q "8-lane backend not available" || fail "$chip: 8-lane backend not reported unavailable"
    fi
  done
done

# AVX2 chips: the class pffft derives from their CPUID; the 8-lane backend
# must run there (an unavailable backend fails test_pffft_wide)
export PFFFT_TEST_REQUIRE_WIDE=1
expect() {
  case $1 in
    hsw|bdw|skl|cnl|icl|tgl) echo inferred;;
    *) echo off;;          # servers, hybrids, Atom-only parts, future families
  esac
}
for chip in hsw bdw skl cnl icl tgl skx clx icx spr emr gnr dmr adl rpl mtl arl lnl ptl srf cwf nvl; do
  exe="$build/tests/test_pffft_wide$ext"
  out=$(run "$chip" "$exe"); rc=$?
  cpu=$(echo "$out" | grep -E "^CPU vendor" | head -1)
  echo "== $chip: rc=$rc $cpu"
  [ $rc -eq 0 ] || fail "$chip test_pffft_wide exited with $rc"
  echo "$out" | grep -q "all checks passed" || fail "$chip test_pffft_wide did not pass"
  echo "$out" | grep -q "8-lane backend available: testing it" || fail "$chip: the 8-lane backend did not run"
  echo "$out" | grep -q "SDE-ERROR" && fail "$chip executed an instruction the chip lacks"
  echo "$cpu" | grep -q "raw class $(expect "$chip")\$" || fail "$chip: expected raw class $(expect "$chip")"
done

# AVX/SSE transitions while the 8-lane code runs (test_pffft_wide forces
# it): SDE's transition checker on an AVX2 chip must count none
ast() {     # executable [expected output]: prints "transitions N" or an error
  local exe=$1 want=${2:-} dir out rc report n
  dir=$(cd "$(dirname "$exe")" && pwd)
  report="$dir/sde-avx-sse-transition-out.txt"
  rm -f "$report"
  out=$(cd "$dir" && "$sde" -hsw -ast -chip_check_disable -- "./$(basename "$exe")" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "error: the program failed ($rc)"; return; fi
  if [ -n "$want" ] && ! echo "$out" | grep -q "$want"; then echo "error: no '$want' in its output"; return; fi
  if [ ! -f "$report" ]; then echo "error: no transition report"; return; fi
  n=$(grep -E "^# (AVX_to_SSE|SSE_to_AVX)_transition_instances:[[:space:]]+[0-9]+$" "$report" | awk '{s += $NF; c++} END {if (c == 2) print s}')
  if [ -z "$n" ]; then echo "error: transition counters missing from the report"; return; fi
  echo "transitions $n"
}
# the checker's own failure modes must not pass: a program that fails, and
# one whose expected output is missing
r=$(ast "$neg" "no such output"); echo "== transition checker control (missing output): $r"
[ "${r%% *}" = "error:" ] || fail "the transition checker accepted a run without the expected output"
r=$(ast "$(dirname "$neg")/no-such-program$ext"); echo "== transition checker control (failed program): $r"
[ "${r%% *}" = "error:" ] || fail "the transition checker accepted a failed run"
r=$(ast "$build/tests/test_pffft_wide$ext" "all checks passed")
echo "== transitions (test_pffft_wide, -hsw): $r"
[ "$r" = "transitions 0" ] || fail "test_pffft_wide under the transition checker: $r"
if [ -n "${AST_NEGATIVE:-}" ]; then
  r=$(ast "$AST_NEGATIVE" "done"); echo "== transitions (negative control): $r"
  case $r in
    "transitions 0"|error:*) fail "the transition checker missed the negative control: $r";;
  esac
fi
unset PFFFT_TEST_REQUIRE_WIDE

# negative controls
out=$( (cd "$(dirname "$neg")" && "$sde" -snb -chip_check_exe_only -- "./$(basename "$neg")") 2>&1 ); rc=$?
echo "== negative control (executable): rc=$rc $(echo "$out" | grep SDE-ERROR | head -1)"
[ $rc -ne 0 ] && echo "$out" | grep -q "SDE-ERROR: Executed instruction not valid" \
  || fail "unguarded AVX2 in the executable not reported"
if [ -n "$negcaller" ]; then
  out=$( (cd "$(dirname "$negcaller")" && "$sde" -snb -chip_check_image "$(basename "$negdll")" -- \
          "./$(basename "$negcaller")") 2>&1 ); rc=$?
  echo "== negative control (DLL): rc=$rc $(echo "$out" | grep SDE-ERROR | head -1)"
  [ $rc -ne 0 ] && echo "$out" | grep -q "SDE-ERROR: Executed instruction not valid" \
    || fail "unguarded AVX2 in a DLL not reported"
fi

[ $status -eq 0 ] && echo "SDE checks passed"
exit $status
