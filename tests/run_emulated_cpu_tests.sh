#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
#
# Run the float tests of a baseline x86-64 Linux build under QEMU user-mode
# emulation of CPUs without AVX (Nehalem), with AVX but without AVX2/FMA
# (SandyBridge) and with AVX2/FMA (Haswell). QEMU raises SIGILL for
# instructions the emulated CPU lacks, so the first two prove that setup
# selection never runs the 8-lane AVX2 backend there; the last runs it.
#
# The binaries are linked statically so that the host's C library, which
# may itself require a newer CPU, is not involved. Two configurations are
# built: plain, and with -fzero-call-used-regs=all, which makes the compiler
# clear registers on return with whatever the function's target allows
# (vzeroall in AVX2 code), so that code compiled for AVX2 but run before the
# CPU check cannot hide. Each is also checked by the ISA audit.
#
# usage: tests/run_emulated_cpu_tests.sh BUILD_DIR   (needs qemu-x86_64,
#        python3 and objdump; BUILD_DIR gets one subdirectory per configuration)
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
top="$1"
qemu="${QEMU:-qemu-x86_64}"

status=0
for config in plain zero-regs; do
  build="$top/$config"
  cflags=""
  if [ "$config" = zero-regs ]; then cflags="-fzero-call-used-regs=all"; fi
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DTARGET_C_ARCH=none -DTARGET_CXX_ARCH=none \
    -DPFFFT_USE_TYPE_DOUBLE=OFF -DPFFFT_BUILD_BENCHMARKS=OFF -DPFFFT_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_FLAGS="$cflags" -DCMAKE_EXE_LINKER_FLAGS=-static
  cmake --build "$build" --parallel --target test_pffft_wide test_pffft_accuracy_float

  if ! ctest --test-dir "$build" -R '^check_isa_baseline$' --output-on-failure --no-tests=error; then
    echo "FAIL: ISA audit ($config)"; status=1
  fi
  for cpu in Nehalem SandyBridge Haswell; do
    for t in test_pffft_wide test_pffft_accuracy_float; do
      log="$build/emulated-$cpu-$t.log"
      if ! "$qemu" -cpu "$cpu" "$build/tests/$t" 1024 >"$log" 2>&1; then
        echo "FAIL: $t on $cpu ($config)"; tail -5 "$log"; status=1; continue
      fi
      if [ "$cpu" = Haswell ]; then expect="8-lane backend[: ]*(available|tested)"; else expect="not available"; fi
      if grep -Eq "$expect" "$log"; then
        echo "ok: $t on $cpu ($config: $(grep -E "8-lane backend" "$log" | head -1))"
      else
        echo "FAIL: $t on $cpu ($config) did not report the expected backend"; tail -5 "$log"; status=1
      fi
    done
  done
done
exit $status
