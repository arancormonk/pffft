/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Whether pffft_new_setup() uses the 8-lane AVX2 backend at a size, from the
  CPU class of pffft_x86_wide_class() and, for the inferred class, the L2
  size. Static; pffft.c and the tests include this file.
*/

#ifndef PFFFT_WIDE_SELECT_H
#define PFFFT_WIDE_SELECT_H

#include "pffft_setup.h"
#include "pffft_wide_sizes.h"
#include "pffft_cpu_x86.h"

/* whether the 8-lane backend handles N: a multiple of 64 (complex) or 128
   (real) with N/8 = 2^a 3^b 5^c, at most PFFFT_MAX_N. Baseline ISA: runs
   before the CPU check. */
static int pffft_wide_supports(int N, pffft_transform_t transform) {
  int m;
  if (N <= 0 || N > PFFFT_MAX_N) return 0;
  if (transform == PFFFT_REAL) {
    if (N % 128) return 0;
  } else if (transform == PFFFT_COMPLEX) {
    if (N % 64) return 0;
  } else {
    return 0;
  }
  m = N / 8;
  while (m % 2 == 0) m /= 2;
  while (m % 3 == 0) m /= 3;
  while (m % 5 == 0) m /= 5;
  return m == 1;
}

static int pffft_wide_in_table(const int *sizes, int count, int N) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (sizes[mid] == N) return 1;
    if (sizes[mid] < N) lo = mid + 1; else hi = mid;
  }
  return 0;
}

/* Bytes the inferred class may need at N: input, output, work and the
   setup's twiddles, one buffer each. Keeping them in L2 is a conservative
   rule for CPUs that were not measured: it stays where the instruction
   saving per element dominates, not a claim about cache residency. */
static unsigned long long pffft_wide_working_set(int N, pffft_transform_t transform) {
  return 4ull * (unsigned long long)N * (transform == PFFFT_REAL ? 4u : 8u);
}

/* nonzero if the 8-lane backend is the default at N for a CPU of class cls
   (PFFFT_X86_WIDE_*) with an L2 of l2_bytes (0: unknown) */
static int pffft_wide_default(int N, pffft_transform_t transform, int cls, unsigned long long l2_bytes) {
  const int real = transform == PFFFT_REAL;
  switch (cls) {
  case PFFFT_X86_WIDE_ZEN3:
    return real ? pffft_wide_in_table(pffft_wide_zen3_real_sizes, PFFFT_WIDE_ZEN3_REAL_COUNT, N)
                : pffft_wide_in_table(pffft_wide_zen3_complex_sizes, PFFFT_WIDE_ZEN3_COMPLEX_COUNT, N);
  case PFFFT_X86_WIDE_ZEN3_DESKTOP:
    return real ? pffft_wide_in_table(pffft_wide_zen3_desktop_real_sizes, PFFFT_WIDE_ZEN3_DESKTOP_REAL_COUNT, N)
                : pffft_wide_in_table(pffft_wide_zen3_desktop_complex_sizes, PFFFT_WIDE_ZEN3_DESKTOP_COMPLEX_COUNT, N);
  case PFFFT_X86_WIDE_ZEN3_SERVER:
    return real ? pffft_wide_in_table(pffft_wide_zen3_server_real_sizes, PFFFT_WIDE_ZEN3_SERVER_REAL_COUNT, N)
                : pffft_wide_in_table(pffft_wide_zen3_server_complex_sizes, PFFFT_WIDE_ZEN3_SERVER_COMPLEX_COUNT, N);
  case PFFFT_X86_WIDE_ZEN4:
    return real ? pffft_wide_in_table(pffft_wide_zen4_real_sizes, PFFFT_WIDE_ZEN4_REAL_COUNT, N)
                : pffft_wide_in_table(pffft_wide_zen4_complex_sizes, PFFFT_WIDE_ZEN4_COMPLEX_COUNT, N);
  case PFFFT_X86_WIDE_INFERRED:
    if (!l2_bytes || pffft_wide_working_set(N, transform) > l2_bytes) return 0;
    return real ? pffft_wide_in_table(pffft_wide_inferred_real_sizes, PFFFT_WIDE_INFERRED_REAL_COUNT, N)
                : pffft_wide_in_table(pffft_wide_inferred_complex_sizes, PFFFT_WIDE_INFERRED_COMPLEX_COUNT, N);
  default:
    return 0;
  }
}

#endif /* PFFFT_WIDE_SELECT_H */
