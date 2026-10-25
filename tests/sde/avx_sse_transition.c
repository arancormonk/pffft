/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/* Negative control for tests/run_sde_checks.sh (GCC, Clang): dirties the
   upper ymm halves and then executes a legacy-SSE instruction, 1000 times,
   which Intel SDE's AVX/SSE transition checker (-ast) must report. */

#include <stdio.h>

int main(void) {
  int k;
  for (k = 0; k < 1000; ++k)
    __asm__ volatile("vxorps %%ymm1, %%ymm1, %%ymm1\n\taddss %%xmm2, %%xmm2" ::: "xmm1", "xmm2");
  printf("done\n");
  return 0;
}
