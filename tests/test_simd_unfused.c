/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#define ALWAYS_INLINE(type) inline type
#include "simd/pf_float.h"

#ifdef PFFFT_X86_HAVE_FMA
#error PFFFT_DISABLE_FMA must disable explicit x86 FMA
#endif

int main(void) {
  volatile float a = 0x1.000002p0f, b = 0x1.fffffcp-1f, c = -1.0f;
  float aa = a, bb = b, cc = c;
  v4sf_union result;
  int i;
  if (FLT_EVAL_METHOD != 0) {
    /* x87 math (e.g. i686 without SSE) keeps a*b in extended precision,
       so no float operation is rounded on its own: nothing to test */
    printf("skipped: FLT_EVAL_METHOD == %d\n", (int)FLT_EVAL_METHOD);
    return 77;
  }
  /* The product rounds to one before the subtraction; FMA yields -2^-46. */
  result.v = VMADD(LD_PS1(aa), LD_PS1(bb), LD_PS1(cc));
  for (i = 0; i < SIMD_SZ; ++i) assert(fabsf(result.f[i]) < 0x1p-47f);
  cc = 1.0f;
  result.v = VMSUB(LD_PS1(aa), LD_PS1(bb), LD_PS1(cc));
  for (i = 0; i < SIMD_SZ; ++i) assert(fabsf(result.f[i]) < 0x1p-47f);
  return 0;
}
