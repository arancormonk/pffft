/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Runs one pffft float operation a fixed number of times inside
  mca_region(), for mca_predict.py: valgrind's callgrind counts the
  instructions executed there (--toggle-collect=mca_region), which gives the
  basic block execution counts per call.

  usage: mca_driver real|complex OP N CALLS
  OP: forward backward ordered-forward ordered-backward inplace-forward
      inplace-backward inplace-ordered-forward inplace-ordered-backward
*/

#include "pffft/pffft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__)
#  define MCA_NOINLINE __attribute__((noinline))
#else
#  define MCA_NOINLINE
#endif

MCA_NOINLINE void mca_region(PFFFT_Setup *s, const float *in, float *out, float *work,
                             int ordered, int backward, int calls) {
  int k;
  for (k = 0; k < calls; ++k) {
    if (ordered) pffft_transform_ordered(s, in, out, work, backward ? PFFFT_BACKWARD : PFFFT_FORWARD);
    else pffft_transform(s, in, out, work, backward ? PFFFT_BACKWARD : PFFFT_FORWARD);
  }
}

int main(int argc, char **argv) {
  const char *op;
  int n, calls, real, inplace, ordered, backward, len, i;
  PFFFT_Setup *s;
  float *a, *b, *w;
  if (argc != 5) {
    fprintf(stderr, "usage: %s real|complex OP N CALLS\n", argv[0]);
    return 2;
  }
  real = strcmp(argv[1], "real") == 0;
  op = argv[2];
  n = atoi(argv[3]);
  calls = atoi(argv[4]);
  inplace = strncmp(op, "inplace-", 8) == 0;
  if (inplace) op += 8;
  ordered = strncmp(op, "ordered-", 8) == 0;
  if (ordered) op += 8;
  backward = strcmp(op, "backward") == 0;
  if (!backward && strcmp(op, "forward") != 0) {
    fprintf(stderr, "unknown operation %s\n", argv[2]);
    return 2;
  }
  s = pffft_new_setup(n, real ? PFFFT_REAL : PFFFT_COMPLEX);
  if (!s) {
    fprintf(stderr, "no setup for N = %d\n", n);
    return 1;
  }
  len = real ? n : 2 * n;
  a = (float*)pffft_aligned_malloc(sizeof(float) * (size_t)len);
  b = (float*)pffft_aligned_malloc(sizeof(float) * (size_t)len);
  w = (float*)pffft_aligned_malloc(sizeof(float) * (size_t)len);
  if (!a || !b || !w) return 1;
  for (i = 0; i < len; ++i) a[i] = (float)((i * 7919) % 1000) / 1000.0f;
  memcpy(b, a, sizeof(float) * (size_t)len);
  /* in place: the same buffer is input and output; the values grow but stay finite */
  mca_region(s, inplace ? b : a, b, w, ordered, backward, calls);
  printf("%d %s %s %d lanes=%d\n", calls, argv[1], argv[2], n, pffft_setup_simd_size(s));
  pffft_aligned_free(a);
  pffft_aligned_free(b);
  pffft_aligned_free(w);
  pffft_destroy_setup(s);
  return 0;
}
