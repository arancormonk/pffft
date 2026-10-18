/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  Checks every pffastconv flag combination against a long double direct
  convolution (or correlation):

    convolution:  y[n] = sum_k h[k] * x[n + L-1 - k]
    correlation:  y[n] = sum_k h[k] * x[n + k]          n = 0 .. inputLen-L

  pffastconv_apply(.., applyFlush = 1) must return inputLen - L + 1 and
  produce exactly these samples; without flush, streamed chunks must
  concatenate to the same result. The DIRECT_* options are exercised
  within their documented conditions for real data, and with complex
  data -- where the header says they are ignored -- outside them.
  The error limit is relative L2 error <= MAX_ERR_EPS * FLT_EPSILON,
  relative to ||h|| * ||x||. new_setup() must reject filterLen < 1.
*/

#include "pffft/pffastconv.h"
#include "pffft/pffft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_ERR_EPS 100.0
#define MAX_REPORTED_FAILURES 30

static int num_failures = 0, num_cases = 0;

static void report(const char *what, int L, int flags, int inLen, double value) {
  ++num_failures;
  if (num_failures <= MAX_REPORTED_FAILURES)
    printf("FAIL %s: filterLen %d flags 0x%x inputLen %d: %g\n", what, L, flags, inLen, value);
}

static unsigned long long rng_state = 88172645463325252ULL;
static float urand(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (float)((double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

/* reference output, ncomp = 1 (real) or 2 (interleaved complex) */
static void reference(const float *x, int inLen, int ncomp, const float *h, int L, int corr, long double *y) {
  int n, k, c;
  for (n = 0; n + L <= inLen; ++n)
    for (c = 0; c < ncomp; ++c) {
      long double acc = 0;
      for (k = 0; k < L; ++k)
        acc += (long double)h[k] * x[ncomp * (corr ? n + k : n + L - 1 - k) + c];
      y[ncomp * n + c] = acc;
    }
}

static double rel_err(const float *out, const long double *ref, int nout, int ncomp,
                      const float *h, int L, const float *x, int inLen) {
  long double e2 = 0, h2 = 0, x2 = 0;
  int i;
  for (i = 0; i < ncomp * nout; ++i) {
    const long double d = (long double)out[i] - ref[i];
    e2 += d * d;
  }
  for (i = 0; i < L; ++i) h2 += (long double)h[i] * h[i];
  for (i = 0; i < ncomp * inLen; ++i) x2 += (long double)x[i] * x[i];
  return (double)(sqrtl(e2 / (h2 * x2 + LDBL_MIN)) / FLT_EPSILON);
}

/* one flushing call on the whole input */
static void test_single_call(const float *h, int L, int flags, int inLen) {
  const int ncomp = (flags & PFFASTCONV_CPLX_INP_OUT) ? 2 : 1;
  int blockLen = 0, ret, expect = inLen - L + 1, cap;
  PFFASTCONV_Setup *s = pffastconv_new_setup(h, L, &blockLen, flags);
  float *x, *y;
  long double *ref;
  ++num_cases;
  if (!s) { report("setup returned NULL", L, flags, inLen, 0); return; }
  if (!(flags & PFFASTCONV_CPLX_INP_OUT) && (flags & (PFFASTCONV_DIRECT_INP | PFFASTCONV_DIRECT_OUT))) {
    /* documented: inputLen <= blockLen, buffers of blockLen, zero padded */
    if (inLen > blockLen) inLen = blockLen;
    expect = inLen - L + 1;
  }
  cap = ncomp * (inLen > blockLen ? inLen : blockLen);
  x = (float*)pffastconv_malloc(sizeof(float) * cap);
  y = (float*)pffastconv_malloc(sizeof(float) * cap);
  ref = (long double*)malloc(sizeof(long double) * (cap + 1));
  memset(x, 0, sizeof(float) * cap);
  { int i; for (i = 0; i < ncomp * inLen; ++i) x[i] = urand(); }
  reference(x, inLen, ncomp, h, L, (flags & PFFASTCONV_CORRELATION) != 0, ref);
  ret = pffastconv_apply(s, x, inLen, y, 1);
  if (ret != expect)
    report("wrong output count", L, flags, inLen, ret);
  else {
    const double e = rel_err(y, ref, ret, ncomp, h, L, x, inLen);
    if (!(e <= MAX_ERR_EPS)) report("output differs from direct convolution (eps)", L, flags, inLen, e);
  }
  pffastconv_free(x); pffastconv_free(y); free(ref);
  pffastconv_destroy_setup(s);
}

/* streaming without flush in uneven chunks, final call flushes */
static void test_streaming(const float *h, int L, int flags, int total) {
  const int ncomp = (flags & PFFASTCONV_CPLX_INP_OUT) ? 2 : 1;
  int blockLen = 0, pending = 0, fed = 0, produced = 0, ret, chunk = 0;
  PFFASTCONV_Setup *s = pffastconv_new_setup(h, L, &blockLen, flags);
  float *x = (float*)malloc(sizeof(float) * ncomp * total);
  float *buf = (float*)malloc(sizeof(float) * ncomp * total);
  float *y = (float*)malloc(sizeof(float) * ncomp * total);
  long double *ref = (long double*)malloc(sizeof(long double) * ncomp * total);
  int i;
  ++num_cases;
  if (!s) { report("setup returned NULL", L, flags, total, 0); free(x); free(buf); free(y); free(ref); return; }
  for (i = 0; i < ncomp * total; ++i) x[i] = urand();
  reference(x, total, ncomp, h, L, (flags & PFFASTCONV_CORRELATION) != 0, ref);
  while (fed < total) {
    int n = (++chunk % 3 == 0) ? 7 : blockLen + 13 * chunk;
    if (n > total - fed) n = total - fed;
    memcpy(buf + ncomp * pending, x + ncomp * fed, sizeof(float) * ncomp * n);
    pending += n; fed += n;
    ret = pffastconv_apply(s, buf, pending, y + ncomp * produced, 0);
    if (ret < 0 || ret > pending) { report("bad streaming output count", L, flags, total, ret); break; }
    produced += ret; pending -= ret;
    memmove(buf, buf + ncomp * ret, sizeof(float) * ncomp * pending);
  }
  produced += pffastconv_apply(s, buf, pending, y + ncomp * produced, 1);
  if (produced != total - L + 1)
    report("streamed output count", L, flags, total, produced);
  else {
    const double e = rel_err(y, ref, produced, ncomp, h, L, x, total);
    if (!(e <= MAX_ERR_EPS)) report("streamed output differs from direct convolution (eps)", L, flags, total, e);
  }
  free(x); free(buf); free(y); free(ref);
  pffastconv_destroy_setup(s);
}

int main(void) {
  static const int filter_lens[] = { 1, 2, 3, 7, 16, 31, 64, 127, 128, 129, 255, 300, 1000 };
  const int base_modes[] = {
    0,
    PFFASTCONV_CPLX_INP_OUT,
    PFFASTCONV_CPLX_INP_OUT | PFFASTCONV_CPLX_SINGLE_FFT
  };
  const int extra_modes[] = {
    0,
    PFFASTCONV_CORRELATION,
    PFFASTCONV_SYMMETRIC,
    PFFASTCONV_DIRECT_INP,
    PFFASTCONV_DIRECT_OUT,
    PFFASTCONV_DIRECT_INP | PFFASTCONV_DIRECT_OUT | PFFASTCONV_CORRELATION
  };
  int fi, bi, ei;
  float h[1000];

  for (fi = -2; fi <= 0; ++fi) {
    int blockLen = 0;
    PFFASTCONV_Setup *s = pffastconv_new_setup(h, fi, &blockLen, 0);
    if (s) {
      report("new_setup accepted an empty filter", fi, 0, 0, 0);
      pffastconv_destroy_setup(s);
    }
  }

  for (fi = 0; fi < (int)(sizeof(filter_lens) / sizeof(filter_lens[0])); ++fi) {
    const int L = filter_lens[fi];
    int k;
    for (k = 0; k < L; ++k) h[k] = urand();
    for (bi = 0; bi < 3; ++bi) {
      for (ei = 0; ei < (int)(sizeof(extra_modes) / sizeof(extra_modes[0])); ++ei) {
        const int flags = base_modes[bi] | extra_modes[ei];
        const int direct = flags & (PFFASTCONV_DIRECT_INP | PFFASTCONV_DIRECT_OUT);
        test_single_call(h, L, flags, L);
        test_single_call(h, L, flags, L + 1);
        test_single_call(h, L, flags, 4 * L + 517);
        if (!direct || (flags & PFFASTCONV_CPLX_INP_OUT))
          test_streaming(h, L, flags, 6 * L + 3001);
      }
    }
  }
  printf("%d cases, %d failures\n", num_cases, num_failures);
  return num_failures ? 1 : 0;
}
