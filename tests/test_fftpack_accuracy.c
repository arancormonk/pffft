/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Mark Connor
 */

/*
  Accuracy test for the FFTPACK port in src/fftpack.c.

  Every transform is compared with its defining sum (netlib FFTPACK
  documentation) evaluated in long double, for all N up to SMALL_MAX_N
  and a few larger sizes with large prime factors, which exercise the
  generic radix passes. Also checked: the documented round trips
  (cfft, rfft: N*x; cosq, sinq: 4N*x; cost: 2(N-1)*x; sint: 2(N+1)*x)
  and that repeating a transform with the same wsave is bit-identical.

  Limits, as relative L2 error in units of eps:
  - FFT_TOL for cfft, rfft, cosq and sinq, which are O(eps log N) for
    composite N and O(eps sqrt(p)) with a prime factor p;
  - COSTSINT_TOL + COSTSINT_TOL_PER_N * N for cost and sint, whose
    pre- and post-processing (inherited from FFTPACK) grows the error
    linearly with N.
  - SMALL_N_TOL for every transform with N <= SMALL_N_MAX, where the
    butterflies' constants dominate the error.
  Before the generic real passes computed their rotation factors
  directly, a real transform of prime length 1009 was off by thousands
  of eps; before the radix-3/5 constants had full double precision,
  sint of n = 2 was off by 7 eps in double.

  Also checked: no init routine writes past the documented wsave size,
  including sizes with more factors than the documented size has room
  for (2 * 3^13 for cffti, rffti, cosqi and sinqi; n + 1 = 2 * 3^11 for
  sinti), and n = 0 returns instead of looping forever.
*/

#include "fftpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

typedef fftpack_real real;
typedef long double ldouble;

#ifdef FFTPACK_DOUBLE_PRECISION
#define REAL_EPS DBL_EPSILON
#define REAL_MANT_DIG DBL_MANT_DIG
#define PREC_NAME "double"
#else
#define REAL_EPS FLT_EPSILON
#define REAL_MANT_DIG FLT_MANT_DIG
#define PREC_NAME "float"
#endif

#define SMALL_MAX_N 128
#define FFT_TOL 32.0
#define COSTSINT_TOL 32.0
#define COSTSINT_TOL_PER_N 0.125
#define SMALL_N_MAX 8
#define SMALL_N_TOL 4.0
#define WSAVE_GUARD 64
/* the reference is no more precise than the code under test where long
   double is no wider than 'real' (MSVC, Apple arm64 with double) */
#define REF_SLACK ((LDBL_MANT_DIG > REAL_MANT_DIG) ? 1.0 : 4.0)

static const ldouble PI_L = 3.141592653589793238462643383279502884L;

enum { F_CFFTF, F_CFFTB, F_RFFTF, F_RFFTB, F_COSQF, F_COSQB, F_SINQF, F_SINQB, F_COST, F_SINT, NUM_F };
static const char *names[NUM_F] = { "cfftf", "cfftb", "rfftf", "rfftb", "cosqf", "cosqb", "sinqf", "sinqb", "cost", "sint" };

/* cos / sin of pi*m/M, with m reduced exactly modulo 2M */
static ldouble *ctab = NULL, *stab = NULL;
static long long tab_m = -1;
static void make_table(long long M) {
  long long r;
  if (M == tab_m) return;
  free(ctab); free(stab);
  ctab = (ldouble*)malloc(sizeof(ldouble) * 2 * M);
  stab = (ldouble*)malloc(sizeof(ldouble) * 2 * M);
  for (r = 0; r < 2 * M; ++r) {
    ctab[r] = cosl(PI_L * (ldouble)r / (ldouble)M);
    stab[r] = sinl(PI_L * (ldouble)r / (ldouble)M);
  }
  tab_m = M;
}
static ldouble cos_pi(long long m, long long M) { make_table(M); m %= 2 * M; return ctab[m < 0 ? m + 2 * M : m]; }
static ldouble sin_pi(long long m, long long M) { make_table(M); m %= 2 * M; return stab[m < 0 ? m + 2 * M : m]; }

static unsigned long long rng_state = 0x123456789abcdefULL;
static double urand(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

/* defining sums, with FFTPACK's 1-based formulas shifted to 0-based arrays */
static void reference(int f, int n, const real *x, ldouble *y) {
  int i, k;
  if (f == F_CFFTF || f == F_CFFTB) {
    const int sg = (f == F_CFFTF) ? -1 : 1;
    for (i = 0; i < n; ++i) {
      ldouble re = 0, im = 0;
      for (k = 0; k < n; ++k) {
        const ldouble c = cos_pi(2LL * i * k, n), s = sg * sin_pi(2LL * i * k, n);
        re += x[2*k] * c - x[2*k+1] * s;
        im += x[2*k] * s + x[2*k+1] * c;
      }
      y[2*i] = re; y[2*i+1] = im;
    }
    return;
  }
  if (f == F_RFFTF) {
    const int last = (n % 2 == 0) ? n / 2 : (n + 1) / 2;
    ldouble s0 = 0;
    for (i = 0; i < n; ++i) s0 += x[i];
    y[0] = s0;
    for (k = 2; k <= last; ++k) {
      ldouble re = 0, im = 0;
      for (i = 0; i < n; ++i) {
        re += x[i] * cos_pi(2LL * (k - 1) * i, n);
        im -= x[i] * sin_pi(2LL * (k - 1) * i, n);
      }
      y[2*k-3] = re; y[2*k-2] = im;
    }
    if (n % 2 == 0) {
      ldouble s = 0;
      for (i = 0; i < n; ++i) s += (i & 1) ? -(ldouble)x[i] : (ldouble)x[i];
      y[n-1] = s;
    }
    return;
  }
  if (f == F_RFFTB) {
    const int last = (n % 2 == 0) ? n / 2 : (n + 1) / 2;
    for (i = 0; i < n; ++i) {
      ldouble s = x[0];
      if (n % 2 == 0) s += (i & 1) ? -(ldouble)x[n-1] : (ldouble)x[n-1];
      for (k = 2; k <= last; ++k)
        s += 2 * (ldouble)x[2*k-3] * cos_pi(2LL * (k - 1) * i, n)
           - 2 * (ldouble)x[2*k-2] * sin_pi(2LL * (k - 1) * i, n);
      y[i] = s;
    }
    return;
  }
  for (i = 1; i <= n; ++i) {
    ldouble s = 0;
    switch (f) {
      case F_COSQF:
        s = x[0];
        for (k = 2; k <= n; ++k) s += 2 * (ldouble)x[k-1] * cos_pi((2LL * i - 1) * (k - 1), 2LL * n);
        break;
      case F_COSQB:
        for (k = 1; k <= n; ++k) s += 4 * (ldouble)x[k-1] * cos_pi((2LL * k - 1) * (i - 1), 2LL * n);
        break;
      case F_SINQF:
        s = ((i - 1) & 1) ? -(ldouble)x[n-1] : (ldouble)x[n-1];
        for (k = 1; k <= n - 1; ++k) s += 2 * (ldouble)x[k-1] * sin_pi((2LL * i - 1) * k, 2LL * n);
        break;
      case F_SINQB:
        for (k = 1; k <= n; ++k) s += 4 * (ldouble)x[k-1] * sin_pi((2LL * k - 1) * i, 2LL * n);
        break;
      case F_COST:
        s = (ldouble)x[0] + (((i - 1) & 1) ? -(ldouble)x[n-1] : (ldouble)x[n-1]);
        for (k = 2; k <= n - 1; ++k) s += 2 * (ldouble)x[k-1] * cos_pi((long long)(k - 1) * (i - 1), n - 1);
        break;
      default: /* F_SINT */
        for (k = 1; k <= n; ++k) s += 2 * (ldouble)x[k-1] * sin_pi((long long)k * i, n + 1);
        break;
    }
    y[i-1] = s;
  }
}

/* wsave sizes as documented in fftpack.h */
static size_t wsave_size(int f, int n) {
  switch (f) {
    case F_CFFTF: case F_CFFTB: return 4 * (size_t)n + 15;
    case F_RFFTF: case F_RFFTB: return 2 * (size_t)n + 15;
    case F_SINT: return (size_t)(2.5 * n + 15);
    default: return 3 * (size_t)n + 15;
  }
}

static void init(int f, int n, real *w) {
  switch (f) {
    case F_CFFTF: case F_CFFTB: cffti(n, w); break;
    case F_RFFTF: case F_RFFTB: rffti(n, w); break;
    case F_COSQF: case F_COSQB: cosqi(n, w); break;
    case F_SINQF: case F_SINQB: sinqi(n, w); break;
    case F_COST: costi(n, w); break;
    default: sinti(n, w); break;
  }
}

static void run(int f, int n, real *x, real *w) {
  switch (f) {
    case F_CFFTF: cfftf(n, x, w); break;
    case F_CFFTB: cfftb(n, x, w); break;
    case F_RFFTF: rfftf(n, x, w); break;
    case F_RFFTB: rfftb(n, x, w); break;
    case F_COSQF: cosqf(n, x, w); break;
    case F_COSQB: cosqb(n, x, w); break;
    case F_SINQF: sinqf(n, x, w); break;
    case F_SINQB: sinqb(n, x, w); break;
    case F_COST: cost(n, x, w); break;
    default: sint(n, x, w); break;
  }
}

/* inverse transform and the round trip's scale factor */
static int inverse_of(int f) {
  switch (f) {
    case F_CFFTF: return F_CFFTB;
    case F_CFFTB: return F_CFFTF;
    case F_RFFTF: return F_RFFTB;
    case F_RFFTB: return F_RFFTF;
    case F_COSQF: return F_COSQB;
    case F_COSQB: return F_COSQF;
    case F_SINQF: return F_SINQB;
    case F_SINQB: return F_SINQF;
    default: return f;
  }
}
static ldouble round_trip_scale(int f, int n) {
  switch (f) {
    case F_CFFTF: case F_CFFTB: case F_RFFTF: case F_RFFTB: return n;
    case F_COST: return 2.0L * (n - 1);
    case F_SINT: return 2.0L * (n + 1);
    default: return 4.0L * n;
  }
}

static double tolerance(int f, int n) {
  const double tol = (n <= SMALL_N_MAX) ? SMALL_N_TOL
                   : (f == F_COST || f == F_SINT) ? COSTSINT_TOL + COSTSINT_TOL_PER_N * n : FFT_TOL;
  return tol * REF_SLACK;
}

/* runs init(f, n) on a wsave of exactly the documented size followed by
   WSAVE_GUARD sentinels; returns the number of failures */
static int check_wsave_bounds(int f, int n) {
  const size_t ws = (n > 0) ? wsave_size(f, n) : 15;
  real *w = (real*)malloc(sizeof(real) * (ws + WSAVE_GUARD));
  size_t j;
  int failures = 0;
  if (!w) {
    printf("FAIL %s %s n=%d: out of memory\n", PREC_NAME, names[f], n);
    return 1;
  }
  for (j = 0; j < ws + WSAVE_GUARD; ++j) w[j] = (real)-12345;
  init(f, n, w);
  for (j = ws; j < ws + WSAVE_GUARD; ++j)
    if (w[j] != (real)-12345) {
      printf("FAIL %s %s n=%d: init wrote wsave[%lu], past the documented %lu\n",
             PREC_NAME, names[f], n, (unsigned long)j, (unsigned long)ws);
      ++failures;
      break;
    }
  free(w);
  return failures;
}

/* round trip of a size whose factors had to be merged to fit wsave */
static int check_large_round_trip(int f, int n) {
  const int len = (f <= F_CFFTB) ? 2 * n : n;
  real *w = (real*)malloc(sizeof(real) * wsave_size(f, n));
  real *x = (real*)malloc(sizeof(real) * len), *x0 = (real*)malloc(sizeof(real) * len);
  ldouble num = 0, den = 0, sc = round_trip_scale(f, n);
  double rt;
  int j, failures = 0;
  if (!w || !x || !x0) {
    printf("FAIL %s %s n=%d: out of memory\n", PREC_NAME, names[f], n);
    free(w); free(x); free(x0);
    return 1;
  }
  init(f, n, w);
  for (j = 0; j < len; ++j) x0[j] = x[j] = (real)urand();
  run(f, n, x, w);
  run(inverse_of(f), n, x, w);
  for (j = 0; j < len; ++j) {
    const ldouble d = (ldouble)x[j] / sc - x0[j];
    num += d * d;
    den += (ldouble)x0[j] * x0[j];
  }
  rt = (double)sqrtl(num / den) / REAL_EPS;
  if (!(rt <= 2 * FFT_TOL * REF_SLACK)) {
    printf("FAIL %s %s n=%d: round trip error %.1f eps > %.1f\n", PREC_NAME, names[f], n, rt, 2 * FFT_TOL * REF_SLACK);
    ++failures;
  }
  free(w); free(x); free(x0);
  return failures;
}

int main(void) {
  /* large prime factors for each transform family: 1009 and 2003 are
     prime; cost runs on n-1 = 1009, sint on n+1 = 2 * 1009 */
  static const int large[] = { 509, 1009, 1010, 1024, 1536, 2003, 2017 };
  int sizes[SMALL_MAX_N + sizeof(large) / sizeof(large[0])];
  int num_sizes = 0, i, f, failures = 0;
  double worst[NUM_F] = { 0 };
  int worst_n[NUM_F] = { 0 };

  for (i = 1; i <= SMALL_MAX_N; ++i) sizes[num_sizes++] = i;
  for (i = 0; i < (int)(sizeof(large) / sizeof(large[0])); ++i) sizes[num_sizes++] = large[i];

  for (f = 0; f < NUM_F; ++f) {
    for (i = 0; i < num_sizes; ++i) {
      const int n = sizes[i];
      const int len = (f <= F_CFFTB) ? 2 * n : n;
      const size_t ws = wsave_size(f, n);
      real *w = (real*)malloc(sizeof(real) * ws);
      real *x = (real*)malloc(sizeof(real) * len), *x0 = (real*)malloc(sizeof(real) * len);
      real *x2 = (real*)malloc(sizeof(real) * len);
      ldouble *y = (ldouble*)malloc(sizeof(ldouble) * len);
      ldouble num = 0, den = 0, sc;
      double err, rt;
      int j;

      if (f == F_COST && n < 2) { free(w); free(x); free(x0); free(x2); free(y); continue; }  /* documented: n > 1 */

      init(f, n, w);
      for (j = 0; j < len; ++j) x0[j] = x[j] = (real)urand();
      reference(f, n, x0, y);
      run(f, n, x, w);

      memcpy(x2, x0, sizeof(real) * len);
      run(f, n, x2, w);
      if (memcmp(x2, x, sizeof(real) * len)) {
        printf("FAIL %s %s n=%d: repeated transform differs\n", PREC_NAME, names[f], n);
        ++failures;
      }

      for (j = 0; j < len; ++j) {
        const ldouble d = (ldouble)x[j] - y[j];
        num += d * d;
        den += y[j] * y[j];
      }
      err = (double)sqrtl(num / (den > 0 ? den : 1)) / REAL_EPS;
      if (err > worst[f]) { worst[f] = err; worst_n[f] = n; }
      if (!(err <= tolerance(f, n))) {
        printf("FAIL %s %s n=%d: rel. error %.1f eps > %.1f\n", PREC_NAME, names[f], n, err, tolerance(f, n));
        ++failures;
      }

      run(inverse_of(f), n, x, w);
      sc = round_trip_scale(f, n);
      num = den = 0;
      for (j = 0; j < len; ++j) {
        const ldouble d = (ldouble)x[j] / sc - x0[j];
        num += d * d;
        den += (ldouble)x0[j] * x0[j];
      }
      rt = (double)sqrtl(num / den) / REAL_EPS;
      if (!(rt <= 2 * tolerance(f, n))) {
        printf("FAIL %s %s n=%d: round trip error %.1f eps > %.1f\n", PREC_NAME, names[f], n, rt, 2 * tolerance(f, n));
        ++failures;
      }
      free(w); free(x); free(x0); free(x2); free(y);
    }
  }
  free(ctab); free(stab);

  for (f = 0; f < NUM_F; f += (f < F_COST) ? 2 : 1) {
    for (i = 1; i <= 300; ++i)
      failures += check_wsave_bounds(f, i);
    failures += check_wsave_bounds(f, 0);
    failures += check_wsave_bounds(f, (f == F_SINT) ? 354293 : 3188646);
  }
  failures += check_large_round_trip(F_CFFTF, 3188646);
  failures += check_large_round_trip(F_RFFTF, 3188646);

  for (f = 0; f < NUM_F; ++f)
    printf("%s %-5s worst rel. error %8.2f eps at n=%d\n", PREC_NAME, names[f], worst[f], worst_n[f]);
  if (failures) {
    printf("%d FAILURES\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
