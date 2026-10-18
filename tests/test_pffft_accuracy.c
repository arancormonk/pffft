/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 Mark Connor
 */

/*
  Accuracy and contract test for pffft / pffftd.

  Every valid transform size up to a limit is checked against an
  independent reference DFT computed in long double:

    forward:  X[k] = sum_n x[n] exp(-2*pi*i*n*k/N)
    backward: x[n] = sum_k X[k] exp(+2*pi*i*n*k/N)   (unnormalized)

  for PFFFT_REAL and PFFFT_COMPLEX, ordered and unordered, together with
  the identities the API documents: zreorder() round trips, in-place
  operation, work == NULL, the zconvolve family as circular convolution,
  the zero-phase path, and the size helpers.

  Error criterion: the relative L2 error of a backward-stable FFT is
  O(eps * log2 N); a defect (wrong twiddle, layout, sign, scaling) shows
  up as an O(1) error.  Sizes fail above TOL_FFT * eps * log2 N.  Where
  long double is no wider than double (MSVC, Apple arm64), the reference
  for the double build is only as accurate as the code under test, and
  the margin below TOL_FFT is correspondingly smaller -- still orders of
  magnitude below what any defect produces.

  usage: test_pffft_accuracy_{float,double} [max_N]
*/

#ifdef PFFFT_ENABLE_DOUBLE
#include "pffft/pffft_double.h"
typedef double pfscalar;
#define PF_EPS DBL_EPSILON
#define PF(name) pffftd_##name
typedef PFFFTD_Setup PF_Setup;
#define PREC_NAME "double"
#else
#include "pffft/pffft.h"
typedef float pfscalar;
#define PF_EPS FLT_EPSILON
#define PF(name) pffft_##name
typedef PFFFT_Setup PF_Setup;
#define PREC_NAME "float"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

/* precision of the long double reference: IBM double-double (PowerPC) has a
   106-bit significand but reports LDBL_EPSILON as the smallest denormal */
#if LDBL_MANT_DIG == 106
#  define REF_EPS 0x1p-105L
#else
#  define REF_EPS LDBL_EPSILON
#endif

/* accepted relative L2 error, in units of eps * log2(N) */
#define TOL_FFT 2.0
/* default largest N; the suite stays within a few seconds */
#define DEFAULT_MAX_N 8192
/* sizes up to this get the impulse-at-every-position check */
#define IMPULSE_MAX_N 64
#define MAX_REPORTED_FAILURES 50

typedef long double ldouble;
typedef struct { ldouble re, im; } cld;

static int num_failures = 0;

static void report_failure(const char *what, int N, int cplx, double err) {
  ++num_failures;
  if (num_failures <= MAX_REPORTED_FAILURES)
    printf("FAIL %s: %s %s N=%d rel_err=%g (%.2f eps*log2N)\n", PREC_NAME, what,
           cplx ? "COMPLEX" : "REAL", N, err,
           err / (PF_EPS * (N > 1 ? log2((double)N) : 1.0)));
}

/* xorshift64 -- reproducible on every platform */
static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;
static double urand(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static cld cmul(cld a, cld b) {
  cld r;
  r.re = a.re * b.re - a.im * b.im;
  r.im = a.re * b.im + a.im * b.re;
  return r;
}

/* ---------------- reference DFT ---------------- */

static cld *tw_table = NULL;
static int tw_n = 0, tw_sign = 0;

/* tw_table[t] = exp(sign*2*pi*i*t/N), with the angle reduced exactly in
   integer arithmetic so that its magnitude never exceeds pi/4 */
static void make_twiddles(int N, int sign) {
  const ldouble two_pi = 6.283185307179586476925286766559005768L;
  int t;
  if (tw_n == N && tw_sign == sign) return;
  free(tw_table);
  tw_table = (cld*)malloc(sizeof(cld) * N);
  tw_n = N; tw_sign = sign;
  for (t = 0; t < N; ++t) {
    /* angle = 2*pi*t/N = q*pi/2 + 2*pi*r/(4N), with |r| <= N/2 */
    long long t4 = 4LL * t;
    long long q = (t4 + N / 2) / N;      /* nearest quarter turn */
    long long r = t4 - q * N;            /* remainder in [-N/2, N/2] */
    ldouble a = two_pi * (ldouble)r / (ldouble)(4LL * N);
    ldouble c = cosl(a), s = sinl(a), cr, sr;
    switch (q & 3) {
      case 0: cr = c; sr = s; break;
      case 1: cr = -s; sr = c; break;
      case 2: cr = -c; sr = -s; break;
      default: cr = s; sr = -c; break;
    }
    tw_table[t].re = cr;
    tw_table[t].im = sign * sr;
  }
}

/* recursive mixed-radix decimation in time, generic radix butterflies */
static void ref_rec(int n, int N, const cld *in, int istride, cld *out, cld *tmp) {
  int p, m, r, k, q;
  if (n == 1) { out[0] = in[0]; return; }
  for (p = 2; p * p <= n; ++p)
    if (n % p == 0) break;
  if (n % p) p = n;
  m = n / p;
  for (r = 0; r < p; ++r)
    ref_rec(m, N, in + r * istride, istride * p, out + r * m, tmp);
  for (k = 0; k < m; ++k) {
    for (q = 0; q < p; ++q) {
      cld s = { 0, 0 };
      for (r = 0; r < p; ++r) {
        int t = (int)(((long long)r * (k + m * q)) % n) * (N / n);
        cld v = cmul(tw_table[t], out[r * m + k]);
        s.re += v.re; s.im += v.im;
      }
      tmp[k + m * q] = s;
    }
  }
  memcpy(out, tmp, sizeof(cld) * n);
}

static void ref_dft(int N, const cld *in, cld *out, int sign) {
  cld *tmp = (cld*)malloc(sizeof(cld) * N);
  make_twiddles(N, sign);
  ref_rec(N, N, in, 1, out, tmp);
  free(tmp);
}

static void naive_dft(int N, const cld *in, cld *out, int sign) {
  int k, n;
  make_twiddles(N, sign);
  for (k = 0; k < N; ++k) {
    cld s = { 0, 0 };
    for (n = 0; n < N; ++n) {
      cld v = cmul(in[n], tw_table[(int)(((long long)n * k) % N)]);
      s.re += v.re; s.im += v.im;
    }
    out[k] = s;
  }
}

/* circular convolution a (*) b via the reference DFT */
static void ref_cconv(int N, const cld *a, const cld *b, cld *out) {
  cld *A = (cld*)malloc(sizeof(cld) * N), *B = (cld*)malloc(sizeof(cld) * N);
  int k;
  ref_dft(N, a, A, -1);
  ref_dft(N, b, B, -1);
  for (k = 0; k < N; ++k) A[k] = cmul(A[k], B[k]);
  ref_dft(N, A, out, +1);
  for (k = 0; k < N; ++k) { out[k].re /= N; out[k].im /= N; }
  free(A); free(B);
}

/* ---------------- helpers ---------------- */

static double log2N(int N) { return N > 1 ? log2((double)N) : 1.0; }

/* relative L2 error; y holds n complex values interleaved (cplx) or n reals */
static double rel_err(int n, const pfscalar *y, const cld *ref, int cplx) {
  ldouble num = 0, den = 0;
  int k;
  for (k = 0; k < n; ++k) {
    ldouble dr = (cplx ? (ldouble)y[2*k] : (ldouble)y[k]) - ref[k].re;
    ldouble di = (cplx ? (ldouble)y[2*k+1] : 0) - (cplx ? ref[k].im : 0);
    num += dr * dr + di * di;
    den += ref[k].re * ref[k].re + (cplx ? ref[k].im * ref[k].im : 0);
  }
  return (double)(den > 0 ? sqrtl(num / den) : sqrtl(num));
}

static void to_cld(int N, const pfscalar *x, int cplx, cld *out) {
  int k;
  for (k = 0; k < N; ++k) {
    out[k].re = cplx ? (ldouble)x[2*k] : (ldouble)x[k];
    out[k].im = cplx ? (ldouble)x[2*k+1] : 0;
  }
}

/* ordered REAL layout (as complex pairs): slot 0 = F(0) + i*F(N/2) */
static void real_spectrum_to_ordered(int N, const cld *F, cld *out) {
  int k;
  out[0].re = F[0].re; out[0].im = F[N/2].re;
  for (k = 1; k < N/2; ++k) out[k] = F[k];
}

/* ordered REAL layout -> full Hermitian spectrum */
static void ordered_to_real_spectrum(int N, const pfscalar *in, cld *F) {
  int k;
  F[0].re = in[0]; F[0].im = 0;
  F[N/2].re = in[1]; F[N/2].im = 0;
  for (k = 1; k < N/2; ++k) {
    F[k].re = in[2*k]; F[k].im = in[2*k+1];
    F[N-k].re = in[2*k]; F[N-k].im = -(ldouble)in[2*k+1];
  }
}

static void scale_add(int N, cld *acc, const cld *v, ldouble s) {
  int k;
  for (k = 0; k < N; ++k) { acc[k].re += s * v[k].re; acc[k].im += s * v[k].im; }
}

static int same_bits(const pfscalar *a, const pfscalar *b, int n) {
  return memcmp(a, b, sizeof(pfscalar) * n) == 0;
}

static pfscalar *alloc_buf(int n) {
  return (pfscalar*)PF(aligned_malloc)(sizeof(pfscalar) * n);
}

static int is_smooth(int m) {
  while (m % 2 == 0) m /= 2;
  while (m % 3 == 0) m /= 3;
  while (m % 5 == 0) m /= 5;
  return m == 1;
}

/* ---------------- per-size checks ---------------- */

typedef void (*transform_fn)(const PF_Setup*, const pfscalar*, pfscalar*, pfscalar*, pffft_direction_t);

static double worst_fwd = 0, worst_bwd = 0, worst_conv = 0;

static void check_err(const char *what, int N, int cplx, double err, double tol_mult, double *worst) {
  double norm = err / (PF_EPS * log2N(N));
  if (worst && norm > *worst) *worst = norm;
  if (!(norm <= tol_mult * TOL_FFT)) report_failure(what, N, cplx, err);
}

static void test_size(int N, int cplx) {
  PF_Setup *s = PF(new_setup)(N, cplx ? PFFFT_COMPLEX : PFFFT_REAL);
  const int nf = cplx ? 2 * N : N;   /* scalars per buffer */
  const int ns = cplx ? N : N / 2;   /* complex slots in the ordered layout */
  pfscalar *x = alloc_buf(nf), *u = alloc_buf(nf), *y = alloc_buf(nf), *z = alloc_buf(nf);
  pfscalar *w = alloc_buf(nf), *work = alloc_buf(nf);
  pfscalar *a = alloc_buf(nf), *b = alloc_buf(nf), *d = alloc_buf(nf);
  pfscalar *A = alloc_buf(nf), *B = alloc_buf(nf);
  cld *xc = (cld*)malloc(sizeof(cld) * N), *Fc = (cld*)malloc(sizeof(cld) * N);
  cld *ref = (cld*)malloc(sizeof(cld) * N);
  cld *ac = (cld*)malloc(sizeof(cld) * N), *bc = (cld*)malloc(sizeof(cld) * N);
  cld *dc = (cld*)malloc(sizeof(cld) * N), *cv = (cld*)malloc(sizeof(cld) * N);
  int k, j, ord, dir, variant;

  if (!s) {
    report_failure("new_setup() returned NULL for a valid size", N, cplx, 1.0);
    goto done;
  }

  for (k = 0; k < nf; ++k) {
    x[k] = (pfscalar)urand(); u[k] = (pfscalar)urand();
    a[k] = (pfscalar)urand(); b[k] = (pfscalar)urand(); d[k] = (pfscalar)urand();
  }

  /* forward, ordered */
  to_cld(N, x, cplx, xc);
  ref_dft(N, xc, Fc, -1);
  PF(transform_ordered)(s, x, y, work, PFFFT_FORWARD);
  if (cplx) {
    check_err("forward", N, cplx, rel_err(N, y, Fc, 1), 1, &worst_fwd);
  } else {
    real_spectrum_to_ordered(N, Fc, ref);
    check_err("forward", N, cplx, rel_err(ns, y, ref, 1), 1, &worst_fwd);
  }

  /* backward, ordered, from an arbitrary spectrum */
  if (cplx) to_cld(N, u, 1, Fc);
  else ordered_to_real_spectrum(N, u, Fc);
  ref_dft(N, Fc, ref, +1);
  PF(transform_ordered)(s, u, y, work, PFFFT_BACKWARD);
  check_err("backward", N, cplx, rel_err(N, y, ref, cplx), 1, &worst_bwd);

  /* unordered + zreorder must reproduce the ordered transforms bit for bit */
  PF(transform_ordered)(s, x, y, work, PFFFT_FORWARD);
  PF(transform)(s, x, z, work, PFFFT_FORWARD);
  PF(zreorder)(s, z, w, PFFFT_FORWARD);
  if (!same_bits(y, w, nf)) report_failure("zreorder(transform) != transform_ordered (forward)", N, cplx, 1);
  PF(zreorder)(s, w, A, PFFFT_BACKWARD);
  if (!same_bits(A, z, nf)) report_failure("zreorder backward does not invert zreorder forward", N, cplx, 1);
  PF(transform_ordered)(s, u, y, work, PFFFT_BACKWARD);
  PF(zreorder)(s, u, w, PFFFT_BACKWARD);
  PF(transform)(s, w, z, work, PFFFT_BACKWARD);
  if (!same_bits(y, z, nf)) report_failure("transform(zreorder) != transform_ordered (backward)", N, cplx, 1);

  /* in-place, work == NULL, garbage in work, input left untouched */
  for (ord = 0; ord < 2; ++ord) {
    for (dir = 0; dir < 2; ++dir) {
      transform_fn fn = ord ? PF(transform_ordered) : PF(transform);
      pffft_direction_t direction = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
      const pfscalar *src = dir ? u : x;
      fn(s, src, y, work, direction);
      memcpy(z, src, sizeof(pfscalar) * nf);
      fn(s, z, z, work, direction);
      if (!same_bits(y, z, nf)) report_failure(ord ? "in-place transform_ordered" : "in-place transform", N, cplx, 1);
      memcpy(z, src, sizeof(pfscalar) * nf);
      fn(s, z, z, NULL, direction);
      if (!same_bits(y, z, nf)) report_failure("in-place with work == NULL", N, cplx, 1);
      fn(s, src, z, NULL, direction);
      if (!same_bits(y, z, nf)) report_failure("work == NULL", N, cplx, 1);
      for (k = 0; k < nf; ++k) work[k] = (pfscalar)1e30;
      memcpy(w, src, sizeof(pfscalar) * nf);
      fn(s, w, z, work, direction);
      if (!same_bits(y, z, nf)) report_failure("result depends on work contents", N, cplx, 1);
      if (!same_bits(w, src, nf)) report_failure("out-of-place transform modified its input", N, cplx, 1);
    }
  }

  /* backward(forward(x)) == N * x */
  PF(transform)(s, x, z, work, PFFFT_FORWARD);
  PF(transform)(s, z, y, work, PFFFT_BACKWARD);
  to_cld(N, x, cplx, ref);
  for (k = 0; k < N; ++k) { ref[k].re *= N; ref[k].im *= N; }
  check_err("round trip", N, cplx, rel_err(N, y, ref, cplx), 2, NULL);

  /* zconvolve family: backward(D + s*A*B) == N*d + s*N*(a (*) b) */
  to_cld(N, a, cplx, ac);
  to_cld(N, b, cplx, bc);
  to_cld(N, d, cplx, dc);
  ref_cconv(N, ac, bc, cv);
  PF(transform)(s, a, A, work, PFFFT_FORWARD);
  PF(transform)(s, b, B, work, PFFFT_FORWARD);
  for (variant = 0; variant < 7; ++variant) {
    const pfscalar sc = (pfscalar)0.75;
    const cld *base = NULL;  /* signal whose spectrum is accumulated onto */
    const char *what = "";
    ldouble prod_scale = 1;
    switch (variant) {
      case 0: what = "zconvolve_scale";
        PF(zconvolve_scale)(s, A, B, w, sc); prod_scale = sc; break;
      case 1: what = "zconvolve";
        PF(zconvolve)(s, A, B, w); break;
      case 2: what = "zconvolve_accumulate";
        PF(transform)(s, d, w, work, PFFFT_FORWARD);
        PF(zconvolve_accumulate)(s, A, B, w, sc); prod_scale = sc; base = dc; break;
      case 3: what = "zconvolve_scale, ab == a";
        memcpy(w, A, sizeof(pfscalar) * nf);
        PF(zconvolve_scale)(s, w, B, w, sc); prod_scale = sc; break;
      case 4: what = "zconvolve_accumulate, ab == b";
        memcpy(w, B, sizeof(pfscalar) * nf);
        PF(zconvolve_accumulate)(s, A, w, w, sc); prod_scale = sc; base = bc; break;
      case 5: what = "zconvolve, ab == a";
        memcpy(w, A, sizeof(pfscalar) * nf);
        PF(zconvolve)(s, w, B, w); break;
      default: what = "zconvolve_scale, a == b == ab";
        memcpy(w, A, sizeof(pfscalar) * nf);
        PF(zconvolve_scale)(s, w, w, w, sc); prod_scale = sc; break;
    }
    PF(transform)(s, w, y, work, PFFFT_BACKWARD);
    memset(ref, 0, sizeof(cld) * N);
    if (base) scale_add(N, ref, base, (ldouble)N);
    if (variant == 6) {
      cld *sq = (cld*)malloc(sizeof(cld) * N);
      ref_cconv(N, ac, ac, sq);
      scale_add(N, ref, sq, prod_scale * N);
      free(sq);
    } else {
      scale_add(N, ref, cv, prod_scale * N);
    }
    check_err(what, N, cplx, rel_err(N, y, ref, cplx), 3, &worst_conv);
  }

  /* zero-phase path: filter h = b reduced to its even part
     h_e[n] = (h[n] + h[-n]) / 2, applied with gain g */
  if (!cplx) {
    const pfscalar g = (pfscalar)1.5;
    int inplace;
    cld *he = (cld*)malloc(sizeof(cld) * N);
    for (k = 0; k < N; ++k) {
      he[k].re = ((ldouble)b[k] + (ldouble)b[(N - k) % N]) / 2;
      he[k].im = 0;
    }
    ref_cconv(N, ac, he, ref);
    for (k = 0; k < N; ++k) { ref[k].re *= g; ref[k].im = 0; }
    for (inplace = 0; inplace < 2; ++inplace) {
      int r1, r2;
      PF(transform)(s, b, B, work, PFFFT_FORWARD);
      PF(transform)(s, a, A, work, PFFFT_FORWARD);
      if (inplace) {
        r1 = PF(zconvert_zp)(s, B, B, g);
        r2 = PF(zconvolve_zp)(s, A, B, A);
        memcpy(w, A, sizeof(pfscalar) * nf);
      } else {
        r1 = PF(zconvert_zp)(s, B, z, g);
        r2 = PF(zconvolve_zp)(s, A, z, w);
      }
      if (r1 || r2) report_failure("zero-phase functions rejected a REAL setup", N, cplx, 1);
      PF(transform)(s, w, y, work, PFFFT_BACKWARD);
      check_err(inplace ? "zero-phase, in-place" : "zero-phase", N, cplx, rel_err(N, y, ref, 0), 3, &worst_conv);
    }
    free(he);
  } else {
    if (PF(zconvert_zp)(s, A, w, 1) == 0 || PF(zconvolve_zp)(s, A, B, w) == 0)
      report_failure("zero-phase functions accepted a COMPLEX setup", N, cplx, 1);
  }

  /* impulses: every output depends on exactly one input */
  if (N <= IMPULSE_MAX_N) {
    for (j = 0; j < N; ++j) {
      memset(x, 0, sizeof(pfscalar) * nf);
      x[cplx ? 2 * j : j] = 1;
      to_cld(N, x, cplx, xc);
      ref_dft(N, xc, Fc, -1);
      PF(transform_ordered)(s, x, y, work, PFFFT_FORWARD);
      if (cplx) {
        check_err("impulse forward", N, cplx, rel_err(N, y, Fc, 1), 1, NULL);
      } else {
        real_spectrum_to_ordered(N, Fc, ref);
        check_err("impulse forward", N, cplx, rel_err(ns, y, ref, 1), 1, NULL);
      }
    }
  }

done:
  if (s) PF(destroy_setup)(s);
  PF(aligned_free)(x); PF(aligned_free)(u); PF(aligned_free)(y); PF(aligned_free)(z);
  PF(aligned_free)(w); PF(aligned_free)(work); PF(aligned_free)(a); PF(aligned_free)(b);
  PF(aligned_free)(d); PF(aligned_free)(A); PF(aligned_free)(B);
  free(xc); free(Fc); free(ref); free(ac); free(bc); free(dc); free(cv);
}

/* ---------------- size helpers ---------------- */

static void test_size_helpers(void) {
  int cplx, N;
  for (cplx = 0; cplx < 2; ++cplx) {
    pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
    const int simd = PF(simd_size)();
    const int nmin = PF(min_fft_size)(tt);
    if (nmin != (cplx ? simd * simd : 2 * simd * simd))
      report_failure("min_fft_size", nmin, cplx, 1);
    for (N = -3; N <= 20000; ++N) {
      const int valid = N > 0 && N % nmin == 0 && is_smooth(N / nmin);
      if (PF(is_valid_size)(N, tt) != valid)
        report_failure("is_valid_size", N, cplx, 1);
      if (N <= 4096) {
        PF_Setup *s = PF(new_setup)(N, tt);
        if ((s != NULL) != valid)
          report_failure(s ? "new_setup accepted an invalid size" : "new_setup rejected a valid size", N, cplx, 1);
        if (s) PF(destroy_setup)(s);
      }
      if (N >= 1 && N <= 4096) {
        int hi = N, lo = N;
        while (!(hi % nmin == 0 && is_smooth(hi / nmin))) ++hi;
        while (lo > nmin && !(lo % nmin == 0 && is_smooth(lo / nmin))) --lo;
        if (lo < nmin) lo = nmin;
        if (PF(nearest_transform_size)(N, tt, 1) != hi)
          report_failure("nearest_transform_size(higher)", N, cplx, 1);
        if (PF(nearest_transform_size)(N, tt, 0) != lo)
          report_failure("nearest_transform_size(lower)", N, cplx, 1);
      }
    }
  }
  /* the helpers agree with new_setup() above its 2^26 limit, and invalid
     arguments make new_setup() / aligned_malloc() fail without asserting */
  for (cplx = 0; cplx < 2; ++cplx) {
    pffft_transform_t tt = cplx ? PFFFT_COMPLEX : PFFFT_REAL;
    static const int too_big[] = { (1 << 26) + (1 << 20), 3 << 25, 1 << 27, 1 << 30, INT_MAX };
    int i;
    if (!PF(is_valid_size)(1 << 26, tt))
      report_failure("is_valid_size(2^26)", 1 << 26, cplx, 1);
    for (i = 0; i < (int)(sizeof(too_big) / sizeof(too_big[0])); ++i) {
      PF_Setup *s = PF(new_setup)(too_big[i], tt);
      if (PF(is_valid_size)(too_big[i], tt))
        report_failure("is_valid_size accepted N > 2^26", too_big[i], cplx, 1);
      if (s) {
        report_failure("new_setup accepted N > 2^26", too_big[i], cplx, 1);
        PF(destroy_setup)(s);
      }
      if (PF(nearest_transform_size)(too_big[i], tt, 1) != (1 << 26))
        report_failure("nearest_transform_size(higher) above 2^26", too_big[i], cplx, 1);
      if (PF(nearest_transform_size)(too_big[i], tt, 0) != (1 << 26))
        report_failure("nearest_transform_size(lower) above 2^26", too_big[i], cplx, 1);
    }
    if (PF(nearest_transform_size)((1 << 26) - 1, tt, 1) != (1 << 26))
      report_failure("nearest_transform_size(higher) below 2^26", (1 << 26) - 1, cplx, 1);
  }
  for (N = -1; N <= 64; ++N) {
    PF_Setup *s = PF(new_setup)(N, (pffft_transform_t)2);
    if (s) {
      report_failure("new_setup accepted an invalid transform type", N, 0, 1);
      PF(destroy_setup)(s);
    }
  }
  {
    void *p = PF(aligned_malloc)(SIZE_MAX - 8);
    if (p) {
      report_failure("aligned_malloc(SIZE_MAX - 8) did not return NULL", 0, 0, 1);
      PF(aligned_free)(p);
    }
  }

  for (N = 1; N <= (1 << 20); ++N) {
    int p = 1;
    while (p < N) p <<= 1;
    if (PF(next_power_of_two)(N) != p) { report_failure("next_power_of_two", N, 0, 1); break; }
    if (PF(is_power_of_two)(N) != (p == N)) { report_failure("is_power_of_two", N, 0, 1); break; }
  }
}

static int test_reference(void) {
  /* the reference must agree with the defining O(N^2) sum */
  int n, k;
  double worst = 0;
  for (n = 1; n <= 200; ++n) {
    cld *in = (cld*)malloc(sizeof(cld) * n), *o1 = (cld*)malloc(sizeof(cld) * n), *o2 = (cld*)malloc(sizeof(cld) * n);
    ldouble num = 0, den = 0;
    for (k = 0; k < n; ++k) { in[k].re = urand(); in[k].im = urand(); }
    ref_dft(n, in, o1, -1);
    naive_dft(n, in, o2, -1);
    for (k = 0; k < n; ++k) {
      ldouble dr = o1[k].re - o2[k].re, di = o1[k].im - o2[k].im;
      num += dr * dr + di * di;
      den += o2[k].re * o2[k].re + o2[k].im * o2[k].im;
    }
    if (sqrtl(num / den) > worst) worst = (double)sqrtl(num / den);
    free(in); free(o1); free(o2);
  }
  printf("reference DFT self-check: max rel err %.3g (long double epsilon %.3g)\n", worst, (double)REF_EPS);
  return worst <= 64 * REF_EPS * log2(200.0);
}

int main(int argc, char **argv) {
  const int max_n = argc > 1 ? atoi(argv[1]) : DEFAULT_MAX_N;
  int cplx, N, counts[2] = { 0, 0 };

  if (max_n < PF(min_fft_size)(PFFFT_REAL)) {
    printf("usage: %s [max_N >= %d]\n", argv[0], PF(min_fft_size)(PFFFT_REAL));
    return 2;
  }
  printf("pffft accuracy test: %s, simd_size %d (%s), sizes up to %d\n",
         PREC_NAME, PF(simd_size)(), PF(simd_arch)(), max_n);
  if (!test_reference()) {
    printf("FAIL: reference DFT disagrees with the naive DFT\n");
    return 1;
  }
  test_size_helpers();

  for (cplx = 0; cplx < 2; ++cplx) {
    const int nmin = PF(min_fft_size)(cplx ? PFFFT_COMPLEX : PFFFT_REAL);
    for (N = nmin; N <= max_n; N += nmin) {
      if (!is_smooth(N / nmin)) continue;
      test_size(N, cplx);
      ++counts[cplx];
    }
  }
  free(tw_table);

  printf("tested %d REAL and %d COMPLEX sizes\n", counts[0], counts[1]);
  printf("worst rel. L2 error in eps*log2(N): forward %.3f, backward %.3f, convolution %.3f (limit %.1f)\n",
         worst_fwd, worst_bwd, worst_conv, TOL_FFT);
  if (num_failures) {
    printf("%d FAILURES\n", num_failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
