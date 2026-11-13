/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 */

/*
  White-box tests of the pass codelets, their drivers and the plan setup
  (src/pffft_passes_impl.h), and of the compile-time switches of
  src/pffft_pass_config.h.

  This file includes the library source, pffft.c (or pffft_double.c with
  PFFFT_ENABLE_DOUBLE, after which `float` means `double`, as in the
  library), so that its static functions can be called. CMake compiles it
  with the PFFFT target's definitions, options and include directories and
  does not link the library.

  The tests that call codelets are in test_pffft_passes_impl.h, included
  once per backend:

    - the native backend of this build (4 lanes, or 1 for the scalar
      build), always;
    - the 8-lane AVX2 float backend, when it is built into pffft.c's
      translation unit (PFFFT_WIDE, GCC and Clang). Its tests run only when
      the CPU can execute it. With MSVC the backend is a translation unit of
      its own (pffft_wide_tu.c, PFFFT_WIDE_SEPARATE); its codelets are not
      visible here, and it is covered through the public API only.

  Without the 8-lane tests (backend not in this translation unit, or a CPU
  that cannot run it) the float test skips them, unless the environment
  has PFFFT_TEST_REQUIRE_WIDE=1 (CMake option PFFFT_TEST_REQUIRE_WIDE):
  then it fails. The round trips of the public API at the end of this file
  run on every backend the build has, the misaligned in-place test on the
  8-lane one.

  Test code indexes buffers as float arrays (T_LANES floats per vector)
  and does not use VLD/VST/VADD or the other vector macros: after the
  8-lane inclusion those are the 256-bit ones. Every buffer comes from
  pffft_aligned_malloc() (pffftd_aligned_malloc()) and is freed; nothing
  large is static or on the stack.

  Prints one line per failing check and exits with a non-zero status if
  any check failed.
*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 1 for a switch of pffft_pass_config.h that the build defines (CMake
   option ON or OFF, the compiler flags, a parent project's definitions),
   0 for one that gets its default from the header; test_config() checks
   the default of the latter only */
#ifdef PFFFT_TERMINAL_RADIX8
#  define T_BUILD_TERMINAL_RADIX8 1
#else
#  define T_BUILD_TERMINAL_RADIX8 0
#endif
#ifdef PFFFT_INPLACE_FIRST_PASS
#  define T_BUILD_INPLACE_FIRST_PASS 1
#else
#  define T_BUILD_INPLACE_FIRST_PASS 0
#endif

#ifdef PFFFT_ENABLE_DOUBLE
#  include "pffft_double.c"
#  define T_EPS DBL_EPSILON
#  define T_ALIGNED_MALLOC pffftd_aligned_malloc
#  define T_ALIGNED_FREE pffftd_aligned_free
#else
#  include "pffft.c"
#  define T_EPS FLT_EPSILON
#  define T_ALIGNED_MALLOC pffft_aligned_malloc
#  define T_ALIGNED_FREE pffft_aligned_free
#endif

/* 1 and a report line when cond is false, else 0: fails += T_CHECK(...) */
#define T_CHECK(cond) t_check((cond) != 0, #cond, __FILE__, __LINE__)

static int t_check(int ok, const char *what, const char *file, int line) {
  if (ok) return 0;
  printf("FAIL %s:%d: %s\n", file, line, what);
  return 1;
}

/* n floats (doubles in the double build), aligned for every backend; NULL
   when out of memory */
static float *t_alloc(size_t n) {
  return (float *)T_ALIGNED_MALLOC(n * sizeof(float));
}

static void t_free(float *p) {
  if (p) T_ALIGNED_FREE(p);
}

/* the larger of two errors; a NaN or an infinity in either is kept (e > err
   alone would drop a NaN e), so that the maximum is not within a tolerance */
static double t_max_err(double err, double e) {
  if (!isfinite(err)) return err;
  return !(e <= err) ? e : err;
}

/* an error within its tolerance: finite and at most limit */
static int t_within(double err, double limit) {
  return isfinite(err) && err <= limit;
}

/* An infinity and a NaN made at run time, for test_tolerance(). NAN and
   INFINITY are constant expressions that overflow in MSVC's <math.h>;
   folded into t_max_err() and t_within(), they stop the build with warning
   C4756 (overflow in constant arithmetic) as an error. A volatile value
   cannot be folded: DBL_MAX times 2 is an infinity, and an infinity minus
   itself is a NaN (no division by zero, which is warning C4723). */
static double t_inf(void) {
  volatile double big = DBL_MAX;
  return big * 2.0;
}

static double t_nan(void) {
  volatile double inf = t_inf();
  return inf - inf;
}

/* t_max_err() and t_within(): an error that is NaN or infinite, anywhere
   in a maximum, is never within a tolerance */
static int test_tolerance(void) {
  const double nan_v = t_nan(), inf_v = t_inf();
  int fails = 0;
  fails += T_CHECK(isnan(nan_v) && isinf(inf_v) && inf_v > 0);
  fails += T_CHECK(!t_within(t_max_err(0.0, nan_v), 1.0));
  fails += T_CHECK(!t_within(t_max_err(t_max_err(0.0, nan_v), 0.5), 1.0));
  fails += T_CHECK(!t_within(t_max_err(t_max_err(0.0, inf_v), 0.5), 1.0));
  fails += T_CHECK(!t_within(nan_v, 1.0) && !t_within(inf_v, 1.0));
  fails += T_CHECK(t_within(t_max_err(0.25, 0.5), 1.0) && !t_within(t_max_err(0.25, 2.0), 1.0));
  return fails;
}

/* n reproducible pseudo-random values in [-1, 1) */
static void t_fill(float *p, size_t n) {
  static unsigned long state = 1;
  size_t i;
  for (i = 0; i < n; ++i) {
    state = (state * 1103515245UL + 12345UL) & 0xffffffffUL;
    p[i] = (float)((double)(state >> 8) / 8388608.0 - 1.0);
  }
}

/* n == 2^a 3^b 5^c */
static int t_is_235(int n) {
  while (n % 2 == 0) n /= 2;
  while (n % 3 == 0) n /= 3;
  while (n % 5 == 0) n /= 5;
  return n == 1;
}

#if PFFFT_TERMINAL_RADIX8
/* n == 4^a 3^b 5^c */
static int t_is_435(int n) {
  while (n % 4 == 0) n /= 4;
  while (n % 3 == 0) n /= 3;
  while (n % 5 == 0) n /= 5;
  return n == 1;
}
#endif

#ifndef PFFFT_ENABLE_DOUBLE
/* PFFFT_TEST_REQUIRE_WIDE=1 in the environment (CMake option
   PFFFT_TEST_REQUIRE_WIDE, float test only): the 8-lane backend must be
   tested, a build or CPU without it fails instead of skipping */
static int t_require_wide(void) {
  const char *v = getenv("PFFFT_TEST_REQUIRE_WIDE");
  return v != NULL && strcmp(v, "1") == 0;
}
#endif

/*
  The backend sections of test_pffft_passes_impl.h. Each inclusion needs:
    T_LANES      floats per vector
    T_V          the vector type
    T_FN(name)   the static function `name` of that backend
    T_FUSED      1 when the backend has the fused layout passes
                 (PF_FUSED_LAYOUT: passf*_il_ps, passf4_ol_ps, ...)
    T_NAME(name) the name of a test function for that backend
  and undefines them at its end.
*/
#define T_LANES SIMD_SZ
#define T_V v4sf
#define T_FN(name) name
#define T_FUSED (SIMD_SZ == 4)
#define T_NAME(name) name##_native
#include "test_pffft_passes_impl.h"

#if defined(PFFFT_WIDE) && PFFFT_WIDE && !defined(PFFFT_WIDE_SEPARATE)
#  define T_TEST_W8 1
#  define T_LANES 8
#  define T_V pffft_v8sf
#  define T_FN(name) pffft_w8_##name
#  define T_FUSED 1
#  define T_NAME(name) name##_w8
#  include "test_pffft_passes_impl.h"
#else
#  define T_TEST_W8 0
#endif

/* the switches are 0 or 1; the defaults, for a switch the build leaves to
   the header: both on, except the terminal radix-8 pass on the
   single-lane scalar build (SIMD_SZ == 1), where it is off */
static int test_config(void) {
  int fails = 0;
  printf("terminal radix 8: %d, in-place first pass: %d, backend %s x%d\n",
         PFFFT_TERMINAL_RADIX8, PFFFT_INPLACE_FIRST_PASS, VARCH, SIMD_SZ);
  fails += T_CHECK(PFFFT_TERMINAL_RADIX8 == 0 || PFFFT_TERMINAL_RADIX8 == 1);
  fails += T_CHECK(PFFFT_INPLACE_FIRST_PASS == 0 || PFFFT_INPLACE_FIRST_PASS == 1);
  fails += T_CHECK(T_BUILD_TERMINAL_RADIX8 || PFFFT_TERMINAL_RADIX8 == (SIMD_SZ == 1 ? 0 : 1));
  fails += T_CHECK(T_BUILD_INPLACE_FIRST_PASS || PFFFT_INPLACE_FIRST_PASS == 1);
  return fails;
}

/* the public API, each backend */
#ifdef PFFFT_ENABLE_DOUBLE
#  define T_SETUP PFFFTD_Setup
#  define T_DESTROY pffftd_destroy_setup
#  define T_TRANSFORM pffftd_transform
#  define T_TRANSFORM_ORDERED pffftd_transform_ordered
#else
#  define T_SETUP PFFFT_Setup
#  define T_DESTROY pffft_destroy_setup
#  define T_TRANSFORM pffft_transform
#  define T_TRANSFORM_ORDERED pffft_transform_ordered
#endif

/* forward then backward, and in place against out of place, for setup s */
static int t_round_trip(const T_SETUP *s, int N, pffft_transform_t t, const char *backend) {
  const int len = t == PFFFT_COMPLEX ? 2*N : N;
  const char *what = t == PFFFT_COMPLEX ? "complex" : "real";
  float *x = t_alloc(len), *y = t_alloc(len), *z = t_alloc(len);
  float *ref = t_alloc(len), *buf = t_alloc(len), *work = t_alloc(len);
  int fails = 0, i, ordered, dir;
  double num = 0, den = 0, err, limit = 4 * T_EPS * (log((double)N) / log(2.0));
  if (!x || !y || !z || !ref || !buf || !work) {
    fails += T_CHECK(x && y && z && ref && buf && work);
    goto done;
  }
  t_fill(x, len);
  T_TRANSFORM(s, x, y, work, PFFFT_FORWARD);
  T_TRANSFORM(s, y, z, work, PFFFT_BACKWARD);
  for (i = 0; i < len; ++i) {
    double d = (double)z[i] / N - (double)x[i];
    num += d * d;
    den += (double)x[i] * (double)x[i];
  }
  err = sqrt(num / den);
  if (!t_within(err, limit))
    printf("  (%s %s N = %d: round-trip error %g > %g)\n", backend, what, N, err, limit);
  fails += T_CHECK(t_within(err, limit));
  for (ordered = 0; ordered < 2; ++ordered) {
    for (dir = 0; dir < 2; ++dir) {
      const pffft_direction_t d = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
      const float *in = dir ? y : x;
      int same;
      memcpy(buf, in, len * sizeof(float));
      if (ordered) {
        T_TRANSFORM_ORDERED(s, in, ref, work, d);
        T_TRANSFORM_ORDERED(s, buf, buf, work, d);
      } else {
        T_TRANSFORM(s, in, ref, work, d);
        T_TRANSFORM(s, buf, buf, work, d);
      }
      same = memcmp(buf, ref, len * sizeof(float)) == 0;
      if (!same)
        printf("  (%s %s N = %d, %s %s: in place differs from out of place)\n", backend, what, N,
               ordered ? "ordered" : "unordered", dir ? "backward" : "forward");
      fails += T_CHECK(same);
    }
  }
done:
  t_free(x); t_free(y); t_free(z); t_free(ref); t_free(buf); t_free(work);
  return fails;
}

/* sizes above the range of the accuracy test, powers of two and mixed radix */
static int test_public_round_trip(void) {
  static const int sizes[] = { 1 << 20, 1 << 21, 921600 };  /* 921600 = 2^12 3^2 5^2 */
  int fails = 0, is, it;
#ifndef PFFFT_ENABLE_DOUBLE
  int w8_missing = 0;
#endif
  for (is = 0; is < (int)(sizeof(sizes)/sizeof(sizes[0])); ++is) {
    for (it = 0; it < 2; ++it) {
      const int N = sizes[is];
      const pffft_transform_t t = it ? PFFFT_REAL : PFFFT_COMPLEX;
      T_SETUP *s;
#ifdef PFFFT_ENABLE_DOUBLE
      s = pffftd_new_setup(N, t);
      fails += T_CHECK(s != NULL);
      if (s) {
        fails += t_round_trip(s, N, t, "native");
        T_DESTROY(s);
      }
#else
      s = pffft_internal_new_setup(N, t, PFFFT_BACKEND_NATIVE);
      fails += T_CHECK(s != NULL);
      if (s) {
        fails += t_round_trip(s, N, t, "native");
        T_DESTROY(s);
      }
      /* NULL: the 8-lane backend is not in this build or not supported by
         this CPU */
      s = pffft_internal_new_setup(N, t, PFFFT_BACKEND_W8);
      if (s) {
        fails += t_round_trip(s, N, t, "8-lane");
        T_DESTROY(s);
      } else {
        ++w8_missing;
      }
#endif
    }
  }
#ifndef PFFFT_ENABLE_DOUBLE
  if (w8_missing && t_require_wide()) {
    printf("FAIL: public round trip: no 8-lane setup, which PFFFT_TEST_REQUIRE_WIDE=1 requires\n");
    ++fails;
  } else if (w8_missing) {
    printf("8-lane backend: not available, its public round trips are skipped\n");
  }
#endif
  return fails;
}

#if !defined(PFFFT_ENABLE_DOUBLE) && defined(PFFFT_WIDE) && PFFFT_WIDE
/* 8-lane complex transforms in place at even pass counts (N / 8 = 24, 40,
   192, 512: plans 3,8 / 5,8 / 3,4,4,4 / 4,4,4,8 with the default
   switches), with work == NULL and the buffer 16 bytes past a 64-byte
   boundary, the least alignment the API allows: the bits of the aligned
   out-of-place transform with a work buffer */
static int test_public_inplace_misaligned(void) {
  static const int sizes[] = { 192, 320, 1536, 4096 };
  int fails = 0, is;
  for (is = 0; is < (int)(sizeof(sizes)/sizeof(sizes[0])); ++is) {
    const int N = sizes[is], len = 2*N;
    PFFFT_Setup *s = pffft_internal_new_setup(N, PFFFT_COMPLEX, PFFFT_BACKEND_W8);
    float *x, *ref, *work, *base, *buf;
    int ordered, dir;
    if (!s) continue;   /* the CPU cannot run the 8-lane backend */
    x = t_alloc(len); ref = t_alloc(len); work = t_alloc(len);
    base = t_alloc(len + 16/sizeof(float));
    if (!x || !ref || !work || !base) {
      fails += T_CHECK(x && ref && work && base);
      goto next;
    }
    buf = (float *)((char *)base + 16);
    t_fill(x, len);
    for (ordered = 0; ordered < 2; ++ordered) {
      for (dir = 0; dir < 2; ++dir) {
        const pffft_direction_t d = dir ? PFFFT_BACKWARD : PFFFT_FORWARD;
        int same;
        memcpy(buf, x, len * sizeof(float));
        if (ordered) {
          pffft_transform_ordered(s, x, ref, work, d);
          pffft_transform_ordered(s, buf, buf, NULL, d);
        } else {
          pffft_transform(s, x, ref, work, d);
          pffft_transform(s, buf, buf, NULL, d);
        }
        same = memcmp(buf, ref, len * sizeof(float)) == 0;
        if (!same)
          printf("  (8-lane complex N = %d, %s %s: in place at 16 mod 64 bytes differs)\n", N,
                 ordered ? "ordered" : "unordered", dir ? "backward" : "forward");
        fails += T_CHECK(same);
      }
    }
next:
    t_free(x); t_free(ref); t_free(work); t_free(base);
    pffft_destroy_setup(s);
  }
  return fails;
}
#endif

int main(void) {
  int fails = 0;
  fails += test_config();
  fails += test_tolerance();
  fails += run_backend_tests_native();
#if T_TEST_W8
  {
    unsigned long long l2;
    if (pffft_wide_cpu_class(&l2) >= 0) {
      fails += run_backend_tests_w8();
    } else if (t_require_wide()) {
      printf("FAIL: the 8-lane backend is not supported by this CPU, and PFFFT_TEST_REQUIRE_WIDE=1 requires its tests\n");
      ++fails;
    } else {
      printf("8-lane backend: not supported by this CPU, its tests are skipped\n");
    }
  }
#elif !defined(PFFFT_ENABLE_DOUBLE) && !defined(PFFFT_WIDE_SEPARATE)
  /* (with PFFFT_WIDE_SEPARATE the codelets are in another translation
     unit; the public round trips cover the backend) */
  if (t_require_wide()) {
    printf("FAIL: the 8-lane backend is not in this build, and PFFFT_TEST_REQUIRE_WIDE=1 requires its tests\n");
    ++fails;
  }
#endif
  fails += test_public_round_trip();
#if !defined(PFFFT_ENABLE_DOUBLE) && defined(PFFFT_WIDE) && PFFFT_WIDE
  fails += test_public_inplace_misaligned();
#endif
  if (fails) {
    printf("%d check(s) failed\n", fails);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
