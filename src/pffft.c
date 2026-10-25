/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB
   (http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
   of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.  

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.


   PFFFT : a Pretty Fast FFT.

   This file is largerly based on the original FFTPACK implementation, modified in
   order to take advantage of SIMD instructions of modern CPUs.
*/

/*
  ChangeLog: 
  - 2011/10/02, version 1: This is the very first release of this file.
*/

#include "pffft.h"

/* detect compiler flavour */
#if defined(_MSC_VER)
#  define COMPILER_MSVC
#elif defined(__GNUC__)
#  define COMPILER_GCC
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#if defined(COMPILER_GCC)
#  define ALWAYS_INLINE(return_type) inline return_type __attribute__ ((always_inline))
#  define NEVER_INLINE(return_type) return_type __attribute__ ((noinline))
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ varname__[size__];
#elif defined(COMPILER_MSVC)
#  define ALWAYS_INLINE(return_type) __forceinline return_type
#  define NEVER_INLINE(return_type) __declspec(noinline) return_type
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ *varname__ = (type__*)_alloca(size__ * sizeof(type__))
#endif


#ifdef COMPILER_MSVC
#pragma warning( disable : 4244 4305 4204 4456 )
#endif

/* 
   vector support macros: the rest of the code is independant of
   SSE/Altivec/NEON -- adding support for other platforms with 4-element
   vectors should be limited to these macros 
*/
#include "simd/pf_float.h"
#include "pffft_internal.h"

/*
  On x86-64 the 4-lane SSE implementation can be complemented by the 8-lane
  AVX2 backend of pffft_wide_impl.h, chosen per setup at runtime. It needs
  GCC or Clang function target pragmas. Not on Windows: GCC there cannot
  realign the stack for spilled 256-bit vectors, and MSVC has no per-function
  target selection.
*/
#if !defined(PFFFT_DISABLE_WIDE) && defined(PFFFT_SSE_FLOAT_BACKEND) && defined(__x86_64__) \
    && !defined(_WIN32) && !defined(__CYGWIN__) \
    && ((defined(__clang__) && __clang_major__ >= 10) \
        || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 5))
#  define PFFFT_WIDE 1
#else
#  define PFFFT_WIDE 0
#endif

/* have code comparable with this definition */
#define SETUP_STRUCT               PFFFT_Setup
#if PFFFT_WIDE
/* the public functions below dispatch to these or to the 8-lane backend */
#define FUNC_NEW_SETUP             pffft_native_new_setup
#define FUNC_DESTROY               pffft_native_destroy_setup
#define FUNC_TRANSFORM_UNORDRD     pffft_native_transform
#define FUNC_TRANSFORM_ORDERED     pffft_native_transform_ordered
#define FUNC_ZREORDER              pffft_native_zreorder
#define FUNC_ZCONVOLVE_ACCUMULATE  pffft_native_zconvolve_accumulate
#define FUNC_ZCONVOLVE              pffft_native_zconvolve
#define FUNC_ZCONVERT_ZP            pffft_native_zconvert_zp
#define FUNC_ZCONVOLVE_ZP           pffft_native_zconvolve_zp
#define FUNC_ZCONVOLVE_SCALE       pffft_native_zconvolve_scale
PFFFT_Setup *pffft_native_new_setup(int N, pffft_transform_t transform);
void pffft_native_destroy_setup(PFFFT_Setup *setup);
void pffft_native_transform(const PFFFT_Setup *setup, const float *input, float *output, float *work, pffft_direction_t direction);
void pffft_native_transform_ordered(const PFFFT_Setup *setup, const float *input, float *output, float *work, pffft_direction_t direction);
void pffft_native_zreorder(const PFFFT_Setup *setup, const float *input, float *output, pffft_direction_t direction);
void pffft_native_zconvolve_accumulate(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab, float scaling);
void pffft_native_zconvolve_scale(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab, float scaling);
void pffft_native_zconvolve(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab);
int pffft_native_zconvert_zp(const PFFFT_Setup *setup, const float *in, float *out, float scaling);
int pffft_native_zconvolve_zp(const PFFFT_Setup *setup, const float *dft_x, const float *dft_hzp, float *dft_ab);
#else
PFFFT_Setup *pffft_native_new_setup(int N, pffft_transform_t transform);
#define FUNC_NEW_SETUP             pffft_native_new_setup
#define FUNC_DESTROY               pffft_destroy_setup
#define FUNC_TRANSFORM_UNORDRD     pffft_transform
#define FUNC_TRANSFORM_ORDERED     pffft_transform_ordered
#define FUNC_ZREORDER              pffft_zreorder
#define FUNC_ZCONVOLVE_ACCUMULATE  pffft_zconvolve_accumulate
#define FUNC_ZCONVOLVE              pffft_zconvolve
#define FUNC_ZCONVERT_ZP            pffft_zconvert_zp
#define FUNC_ZCONVOLVE_ZP           pffft_zconvolve_zp
#define FUNC_ZCONVOLVE_SCALE       pffft_zconvolve_scale
#endif

#define FUNC_ALIGNED_MALLOC        pffft_aligned_malloc
#define FUNC_ALIGNED_FREE          pffft_aligned_free
#define FUNC_SIMD_SIZE             pffft_simd_size
#define FUNC_MIN_FFT_SIZE          pffft_min_fft_size
#define FUNC_IS_VALID_SIZE         pffft_is_valid_size
#define FUNC_NEAREST_SIZE          pffft_nearest_transform_size
#define FUNC_SIMD_ARCH             pffft_simd_arch
#define FUNC_VALIDATE_SIMD_A       validate_pffft_simd
#define FUNC_VALIDATE_SIMD_EX      validate_pffft_simd_ex

#define FUNC_CPLX_FINALIZE         pffft_cplx_finalize
#define FUNC_CPLX_PREPROCESS       pffft_cplx_preprocess
#define FUNC_REAL_PREPROCESS_4X4   pffft_real_preprocess_4x4
#define FUNC_REAL_PREPROCESS       pffft_real_preprocess
#define FUNC_REAL_FINALIZE_4X4     pffft_real_finalize_4x4
#define FUNC_REAL_FINALIZE         pffft_real_finalize
#define FUNC_TRANSFORM_INTERNAL    pffft_transform_internal



#include "pffft_priv_impl.h"

int pffft_setup_simd_size(const PFFFT_Setup *setup) {
  return setup->kind == PFFFT_SETUP_KIND_W8 ? 8 : SIMD_SZ;
}

#if PFFFT_WIDE

#include "pffft_wide_impl.h"
#include "pffft_wide_sizes.h"
#include "pffft_cpu_x86.h"
#include <cpuid.h>

static void pffft_x86_cpuid(void *ctx, unsigned leaf, unsigned subleaf, unsigned *r) {
  (void)ctx;
  __cpuid_count(leaf, subleaf, r[0], r[1], r[2], r[3]);
}

static unsigned long long pffft_x86_xgetbv(void *ctx, unsigned index) {
  unsigned lo, hi;
  (void)ctx;
  /* xgetbv, spelled out for assemblers that do not know it */
  __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(index));
  return ((unsigned long long)hi << 32) | lo;
}

#if defined(PFFFT_DISABLE_FMA)
#  define PFFFT_W8_NEED_FMA 0
#else
#  define PFFFT_W8_NEED_FMA 1
#endif

/* 0: cannot run the 8-lane backend, 1: can, 2: can and is measured faster */
static int pffft_wide_cpu_level(void) {
  pffft_x86_cpu_ops ops;
  pffft_x86_cpu_id id;
  ops.cpuid = pffft_x86_cpuid;
  ops.xgetbv = pffft_x86_xgetbv;
  ops.ctx = NULL;
  if (!pffft_x86_probe_wide(&ops, PFFFT_W8_NEED_FMA, &id)) return 0;
#if defined(PFFFT_WIDE_UNKNOWN_CPUS)
  return 2;
#else
  return pffft_x86_wide_measured(&id) ? 2 : 1;
#endif
}

/* Whether the 8-lane backend was measured faster at N in every mode (by at
   least 10%, in two independent sweeps) on the CPUs of
   pffft_x86_wide_measured(): pffft_wide_sizes.h, generated from the sweeps
   by benchmarks/perf_analysis/wide_size_table.py. */
static int pffft_wide_size_enabled(int N, pffft_transform_t transform) {
  const int *sizes = transform == PFFFT_REAL ? pffft_wide_real_sizes : pffft_wide_complex_sizes;
  int lo = 0, hi = transform == PFFFT_REAL ? PFFFT_WIDE_REAL_COUNT : PFFFT_WIDE_COMPLEX_COUNT;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (sizes[mid] == N) return 1;
    if (sizes[mid] < N) lo = mid + 1; else hi = mid;
  }
  return 0;
}

PFFFT_Setup *pffft_internal_new_setup(int N, pffft_transform_t transform, int backend) {
  if (backend != PFFFT_BACKEND_NATIVE) {
    const int supported = pffft_wide_supports(N, transform);
    const int level = supported ? pffft_wide_cpu_level() : 0;
    if (backend == PFFFT_BACKEND_W8)
      return level ? pffft_w8_new_setup(N, transform) : NULL;
    /* an allocation failure is reported, not retried with the other backend */
    if (level == 2 && pffft_wide_size_enabled(N, transform))
      return pffft_w8_new_setup(N, transform);
  }
  return pffft_native_new_setup(N, transform);
}

void pffft_destroy_setup(PFFFT_Setup *setup) {
  /* both kinds hold the same two allocations */
  pffft_native_destroy_setup(setup);
}

void pffft_transform(const PFFFT_Setup *setup, const float *input, float *output, float *work, pffft_direction_t direction) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_transform(setup, input, output, work, direction);
  else pffft_native_transform(setup, input, output, work, direction);
}

void pffft_transform_ordered(const PFFFT_Setup *setup, const float *input, float *output, float *work, pffft_direction_t direction) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_transform_ordered(setup, input, output, work, direction);
  else pffft_native_transform_ordered(setup, input, output, work, direction);
}

void pffft_zreorder(const PFFFT_Setup *setup, const float *input, float *output, pffft_direction_t direction) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_zreorder(setup, input, output, direction);
  else pffft_native_zreorder(setup, input, output, direction);
}

void pffft_zconvolve_accumulate(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab, float scaling) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_zconvolve_accumulate(setup, dft_a, dft_b, dft_ab, scaling);
  else pffft_native_zconvolve_accumulate(setup, dft_a, dft_b, dft_ab, scaling);
}

void pffft_zconvolve_scale(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab, float scaling) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
  else pffft_native_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
}

void pffft_zconvolve(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) pffft_w8_zconvolve(setup, dft_a, dft_b, dft_ab);
  else pffft_native_zconvolve(setup, dft_a, dft_b, dft_ab);
}

int pffft_zconvert_zp(const PFFFT_Setup *setup, const float *in, float *out, float scaling) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) return pffft_w8_zconvert_zp(setup, in, out, scaling);
  return pffft_native_zconvert_zp(setup, in, out, scaling);
}

int pffft_zconvolve_zp(const PFFFT_Setup *setup, const float *dft_x, const float *dft_hzp, float *dft_ab) {
  if (setup->kind == PFFFT_SETUP_KIND_W8) return pffft_w8_zconvolve_zp(setup, dft_x, dft_hzp, dft_ab);
  return pffft_native_zconvolve_zp(setup, dft_x, dft_hzp, dft_ab);
}

#else  /* PFFFT_WIDE */

PFFFT_Setup *pffft_internal_new_setup(int N, pffft_transform_t transform, int backend) {
  if (backend == PFFFT_BACKEND_W8) return NULL;
  return pffft_native_new_setup(N, transform);
}

#endif /* PFFFT_WIDE */

PFFFT_Setup *pffft_new_setup(int N, pffft_transform_t transform) {
  return pffft_internal_new_setup(N, transform, PFFFT_BACKEND_AUTO);
}



/* deprecated synonym for pffft_zconvolve_scale() -- kept for ABI compatibility */
void pffft_zconvolve_no_accu(const PFFFT_Setup *setup, const float *dft_a, const float *dft_b, float *dft_ab, float scaling) {
  pffft_zconvolve_scale(setup, dft_a, dft_b, dft_ab, scaling);
}
