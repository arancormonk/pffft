/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2026 by arancormonk <180709949+arancormonk@users.noreply.github.com>
 *
 * The FFTPACK pass codelets reused here keep their original FFTPACK license,
 * see pffft_passes_impl.h.
 */

/*
  8-lane AVX2 float backend for pffft.c.

  The 4-lane implementation splits an N-point transform into 4 interleaved
  sub-sequences x[4t+j], runs the FFTPACK passes on all of them at once
  (one per SIMD lane) and combines them with a radix-4 step across lanes.
  This backend uses the same algorithm with 8 lanes:

    A_j(k) = DFT_M{ x[8t+j] }_t          lane j, M = N/8 (complex)
    X[k + M*r] = sum_j A_j(k) W_N^(jk) W_8^(jr),   r < 8, k < M

  The pass codelets (pffft_passes_impl.h) are width-agnostic and are
  included again here under new names. Only the steps that move data
  between lanes are 8-lane specific: the (un)interleaving of the input,
  the 8x8 transposes and radix-8 combinations (finalize / preprocess),
  the spectrum reordering and the real-transform boundary terms.

  Spectrum layouts produced by this backend:

  - time-domain complex data in the passes: lane l of vector t holds
    x[8t + sigma(l)], sigma = [0 1 4 5 2 3 6 7] (see W8_TD_INTERLEAVE2).

  - complex, unordered: bins in natural order, as blocks of 8 bins with
    the 8 real parts followed by the 8 imaginary parts. Ordered output is
    the same with each block interleaved.

  - real, unordered: per block b of 8 sub-transform bins k = 8b+t,
    L = N/8, four vector pairs Y_m holding X[k + L*m] (m = 0..3) followed by
    four pairs Z_m holding X[(L-k) + L*m]. Lane 0 of block 0 is special:
    the first pair holds F(0) + i*F(N/2) as in the 4-lane layout, and the
    Z_m lanes hold X[L/2 + L*m]. Every bin 0..N/2 appears exactly once.

  The public API promises only 16-byte aligned buffers. All vector memory
  accesses therefore use VLD/VST, explicit unaligned AVX loads and stores;
  on 32-byte aligned data that costs nothing. With GCC and Clang the vector
  type is also declared 16-byte aligned.

  Requires x86-64 with GCC, Clang or MSVC; pffft.c includes this file only
  then. With GCC and Clang every function between the target pragmas below
  is compiled for AVX2 (+FMA), even ones that use no vector code: a compiler
  may still emit VEX instructions there (e.g. vzeroupper, or vzeroall with
  -fzero-call-used-regs). MSVC compiles them for the TU's target and
  VEX-encodes the AVX intrinsics. They are named pffft_w8_* or w8_*, which
  tests/check_isa_baseline.py exempts, and may only run after
  pffft_wide_cpu_class() found the CPU able to. Code that runs before that
  check, such as pffft_wide_supports() (pffft_wide_select.h), is outside.
*/

#ifndef PFFFT_WIDE_IMPL_H
#define PFFFT_WIDE_IMPL_H

#include <immintrin.h>

#define W8_SIMD_SZ 8

/* The functions pffft.c calls: external when the backend is its own
   translation unit (MSVC: pffft_wide_tu.c, see pffft_wide_api.h), else static. */
#if defined(PFFFT_WIDE_SEPARATE)
#  include "pffft_wide_api.h"
#  define W8_ENTRY(return_type) NEVER_INLINE(return_type)
#else
#  define W8_ENTRY(return_type) static NEVER_INLINE(return_type)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
   /* compiled with /arch:AVX2 (pffft_wide_tu.c, or the whole library) */
#elif defined(__clang__)
#  if defined(PFFFT_DISABLE_FMA)
#    pragma clang attribute push (__attribute__((target("avx2"))), apply_to = function)
#  else
#    pragma clang attribute push (__attribute__((target("avx2,fma"))), apply_to = function)
#  endif
#else
#  pragma GCC push_options
#  if defined(PFFFT_DISABLE_FMA)
#    pragma GCC target("avx2")
#  else
#    pragma GCC target("avx2,fma")
#  endif
#endif

#if defined(_MSC_VER) && !defined(__clang__)
typedef __m256 pffft_v8sf;
/* the callers in pffft.c may be compiled with legacy SSE encodings: clear
   the upper halves of the ymm registers before returning to them */
#  define W8_ZEROUPPER() _mm256_zeroupper()
#else
typedef __m256 pffft_v8sf __attribute__((aligned(16)));
#  define W8_ZEROUPPER() ((void)0)
#endif

/* the pass codelets and everything below use these names */
#define v4sf pffft_v8sf
#undef VZERO
#undef VMUL
#undef VADD
#undef VSUB
#undef VMADD
#undef VMSUB
#undef LD_PS1
#undef SVMUL
#undef SVMADD
#undef SVMSUB
#define VZERO() _mm256_setzero_ps()
#define VMUL(a,b) _mm256_mul_ps(a,b)
#define VADD(a,b) _mm256_add_ps(a,b)
#define VSUB(a,b) _mm256_sub_ps(a,b)
#define LD_PS1(p) _mm256_set1_ps(p)
#if defined(PFFFT_DISABLE_FMA)
#  define VMADD(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b), c)
#  define VMSUB(a,b,c) _mm256_sub_ps(c, _mm256_mul_ps(a,b))
#else
#  define VMADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#  define VMSUB(a,b,c) _mm256_fnmadd_ps(a,b,c)
#endif
#define SVMUL(f,v) VMUL(LD_PS1(f),v)
#define SVMADD(f,v,c) VMADD(LD_PS1(f),v,c)
#define SVMSUB(f,v,c) VMSUB(LD_PS1(f),v,c)
#undef VLD
#undef VST
/* p[i]: the API promises 16-byte aligned buffers only */
#define VLD(p,i) _mm256_loadu_ps((const float *)((p) + (i)))
#define VST(p,i,v) _mm256_storeu_ps((float *)((p) + (i)), (v))

#define passf2_ps        pffft_w8_passf2_ps
#define passf3_ps        pffft_w8_passf3_ps
#define passf4_ps        pffft_w8_passf4_ps
#define passf5_ps        pffft_w8_passf5_ps
#define passf8_ps        pffft_w8_passf8_ps
#define radf2_ps         pffft_w8_radf2_ps
#define radb2_ps         pffft_w8_radb2_ps
#define radf3_ps         pffft_w8_radf3_ps
#define radb3_ps         pffft_w8_radb3_ps
#define radf4_ps         pffft_w8_radf4_ps
#define radb4_ps         pffft_w8_radb4_ps
#define radf5_ps         pffft_w8_radf5_ps
#define radb5_ps         pffft_w8_radb5_ps
#define rfftf1_ps        pffft_w8_rfftf1_ps
#define rfftb1_ps        pffft_w8_rfftb1_ps
#define decompose        pffft_w8_decompose
#define twiddle_cos_sin  pffft_w8_twiddle_cos_sin
#define rffti1_ps        pffft_w8_rffti1_ps
#define cffti1_ps        pffft_w8_cffti1_ps
#define cfftf1_il_ps     pffft_w8_cfftf1_il_ps
#define cfftf1_tail_ps   pffft_w8_cfftf1_tail_ps
#define cfftf1_tail_ol_ps pffft_w8_cfftf1_tail_ol_ps
#define cfftf1_passes_ps pffft_w8_cfftf1_passes_ps
#define passf2_il_ps     pffft_w8_passf2_il_ps
#define passf3_il_ps     pffft_w8_passf3_il_ps
#define passf4_il_ps     pffft_w8_passf4_il_ps
#define passf5_il_ps     pffft_w8_passf5_il_ps
#define passf4_ol_ps     pffft_w8_passf4_ol_ps
#define passf4_il_butterfly pffft_w8_passf4_il_butterfly

/* [r0 i0 r1 i1 r2 i2 r3 i3], [r4 i4 .. r7 i7] <-> [r0 .. r7], [i0 .. i7] */
#define W8_INTERLEAVE2(in1, in2, out1, out2) {                         \
    __m256 lo__ = _mm256_unpacklo_ps(in1, in2);                         \
    __m256 hi__ = _mm256_unpackhi_ps(in1, in2);                         \
    out1 = _mm256_permute2f128_ps(lo__, hi__, 0x20);                    \
    out2 = _mm256_permute2f128_ps(lo__, hi__, 0x31); }
#define W8_UNINTERLEAVE2(in1, in2, out1, out2) {                       \
    __m256 re__ = _mm256_shuffle_ps(in1, in2, _MM_SHUFFLE(2,0,2,0));    \
    __m256 im__ = _mm256_shuffle_ps(in1, in2, _MM_SHUFFLE(3,1,3,1));    \
    out1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(re__), _MM_SHUFFLE(3,1,2,0))); \
    out2 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(im__), _MM_SHUFFLE(3,1,2,0))); }

/* Time-domain complex data uses the lane order sigma = [0 1 4 5 2 3 6 7]:
   lane l of vector t holds x[8t + sigma(l)]. Converting between that and
   interleaved complex only needs in-lane shuffles. The finalize and
   preprocess steps account for sigma (an involution) when they pick the
   lanes. The spectra use the natural lane order (W8_(UN)INTERLEAVE2). */
#define W8_TD_INTERLEAVE2(in1, in2, out1, out2) {                      \
    __m256 lo__ = _mm256_unpacklo_ps(in1, in2);                         \
    out2 = _mm256_unpackhi_ps(in1, in2);                                \
    out1 = lo__; }
#define W8_TD_UNINTERLEAVE2(in1, in2, out1, out2) {                    \
    __m256 re__ = _mm256_shuffle_ps(in1, in2, _MM_SHUFFLE(2,0,2,0));    \
    out2 = _mm256_shuffle_ps(in1, in2, _MM_SHUFFLE(3,1,3,1));           \
    out1 = re__; }
/* MAC (one of the (UN)INTERLEAVE2 macros) with its outputs stored in p[i1], p[i2] */
#define W8_STORE2(MAC, in1, in2, p, i1, i2) {                          \
    v4sf o1__, o2__; MAC(in1, in2, o1__, o2__);                         \
    VST(p, i1, o1__); VST(p, i2, o2__); }
#define W8_SIGMA(l) ((l) == 2 ? 4 : (l) == 3 ? 5 : (l) == 4 ? 2 : (l) == 5 ? 3 : (l))

#define PF_FUSED_LAYOUT 1
#define PF_UNINTERLEAVE2 W8_TD_UNINTERLEAVE2
#define PF_INTERLEAVE2 W8_TD_INTERLEAVE2
#define PF_UNINTERLEAVE2_ST(in1, in2, p, i1, i2) W8_STORE2(W8_TD_UNINTERLEAVE2, in1, in2, p, i1, i2)
#define PF_INTERLEAVE2_ST(in1, in2, p, i1, i2) W8_STORE2(W8_TD_INTERLEAVE2, in1, in2, p, i1, i2)
#include "pffft_passes_impl.h"
#undef PF_FUSED_LAYOUT
#undef PF_UNINTERLEAVE2
#undef PF_INTERLEAVE2

#define W8_ALIGNED16(p) ((((uintptr_t)(p)) & 0xF) == 0)

/* replace lane 0 of a with lane 0 of b */
#define W8_SET_LANE0(a, b) _mm256_blend_ps(a, b, 0x01)

static ALWAYS_INLINE(float) w8_lane0(v4sf v) {
  return _mm_cvtss_f32(_mm256_castps256_ps128(v));
}

static ALWAYS_INLINE(void) w8_transpose8(v4sf *r) {
  __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
  __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
  __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
  __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
  __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
  __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
  __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
  __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
  __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
  __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
  __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
  __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));
  r[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
  r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
  r[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
  r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
  r[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
  r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
  r[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
  r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

/* in-place 8-point DFT across the vectors x[0..7] (same arithmetic as
   passf8_ps): fsign = -1 forward, +1 backward, unscaled */
static ALWAYS_INLINE(void) w8_dft8(v4sf *xr, v4sf *xi, float fsign) {
  static const float K8 = (float)M_SQRT1_2;
  const float K8s = fsign * K8, nK8 = -K8;
  v4sf s0r = VADD(xr[0], xr[4]), s0i = VADD(xi[0], xi[4]);
  v4sf s1r = VADD(xr[1], xr[5]), s1i = VADD(xi[1], xi[5]);
  v4sf s2r = VADD(xr[2], xr[6]), s2i = VADD(xi[2], xi[6]);
  v4sf s3r = VADD(xr[3], xr[7]), s3i = VADD(xi[3], xi[7]);
  v4sf d0r = VSUB(xr[0], xr[4]), d0i = VSUB(xi[0], xi[4]);
  v4sf d1r = VSUB(xr[1], xr[5]), d1i = VSUB(xi[1], xi[5]);
  v4sf d2r = VSUB(xr[2], xr[6]), d2i = VSUB(xi[2], xi[6]);
  v4sf d3r = VSUB(xr[3], xr[7]), d3i = VSUB(xi[3], xi[7]);
  v4sf t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;

  VCPLXMUL(d1r, d1i, LD_PS1(K8), LD_PS1(K8s));        /* * W8^1 */
  { v4sf tr = VMUL(LD_PS1(-fsign), d2i), ti = VMUL(LD_PS1(fsign), d2r);
    d2r = tr; d2i = ti; }                              /* * W8^2 */
  VCPLXMUL(d3r, d3i, LD_PS1(nK8), LD_PS1(K8s));       /* * W8^3 */

  t1r = VSUB(s0r, s2r); t1i = VSUB(s0i, s2i);
  t2r = VADD(s0r, s2r); t2i = VADD(s0i, s2i);
  t4i = VMUL(VSUB(s1r, s3r), LD_PS1(fsign));
  t4r = VMUL(VSUB(s3i, s1i), LD_PS1(fsign));
  t3r = VADD(s1r, s3r); t3i = VADD(s1i, s3i);
  xr[0] = VADD(t2r, t3r); xi[0] = VADD(t2i, t3i);
  xr[2] = VADD(t1r, t4r); xi[2] = VADD(t1i, t4i);
  xr[4] = VSUB(t2r, t3r); xi[4] = VSUB(t2i, t3i);
  xr[6] = VSUB(t1r, t4r); xi[6] = VSUB(t1i, t4i);

  t1r = VSUB(d0r, d2r); t1i = VSUB(d0i, d2i);
  t2r = VADD(d0r, d2r); t2i = VADD(d0i, d2i);
  t4i = VMUL(VSUB(d1r, d3r), LD_PS1(fsign));
  t4r = VMUL(VSUB(d3i, d1i), LD_PS1(fsign));
  t3r = VADD(d1r, d3r); t3i = VADD(d1i, d3i);
  xr[1] = VADD(t2r, t3r); xi[1] = VADD(t2i, t3i);
  xr[3] = VADD(t1r, t4r); xi[3] = VADD(t1i, t4i);
  xr[5] = VSUB(t2r, t3r); xi[5] = VSUB(t2i, t3i);
  xr[7] = VSUB(t1r, t4r); xi[7] = VSUB(t1i, t4i);
}

/* ---------------------------------------------------------------------- */
/* complex transforms                                                     */

/* Combine the 8 lanes after the passes: block b holds A_j(k) for
   k = 8b..8b+7 (sub-sequence j in lane sigma(j)) and produces the bins
   8b+t + M*m at block position b + nb*m, interleaved when 'ordered'. */
static ALWAYS_INLINE(void) w8_cplx_finalize_impl(int Ncvec, const v4sf *in, v4sf *out,
                                                 const v4sf *e, int ordered) {
  int b, j, nb = Ncvec/W8_SIMD_SZ;
  assert(in != out);
  for (b = 0; b < nb; ++b, in += 16, e += 14) {
    v4sf tr[8], ti[8], xr[8], xi[8];
    for (j = 0; j < 8; ++j) { tr[j] = VLD(in, 2*j); ti[j] = VLD(in, 2*j+1); }
    w8_transpose8(tr);
    w8_transpose8(ti);
    for (j = 0; j < 8; ++j) { xr[j] = tr[W8_SIGMA(j)]; xi[j] = ti[W8_SIGMA(j)]; }
    for (j = 1; j < 8; ++j) VCPLXMUL(xr[j], xi[j], VLD(e, 2*j-2), VLD(e, 2*j-1));
    w8_dft8(xr, xi, -1.f);
    for (j = 0; j < 8; ++j) {
      v4sf *o = out + 2*(b + nb*j);
      if (ordered) {
        W8_STORE2(W8_INTERLEAVE2, xr[j], xi[j], o, 0, 1);
      } else {
        VST(o, 0, xr[j]); VST(o, 1, xi[j]);
      }
    }
  }
}

static ALWAYS_INLINE(void) w8_cplx_preprocess_impl(int Ncvec, const v4sf *in, v4sf *out,
                                                   const v4sf *e, int ordered) {
  int b, j, nb = Ncvec/W8_SIMD_SZ;
  assert(in != out);
  for (b = 0; b < nb; ++b, out += 16, e += 14) {
    v4sf xr[8], xi[8], tr[8], ti[8];
    for (j = 0; j < 8; ++j) {
      const v4sf *i = in + 2*(b + nb*j);
      if (ordered) {
        W8_UNINTERLEAVE2(VLD(i, 0), VLD(i, 1), xr[j], xi[j]);
      } else {
        xr[j] = VLD(i, 0); xi[j] = VLD(i, 1);
      }
    }
    w8_dft8(xr, xi, +1.f);
    for (j = 1; j < 8; ++j) VCPLXMULCONJ(xr[j], xi[j], VLD(e, 2*j-2), VLD(e, 2*j-1));
    for (j = 0; j < 8; ++j) { tr[j] = xr[W8_SIGMA(j)]; ti[j] = xi[W8_SIGMA(j)]; }
    w8_transpose8(tr);
    w8_transpose8(ti);
    for (j = 0; j < 8; ++j) { VST(out, 2*j, tr[j]); VST(out, 2*j+1, ti[j]); }
  }
}

static NEVER_INLINE(void) w8_cplx_finalize(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_cplx_finalize_impl(Ncvec, in, out, e, 0);
}
static NEVER_INLINE(void) w8_cplx_finalize_ordered(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_cplx_finalize_impl(Ncvec, in, out, e, 1);
}
static NEVER_INLINE(void) w8_cplx_preprocess(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_cplx_preprocess_impl(Ncvec, in, out, e, 0);
}
static NEVER_INLINE(void) w8_cplx_preprocess_ordered(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_cplx_preprocess_impl(Ncvec, in, out, e, 1);
}

/* ---------------------------------------------------------------------- */
/* real transforms                                                        */

/* X_m = sum_j b_j W16^(j(2m+1)): row j holds cos(pi j(2m+1)/8) for m = 0..3, then -sin(pi j(2m+1)/8) */
static const float w8_nyq_fwd[8][8] = {
  { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
  { 0.92387953251128673848f, 0.38268343236508983729f, -0.38268343236508972627f, -0.92387953251128673848f, -0.38268343236508978178f, -0.92387953251128673848f, -0.92387953251128673848f, -0.3826834323650898928f },
  { 0.70710678118654757274f, -0.70710678118654746172f, -0.70710678118654768376f, 0.70710678118654735069f, -0.70710678118654746172f, -0.70710678118654757274f, 0.70710678118654746172f, 0.70710678118654768376f },
  { 0.38268343236508983729f, -0.92387953251128684951f, 0.92387953251128651644f, -0.3826834323650898928f, -0.92387953251128673848f, 0.38268343236508967076f, 0.3826834323650903924f, -0.92387953251128673848f },
  { 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, -1.0f, 1.0f },
  { -0.38268343236508972627f, 0.92387953251128651644f, -0.92387953251128673848f, 0.38268343236509066996f, -0.92387953251128673848f, 0.3826834323650903924f, 0.38268343236508983729f, -0.92387953251128640542f },
  { -0.70710678118654746172f, 0.70710678118654768376f, 0.70710678118654657354f, -0.70710678118654735069f, -0.70710678118654757274f, -0.70710678118654735069f, 0.70710678118654846092f, 0.70710678118654768376f },
  { -0.92387953251128673848f, -0.3826834323650898928f, 0.38268343236509066996f, 0.92387953251128751564f, -0.3826834323650898928f, -0.92387953251128673848f, -0.92387953251128640542f, -0.38268343236508783889f },
};

/* B_j = sum_m Re(X_m) 2cos(pi j(2m+1)/8) - Im(X_m) 2sin(pi j(2m+1)/8): row m, lane j */
static const float w8_nyq_bwd_c[4][8] = {
  { 2.0f, 1.84775906502257347697f, 1.41421356237309514547f, 0.76536686473017967458f, 0.0f, -0.76536686473017945254f, -1.41421356237309492343f, -1.84775906502257347697f },
  { 2.0f, 0.76536686473017967458f, -1.41421356237309492343f, -1.84775906502257369901f, 0.0f, 1.84775906502257303288f, 1.41421356237309536752f, -0.7653668647301797856f },
  { 2.0f, -0.76536686473017945254f, -1.41421356237309536752f, 1.84775906502257303288f, 0.0f, -1.84775906502257347697f, 1.41421356237309314707f, 0.76536686473018133992f },
  { 2.0f, -1.84775906502257347697f, 1.41421356237309470139f, -0.7653668647301797856f, 0.0f, 0.76536686473018133992f, -1.41421356237309470139f, 1.84775906502257503128f },
};
/* see w8_nyq_bwd_c */
static const float w8_nyq_bwd_s[4][8] = {
  { 0.0f, 0.76536686473017956356f, 1.41421356237309492343f, 1.84775906502257347697f, 2.0f, 1.84775906502257347697f, 1.41421356237309514547f, 0.7653668647301797856f },
  { 0.0f, 1.84775906502257347697f, 1.41421356237309514547f, -0.76536686473017934151f, -2.0f, -0.7653668647301807848f, 1.41421356237309470139f, 1.84775906502257347697f },
  { 0.0f, 1.84775906502257347697f, -1.41421356237309492343f, -0.7653668647301807848f, 2.0f, -0.76536686473017967458f, -1.41421356237309692183f, 1.84775906502257281083f },
  { 0.0f, 0.7653668647301797856f, -1.41421356237309536752f, 1.84775906502257347697f, -2.0f, 1.84775906502257281083f, -1.41421356237309536752f, 0.76536686473017567778f },
};

/* Bins X[L/2 + L*m], m = 0..3, from the sub-transforms' Nyquist terms b_j
   (lanes of bv): X_m = sum_j b_j W16^(j(2m+1)). */
static ALWAYS_INLINE(void) w8_real_subnyquist_fwd(v4sf bv, float *xr, float *xi) {
  float x[8];
  v4sf acc = VMUL(_mm256_permutevar8x32_ps(bv, _mm256_set1_epi32(0)), _mm256_loadu_ps(w8_nyq_fwd[0]));
  int j;
  for (j = 1; j < 8; ++j)
    acc = VMADD(_mm256_permutevar8x32_ps(bv, _mm256_set1_epi32(j)), _mm256_loadu_ps(w8_nyq_fwd[j]), acc);
  _mm256_storeu_ps(x, acc);
  for (j = 0; j < 4; ++j) { xr[j] = x[j]; xi[j] = x[4+j]; }
}

/* Inverse: the sub-transforms' Nyquist terms B_j = 2 sum_m Re(X_m W16^(-j(2m+1))),
   with X_m in lane 0 of re[m], im[m]. */
static ALWAYS_INLINE(v4sf) w8_real_subnyquist_bwd(const v4sf *re, const v4sf *im) {
  v4sf acc = VZERO();
  int m;
  for (m = 0; m < 4; ++m) {
    acc = VMADD(_mm256_broadcastss_ps(_mm256_castps256_ps128(re[m])), _mm256_loadu_ps(w8_nyq_bwd_c[m]), acc);
    acc = VMSUB(_mm256_broadcastss_ps(_mm256_castps256_ps128(im[m])), _mm256_loadu_ps(w8_nyq_bwd_s[m]), acc);
  }
  return acc;
}

/* Forward combination of real block b. FFTPACK leaves each lane's
   half-complex spectrum as f0r f1r f1i ... f(L/2-1)i f(L/2)r, so the pair
   (fkr, fki) of k = 8b+t sits at in[16b-1+2t], in[16b+2t]; k = 0 has no
   imaginary part. y receives Y_0..Y_3 then Z_0..Z_3 as (re, im) pairs,
   Z_m = conj(Y_(7-m)) = X[(L-k) + L*m]. Block 0's lane 0 is completed by
   the caller. */
static ALWAYS_INLINE(void) w8_real_finalize_block(const v4sf *in, int b, const v4sf *e, v4sf *y) {
  v4sf xr[8], xi[8];
  int t;
  const v4sf *p = in + 16*b + 1;   /* pair t >= 1 at p[2t-2], p[2t-1] */
  if (b) { xr[0] = VLD(p, -2); xi[0] = VLD(p, -1); }
  else   { xr[0] = VLD(in, 0); xi[0] = VZERO(); }
  for (t = 1; t < 8; ++t) { xr[t] = VLD(p, 2*t-2); xi[t] = VLD(p, 2*t-1); }
  w8_transpose8(xr);
  w8_transpose8(xi);
  for (t = 1; t < 8; ++t) VCPLXMUL(xr[t], xi[t], VLD(e, 2*t-2), VLD(e, 2*t-1));
  w8_dft8(xr, xi, -1.f);
  for (t = 0; t < 4; ++t) {
    y[2*t] = xr[t]; y[2*t+1] = xi[t];
    y[8+2*t] = xr[7-t]; y[9+2*t] = VSUB(VZERO(), xi[7-t]);
  }
}

/* Inverse of w8_real_finalize_block: y holds Y_0..Y_3, Z_0..Z_3 with
   block 0's lane 0 already turned into Y_0..Y_7 = X[L*m]; writes the
   half-complex pairs of k = 8b..8b+7. */
static ALWAYS_INLINE(void) w8_real_preprocess_block(const v4sf *y, int b, const v4sf *e, v4sf *out) {
  v4sf xr[8], xi[8];
  int t;
  for (t = 0; t < 4; ++t) {
    xr[t] = y[2*t]; xi[t] = y[2*t+1];
    xr[7-t] = y[8+2*t]; xi[7-t] = VSUB(VZERO(), y[9+2*t]);
  }
  w8_dft8(xr, xi, +1.f);
  for (t = 1; t < 8; ++t) VCPLXMULCONJ(xr[t], xi[t], VLD(e, 2*t-2), VLD(e, 2*t-1));
  w8_transpose8(xr);
  w8_transpose8(xi);
  if (b) { VST(out, 16*b-1, xr[0]); VST(out, 16*b, xi[0]); }
  else   { VST(out, 0, xr[0]); }   /* f0i does not exist: the imaginary part is 0 */
  for (t = 1; t < 8; ++t) { VST(out, 16*b+2*t-1, xr[t]); VST(out, 16*b+2*t, xi[t]); }
}

/* Ordered real layout: bin j at floats 2j, 2j+1 (bin 0 holds F(0), F(N/2)).
   Y_m of block b holds the bins 8b+t + L*m: one interleaved chunk. The Z_m
   bins (L-8b-t) + L*m run backwards and are offset by one bin: the chunk
   C(b) at bin L*(m+1) - 8(b+1) is [Z^(b+1) lane 0, Z^b lanes 7..1], where
   Z^(dk) lane 0 stands for the X[L/2 + L*m] that the unordered layout keeps
   in block 0's lane 0. The stores below write C(b) with Z^b lane 0 in its
   first bin, which the next block (or w8_real_store_ordered_finish)
   overwrites with a 64-bit store; the loads read that bin as scalars. */
static ALWAYS_INLINE(__m256i) w8_rev_rot(void) {
  return _mm256_setr_epi32(0, 7, 6, 5, 4, 3, 2, 1);
}

static ALWAYS_INLINE(void) w8_store_pair(float *p, v4sf interleaved) {
  /* the first complex value, as 64 bits; p is 8-byte aligned */
  _mm_storel_epi64((__m128i*)p, _mm_castps_si128(_mm256_castps256_ps128(interleaved)));
}

static ALWAYS_INLINE(void) w8_real_store_ordered(const v4sf *y, float *out, int L, int b) {
  const __m256i rev = w8_rev_rot();
  int m;
  for (m = 0; m < 4; ++m) {
    v4sf *o = (v4sf*)(out + 2*(8*b + L*m));
    W8_STORE2(W8_INTERLEAVE2, y[2*m], y[2*m+1], o, 0, 1);
  }
  for (m = 0; m < 4; ++m) {
    float *c = out + 2*(L*(m+1) - 8*(b+1));   /* chunk C(b) */
    v4sf r = _mm256_permutevar8x32_ps(y[8+2*m], rev), i = _mm256_permutevar8x32_ps(y[9+2*m], rev);
    v4sf o0, o1;
    W8_INTERLEAVE2(r, i, o0, o1);
    VST((v4sf*)c, 0, o0); VST((v4sf*)c, 1, o1);
    if (b) w8_store_pair(c + 16, o0);          /* Z^b lane 0: first bin of C(b-1) */
  }
}

/* the first bin of C(dk-1): X[L/2 + L*m] */
static ALWAYS_INLINE(void) w8_real_store_ordered_finish(float *out, int L, const float *sr, const float *si) {
  int m;
  for (m = 0; m < 4; ++m) {
    out[2*(L/2 + L*m)] = sr[m];
    out[2*(L/2 + L*m)+1] = si[m];
  }
}

static ALWAYS_INLINE(void) w8_real_load_ordered(const float *in, v4sf *y, int L, int b) {
  /* undo the deinterleaving shuffle's lane order and reverse lanes 1..7 at once */
  const __m256i idx = _mm256_setr_epi32(0, 7, 6, 3, 2, 5, 4, 1);
  int m;
  for (m = 0; m < 4; ++m) {
    const v4sf *i = (const v4sf*)(in + 2*(8*b + L*m));
    W8_UNINTERLEAVE2(VLD(i, 0), VLD(i, 1), y[2*m], y[2*m+1]);
  }
  for (m = 0; m < 4; ++m) {
    const float *c = in + 2*(L*(m+1) - 8*(b+1));                 /* chunk C(b) */
    const float *z0 = b ? c + 16 : in + 2*(L/2 + L*m);           /* lane 0 */
    v4sf c0 = VLD((const v4sf*)c, 0), c1 = VLD((const v4sf*)c, 1);
    v4sf re = _mm256_shuffle_ps(c0, c1, _MM_SHUFFLE(2,0,2,0));
    v4sf im = _mm256_shuffle_ps(c0, c1, _MM_SHUFFLE(3,1,3,1));
    y[8+2*m] = W8_SET_LANE0(_mm256_permutevar8x32_ps(re, idx), _mm256_broadcast_ss(z0));
    y[9+2*m] = W8_SET_LANE0(_mm256_permutevar8x32_ps(im, idx), _mm256_broadcast_ss(z0 + 1));
  }
}

static ALWAYS_INLINE(void) w8_real_finalize_impl(int Ncvec, const v4sf *in, v4sf *out,
                                                 const v4sf *e, int ordered) {
  int b, m, dk = Ncvec/W8_SIMD_SZ, L = 2*Ncvec;
  v4sf y[16];
  float sr[4], si[4];
  assert(in != out);
  w8_real_subnyquist_fwd(VLD(in, 2*Ncvec-1), sr, si);
  for (b = 0; b < dk; ++b) {
    w8_real_finalize_block(in, b, e + 14*b, y);
    if (b == 0) {
      /* pack the real Nyquist term Y_4 next to the real DC term, then put
         X[L/2 + L*m] into the Z lanes, which only duplicate Y_1..Y_4 */
      y[1] = W8_SET_LANE0(y[1], y[14]);
      for (m = 0; m < 4; ++m) {
        y[8+2*m] = W8_SET_LANE0(y[8+2*m], LD_PS1(sr[m]));
        y[9+2*m] = W8_SET_LANE0(y[9+2*m], LD_PS1(si[m]));
      }
    }
    if (ordered) {
      w8_real_store_ordered(y, (float*)out, L, b);
    } else {
      for (m = 0; m < 16; ++m) VST(out, 16*b+m, y[m]);
    }
  }
  W8_ZEROUPPER();
  if (ordered) w8_real_store_ordered_finish((float*)out, L, sr, si);
}

static ALWAYS_INLINE(void) w8_real_preprocess_impl(int Ncvec, const v4sf *in, v4sf *out,
                                                   const v4sf *e, int ordered) {
  int b, m, dk = Ncvec/W8_SIMD_SZ, L = 2*Ncvec;
  v4sf y[16], zr[4], zi[4];
  assert(in != out);
  for (b = 0; b < dk; ++b) {
    if (ordered) {
      w8_real_load_ordered((const float*)in, y, L, b);
    } else {
      for (m = 0; m < 16; ++m) y[m] = VLD(in, 16*b+m);
    }
    if (b == 0) {
      /* lane 0: unpack F(0), F(N/2), keep X[L/2 + L*m] for the Nyquist
         terms, and complete Y_4..Y_7 = conj(Y_4..Y_1) */
      v4sf nyq = LD_PS1(w8_lane0(y[1]));
      for (m = 0; m < 4; ++m) { zr[m] = y[8+2*m]; zi[m] = y[9+2*m]; }
      y[1] = W8_SET_LANE0(y[1], VZERO());
      for (m = 0; m < 3; ++m) {
        y[8+2*m] = W8_SET_LANE0(y[8+2*m], y[2*m+2]);
        y[9+2*m] = W8_SET_LANE0(y[9+2*m], y[2*m+3]);
      }
      y[14] = W8_SET_LANE0(y[14], nyq);
      y[15] = W8_SET_LANE0(y[15], VZERO());
    }
    w8_real_preprocess_block(y, b, e + 14*b, out);
  }
  VST(out, 2*Ncvec-1, w8_real_subnyquist_bwd(zr, zi));
}

static NEVER_INLINE(void) w8_real_finalize(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_real_finalize_impl(Ncvec, in, out, e, 0);
}
static NEVER_INLINE(void) w8_real_finalize_ordered(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_real_finalize_impl(Ncvec, in, out, e, 1);
}
static NEVER_INLINE(void) w8_real_preprocess(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_real_preprocess_impl(Ncvec, in, out, e, 0);
}
static NEVER_INLINE(void) w8_real_preprocess_ordered(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  w8_real_preprocess_impl(Ncvec, in, out, e, 1);
}

/* ---------------------------------------------------------------------- */
/* reordering and transform drivers (same buffer logic as the 4-lane code) */

W8_ENTRY(void) pffft_w8_zreorder(const PFFFT_Setup *setup, const float *in, float *out,
                                            pffft_direction_t direction) {
  int k, m, Ncvec = setup->Ncvec;
  const v4sf *vin = (const v4sf*)in;
  v4sf *vout = (v4sf*)out;
  assert(in != out);
  if (setup->transform == PFFFT_REAL) {
    int dk = Ncvec/W8_SIMD_SZ, L = 2*Ncvec;
    v4sf y[16];
    if (direction == PFFFT_FORWARD) {
      float sr[4], si[4];
      for (m = 0; m < 4; ++m) {     /* block 0's Z lanes 0: X[L/2 + L*m] */
        sr[m] = in[(8+2*m)*W8_SIMD_SZ];
        si[m] = in[(9+2*m)*W8_SIMD_SZ];
      }
      for (k = 0; k < dk; ++k) {
        for (m = 0; m < 16; ++m) y[m] = VLD(vin, 16*k+m);
        w8_real_store_ordered(y, out, L, k);
      }
      W8_ZEROUPPER();
      w8_real_store_ordered_finish(out, L, sr, si);
    } else {
      for (k = 0; k < dk; ++k) {
        w8_real_load_ordered(in, y, L, k);
        for (m = 0; m < 16; ++m) VST(vout, 16*k+m, y[m]);
      }
    }
  } else if (direction == PFFFT_FORWARD) {
    for (k = 0; k < Ncvec; ++k) {
      W8_STORE2(W8_INTERLEAVE2, VLD(vin, 2*k), VLD(vin, 2*k+1), vout, 2*k, 2*k+1);
    }
  } else {
    for (k = 0; k < Ncvec; ++k) {
      W8_STORE2(W8_UNINTERLEAVE2, VLD(vin, 2*k), VLD(vin, 2*k+1), vout, 2*k, 2*k+1);
    }
  }
  W8_ZEROUPPER();
}

static NEVER_INLINE(void) pffft_w8_transform_internal(const PFFFT_Setup *setup, const float *finput,
                                                      float *foutput, v4sf *scratch,
                                                      pffft_direction_t direction, int ordered) {
  int k, Ncvec = setup->Ncvec;
  int nf_odd = (setup->ifac[1] & 1);
  const v4sf *e = (const v4sf*)setup->e;

  /* temporary buffer is allocated on the stack if the scratch pointer is NULL */
  int stack_allocate = (scratch == 0 ? Ncvec*2 : 1);
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);

  const v4sf *vinput = (const v4sf*)finput;
  v4sf *voutput      = (v4sf*)foutput;
  v4sf *buff[2]      = { voutput, scratch ? scratch : scratch_on_stack };
  int ib = (nf_odd ^ ordered ? 1 : 0);

  assert(W8_ALIGNED16(finput) && W8_ALIGNED16(foutput));

  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    if (setup->transform == PFFFT_REAL) {
      ib = (rfftf1_ps(Ncvec*2, vinput, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
      w8_real_finalize(Ncvec, buff[ib], buff[!ib], e);
    } else {
      ib = (cfftf1_il_ps(Ncvec, vinput, buff[!ib], buff[ib],
                         setup->twiddle, &setup->ifac[0], -1) == buff[0] ? 0 : 1);
      w8_cplx_finalize(Ncvec, buff[ib], buff[!ib], e);
    }
    if (ordered) {
      pffft_w8_zreorder(setup, (float*)buff[!ib], (float*)buff[ib], PFFFT_FORWARD);
    } else ib = !ib;
  } else {
    if (vinput == buff[ib]) {
      ib = !ib; /* may happen when finput == foutput */
    }
    if (ordered) {
      pffft_w8_zreorder(setup, (const float*)vinput, (float*)buff[ib], PFFFT_BACKWARD);
      vinput = buff[ib]; ib = !ib;
    }
    if (setup->transform == PFFFT_REAL) {
      w8_real_preprocess(Ncvec, vinput, buff[ib], e);
      ib = (rfftb1_ps(Ncvec*2, buff[ib], buff[0], buff[1],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
    } else {
      w8_cplx_preprocess(Ncvec, vinput, buff[ib], e);
      cfftf1_tail_ol_ps(Ncvec, buff[ib], buff[!ib], 2, 1, 0,
                        setup->twiddle, &setup->ifac[0], +1, voutput);
      ib = 0;
    }
  }

  if (buff[ib] != voutput) {
    /* extra copy required -- this situation should only happen when finput == foutput */
    assert(finput==foutput);
    for (k=0; k < Ncvec; ++k) {
      v4sf a = VLD(buff[ib], 2*k), b = VLD(buff[ib], 2*k+1);
      VST(voutput, 2*k, a); VST(voutput, 2*k+1, b);
    }
    ib = !ib;
  }
  assert(buff[ib] == voutput);
}

/* ordered real transforms with the reorder fused into finalize/preprocess */
static NEVER_INLINE(void) pffft_w8_transform_real_ordered(const PFFFT_Setup *setup,
                                          const float *input, float *output,
                                          v4sf *scratch, pffft_direction_t direction) {
  int Ncvec = setup->Ncvec;
  int ib = setup->ifac[1] & 1;
  int stack_allocate = scratch ? 1 : Ncvec*2;
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  const v4sf *vinput = (const v4sf*)input;
  const v4sf *e = (const v4sf*)setup->e;
  v4sf *voutput = (v4sf*)output;
  v4sf *buff[2] = { voutput, scratch ? scratch : scratch_on_stack };

  assert(W8_ALIGNED16(input) && W8_ALIGNED16(output));
  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    ib = (rfftf1_ps(Ncvec*2, vinput, buff[ib], buff[!ib],
                    setup->twiddle, setup->ifac) == buff[0] ? 0 : 1);
    w8_real_finalize_ordered(Ncvec, buff[ib], buff[!ib], e);
    ib = !ib;
  } else {
    if (vinput == buff[ib]) ib = !ib;
    w8_real_preprocess_ordered(Ncvec, vinput, buff[ib], e);
    ib = (rfftb1_ps(Ncvec*2, buff[ib], buff[0], buff[1],
                    setup->twiddle, setup->ifac) == buff[0] ? 0 : 1);
  }
  assert(buff[ib] == voutput);
  (void)ib;
}

/* ordered complex transforms with the reorder fused into finalize/preprocess */
static NEVER_INLINE(void) pffft_w8_transform_complex_ordered(const PFFFT_Setup *setup,
                                          const float *input, float *output,
                                          v4sf *scratch, pffft_direction_t direction) {
  int k, Ncvec = setup->Ncvec;
  int ib = setup->ifac[1] & 1;
  int stack_allocate = scratch ? 1 : Ncvec*2;
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  const v4sf *vinput = (const v4sf*)input;
  const v4sf *e = (const v4sf*)setup->e;
  v4sf *voutput = (v4sf*)output;
  v4sf *buff[2] = { voutput, scratch ? scratch : scratch_on_stack };

  assert(W8_ALIGNED16(input) && W8_ALIGNED16(output));
  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    ib = (cfftf1_il_ps(Ncvec, vinput, buff[!ib], buff[ib],
                       setup->twiddle, setup->ifac, -1) == buff[0] ? 0 : 1);
    w8_cplx_finalize_ordered(Ncvec, buff[ib], buff[!ib], e);
    ib = !ib;
  } else {
    if (vinput == buff[ib]) ib = !ib;
    w8_cplx_preprocess_ordered(Ncvec, vinput, buff[ib], e);
    cfftf1_tail_ol_ps(Ncvec, buff[ib], buff[!ib], 2, 1, 0,
                      setup->twiddle, setup->ifac, +1, voutput);
    ib = 0;
  }
  if (buff[ib] != voutput) {
    assert(input == output);
    for (k=0; k < 2*Ncvec; ++k) VST(voutput, k, VLD(buff[ib], k));
  }
}

W8_ENTRY(void) pffft_w8_transform(const PFFFT_Setup *setup, const float *input,
                                             float *output, float *work, pffft_direction_t direction) {
  pffft_w8_transform_internal(setup, input, output, (v4sf*)work, direction, 0);
  W8_ZEROUPPER();
}

W8_ENTRY(void) pffft_w8_transform_ordered(const PFFFT_Setup *setup, const float *input,
                                                     float *output, float *work, pffft_direction_t direction) {
  if (setup->transform == PFFFT_COMPLEX) {
    pffft_w8_transform_complex_ordered(setup, input, output, (v4sf*)work, direction);
  } else if (input == output && !(setup->ifac[1] & 1)) {
    pffft_w8_transform_internal(setup, input, output, (v4sf*)work, direction, 1);
  } else {
    pffft_w8_transform_real_ordered(setup, input, output, (v4sf*)work, direction);
  }
  W8_ZEROUPPER();
}

/* ---------------------------------------------------------------------- */
/* spectral operations: as in the 4-lane code, with the real transforms'
   F(0), F(N/2) in lane 0 of the first two vectors, i.e. floats 0 and 8 */

W8_ENTRY(void) pffft_w8_zconvolve_accumulate(const PFFFT_Setup *s, const float *a,
                                                        const float *b, float *ab, float scaling) {
  int i, Ncvec = s->Ncvec;
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;
  v4sf vscal = LD_PS1(scaling);
  const float ar0 = a[0], ai0 = a[W8_SIMD_SZ], br0 = b[0], bi0 = b[W8_SIMD_SZ];
  const float abr0 = ab[0], abi0 = ab[W8_SIMD_SZ];

  assert(W8_ALIGNED16(a) && W8_ALIGNED16(b) && W8_ALIGNED16(ab));
  for (i=0; i < Ncvec; i += 2) {
    v4sf ar, ai, br, bi;
    ar = VLD(va, 2*i+0); ai = VLD(va, 2*i+1);
    br = VLD(vb, 2*i+0); bi = VLD(vb, 2*i+1);
    VCPLXMUL(ar, ai, br, bi);
    VST(vab, 2*i+0, VMADD(ar, vscal, VLD(vab, 2*i+0)));
    VST(vab, 2*i+1, VMADD(ai, vscal, VLD(vab, 2*i+1)));
    ar = VLD(va, 2*i+2); ai = VLD(va, 2*i+3);
    br = VLD(vb, 2*i+2); bi = VLD(vb, 2*i+3);
    VCPLXMUL(ar, ai, br, bi);
    VST(vab, 2*i+2, VMADD(ar, vscal, VLD(vab, 2*i+2)));
    VST(vab, 2*i+3, VMADD(ai, vscal, VLD(vab, 2*i+3)));
  }
  W8_ZEROUPPER();
  if (s->transform == PFFFT_REAL) {
    ab[0] = abr0 + ar0*br0*scaling;
    ab[W8_SIMD_SZ] = abi0 + ai0*bi0*scaling;
  }
}

W8_ENTRY(void) pffft_w8_zconvolve_scale(const PFFFT_Setup *s, const float *a,
                                                   const float *b, float *ab, float scaling) {
  int k, NcvecMulTwo = 2*s->Ncvec;
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;
  v4sf vscal = LD_PS1(scaling);
  const float sar = a[0], sai = a[W8_SIMD_SZ], sbr = b[0], sbi = b[W8_SIMD_SZ];

  assert(W8_ALIGNED16(a) && W8_ALIGNED16(b) && W8_ALIGNED16(ab));
  for (k=0; k < NcvecMulTwo; k += 4) {
    v4sf var, vai, vbr, vbi;
    var = VLD(va, k+0); vai = VLD(va, k+1);
    vbr = VLD(vb, k+0); vbi = VLD(vb, k+1);
    VCPLXMUL(var, vai, vbr, vbi);
    VST(vab, k+0, VMUL(var, vscal));
    VST(vab, k+1, VMUL(vai, vscal));
    var = VLD(va, k+2); vai = VLD(va, k+3);
    vbr = VLD(vb, k+2); vbi = VLD(vb, k+3);
    VCPLXMUL(var, vai, vbr, vbi);
    VST(vab, k+2, VMUL(var, vscal));
    VST(vab, k+3, VMUL(vai, vscal));
  }
  W8_ZEROUPPER();
  if (s->transform == PFFFT_REAL) {
    ab[0] = sar*sbr*scaling;
    ab[W8_SIMD_SZ] = sai*sbi*scaling;
  }
}

W8_ENTRY(void) pffft_w8_zconvolve(const PFFFT_Setup *s, const float *a,
                                             const float *b, float *ab) {
  int k, NcvecMulTwo = 2*s->Ncvec;
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;
  const float sar = a[0], sai = a[W8_SIMD_SZ], sbr = b[0], sbi = b[W8_SIMD_SZ];

  assert(W8_ALIGNED16(a) && W8_ALIGNED16(b) && W8_ALIGNED16(ab));
  for (k=0; k < NcvecMulTwo; k += 4) {
    v4sf var, vai, vbr, vbi;
    var = VLD(va, k+0); vai = VLD(va, k+1);
    vbr = VLD(vb, k+0); vbi = VLD(vb, k+1);
    VCPLXMUL(var, vai, vbr, vbi);
    VST(vab, k+0, var);
    VST(vab, k+1, vai);
    var = VLD(va, k+2); vai = VLD(va, k+3);
    vbr = VLD(vb, k+2); vbi = VLD(vb, k+3);
    VCPLXMUL(var, vai, vbr, vbi);
    VST(vab, k+2, var);
    VST(vab, k+3, vai);
  }
  W8_ZEROUPPER();
  if (s->transform == PFFFT_REAL) {
    ab[0] = sar*sbr;
    ab[W8_SIMD_SZ] = sai*sbi;
  }
}

W8_ENTRY(int) pffft_w8_zconvert_zp(const PFFFT_Setup *s, const float *in, float *out,
                                              float scaling) {
  int j, nv = s->Ncvec * 2;
  const v4sf *vin = (const v4sf*)in;
  v4sf *vout = (v4sf*)out;
  const float nscaling = scaling / (float)s->N;
  v4sf vs = LD_PS1(nscaling);
  if (s->transform != PFFFT_REAL) return -1;
  assert(W8_ALIGNED16(in) && W8_ALIGNED16(out));
  {
    /* latch the scaled Nyquist coefficient before zeroing (in may be out);
       it is stored after the vector code, which does not touch float 8 */
    const float nyquist = in[W8_SIMD_SZ] * nscaling;
    const v4sf vz = VZERO();
    VST(vout, 0, VMUL(VLD(vin, 0), vs));
    VST(vout, 1, vz);
    for (j = 2; j < nv; j += 2) {
      VST(vout, j, VMUL(VLD(vin, j), vs));
      VST(vout, j+1, vz);
    }
    W8_ZEROUPPER();
    out[W8_SIMD_SZ] = nyquist;
  }
  return 0;
}

W8_ENTRY(int) pffft_w8_zconvolve_zp(const PFFFT_Setup *s, const float *x,
                                               const float *hzp, float *ab) {
  int j, np = s->Ncvec;
  const v4sf *vx = (const v4sf*)x;
  const v4sf *vh = (const v4sf*)hzp;
  v4sf *vab = (v4sf*)ab;
  if (s->transform != PFFFT_REAL) return -1;
  assert(W8_ALIGNED16(x) && W8_ALIGNED16(hzp) && W8_ALIGNED16(ab));
  {
    /* first pair: lane 0 of the imaginary vector is the Nyquist term,
       which multiplies the filter's Nyquist term, not its DC term. It is
       read before and stored after the vector code, which does not touch
       float 8 again (ab may be x or hzp). */
    const float nyquist = x[W8_SIMD_SZ] * hzp[W8_SIMD_SZ];
    const v4sf x0 = VLD(vx, 0), x1 = VLD(vx, 1), h0 = VLD(vh, 0);
    VST(vab, 0, VMUL(x0, h0));
    VST(vab, 1, VMUL(x1, h0));
    for (j = 1; j < np; ++j) {
      v4sf hr = VLD(vh, 2*j);
      VST(vab, 2*j, VMUL(VLD(vx, 2*j), hr));
      VST(vab, 2*j+1, VMUL(VLD(vx, 2*j+1), hr));
    }
    W8_ZEROUPPER();
    ab[W8_SIMD_SZ] = nyquist;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */
/* setup                                                                  */

/* requires pffft_wide_supports(N, transform); NULL only when an allocation fails */
W8_ENTRY(PFFFT_Setup *) pffft_w8_new_setup(int N, pffft_transform_t transform) {
  PFFFT_Setup *s;
  int k, m;
#if !defined(PFFFT_WIDE_SEPARATE)
  assert(pffft_wide_supports(N, transform));   /* checked by the caller */
#endif
  s = (PFFFT_Setup*)malloc(sizeof(PFFFT_Setup));
  if (!s) return NULL;
  s->N = N;
  s->transform = transform;
  s->kind = PFFFT_SETUP_KIND_W8;
  s->Ncvec = (transform == PFFFT_REAL ? N/2 : N)/W8_SIMD_SZ;
  s->data = pffft_aligned_malloc(2*(size_t)s->Ncvec * sizeof(pffft_v8sf));
  if (!s->data) {
    free(s);
    return NULL;
  }
  s->e = (float*)s->data;
  s->twiddle = s->e + 2*s->Ncvec*(W8_SIMD_SZ-1);

  for (k=0; k < s->Ncvec; ++k) {
    int i = k/W8_SIMD_SZ;
    int j = k%W8_SIMD_SZ;
    for (m=0; m < W8_SIMD_SZ-1; ++m) {
      /* exp(-2*pi*i*(m+1)*k/N), with (m+1)*k < N */
      float c, sn;
      twiddle_cos_sin((m+1)*k, N, &c, &sn);
      s->e[(2*(i*(W8_SIMD_SZ-1) + m) + 0) * W8_SIMD_SZ + j] = c;
      s->e[(2*(i*(W8_SIMD_SZ-1) + m) + 1) * W8_SIMD_SZ + j] = -sn;
    }
  }

  if (transform == PFFFT_REAL) {
    rffti1_ps(N/W8_SIMD_SZ, s->twiddle, s->ifac);
  } else {
    cffti1_ps(N/W8_SIMD_SZ, s->twiddle, s->ifac);
  }

  /* pffft_wide_supports() guarantees the decomposition */
  for (k=0, m=1; k < s->ifac[1]; ++k) { m *= s->ifac[2+k]; }
  assert(m == N/W8_SIMD_SZ);
  (void)m;
  return s;
}

#undef passf2_ps
#undef passf3_ps
#undef passf4_ps
#undef passf5_ps
#undef passf8_ps
#undef radf2_ps
#undef radb2_ps
#undef radf3_ps
#undef radb3_ps
#undef radf4_ps
#undef radb4_ps
#undef radf5_ps
#undef radb5_ps
#undef rfftf1_ps
#undef rfftb1_ps
#undef decompose
#undef twiddle_cos_sin
#undef rffti1_ps
#undef cffti1_ps
#undef cfftf1_il_ps
#undef cfftf1_tail_ps
#undef cfftf1_tail_ol_ps
#undef cfftf1_passes_ps
#undef passf2_il_ps
#undef passf3_il_ps
#undef passf4_il_ps
#undef passf5_il_ps
#undef passf4_ol_ps
#undef passf4_il_butterfly
#undef v4sf

#undef W8_ZEROUPPER
#undef W8_ENTRY

#if defined(_MSC_VER) && !defined(__clang__)
#elif defined(__clang__)
#  pragma clang attribute pop
#else
#  pragma GCC pop_options
#endif

#endif /* PFFFT_WIDE_IMPL_H */
