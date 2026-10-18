/* Copyright (c) 2026  Mark Connor
   Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

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
*/

#ifndef PF_VSX_DBL_H
#define PF_VSX_DBL_H

/*
  PowerPC VSX support macros (POWER7 and later; every ppc64le target).
  Packs 4 doubles into a pair of 'vector double' (2x 128-bit VSX registers),
  the same layout as the AArch64 NEON double backend.

  Only the endian-neutral intrinsics are used: vec_mergeh/vec_mergel number
  the elements in memory order on both big- and little-endian, while for
  instance vec_xxpermdi does not.
*/
#if !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) && defined(__VSX__) && \
    (defined(__ppc__) || defined(__ppc64__) || defined(__powerpc__) || defined(__powerpc64__))

#include <altivec.h>

/* v4sf holds 4 doubles as two VSX 128-bit registers */
typedef struct {
  __vector double lo;
  __vector double hi;
} v4sf;

/* 4 doubles by simd vector */
#  define SIMD_SZ 4

typedef union v4sf_union {
  v4sf  v;
  double f[SIMD_SZ];
} v4sf_union;

#  define VARCH "VSX"
#  define VREQUIRES_ALIGN 0

static inline v4sf vsx_v4sf(__vector double lo, __vector double hi) {
  v4sf r;
  r.lo = lo;
  r.hi = hi;
  return r;
}

#  define VZERO() vsx_v4sf(vec_splats(0.0), vec_splats(0.0))
#  define VMUL(a,b) vsx_v4sf(vec_mul((a).lo, (b).lo), vec_mul((a).hi, (b).hi))
#  define VADD(a,b) vsx_v4sf(vec_add((a).lo, (b).lo), vec_add((a).hi, (b).hi))
#  define VSUB(a,b) vsx_v4sf(vec_sub((a).lo, (b).lo), vec_sub((a).hi, (b).hi))
#if defined(PFFFT_DISABLE_FMA)
#  define VMADD(a,b,c) VADD(c,VMUL(a,b))
#  define VMSUB(a,b,c) VSUB(c,VMUL(a,b))
#else
/* vec_madd(a,b,c) = a*b+c and vec_nmsub(a,b,c) = c-a*b, both fused */
#  define VMADD(a,b,c) vsx_v4sf(vec_madd((a).lo, (b).lo, (c).lo), vec_madd((a).hi, (b).hi, (c).hi))
#  define VMSUB(a,b,c) vsx_v4sf(vec_nmsub((a).lo, (b).lo, (c).lo), vec_nmsub((a).hi, (b).hi, (c).hi))
#endif
#  define LD_PS1(p) vsx_v4sf(vec_splats((double)(p)), vec_splats((double)(p)))
/* memcpy rather than vec_xl, which older clang lacks or declares without
   const: compilers turn it into lxvd2x/lxvx, in memory element order */
static inline v4sf vsx_load_unaligned(const double *p) {
  v4sf r;
  memcpy(&r, p, sizeof(r));
  return r;
}
#  define VLOAD_UNALIGNED(ptr)  vsx_load_unaligned((const double*)(ptr))
#  define VLOAD_ALIGNED(ptr)    VLOAD_UNALIGNED(ptr)

/* INTERLEAVE2 (in1, in2, out1, out2) pseudo code:
   out1 = [ in1[0], in2[0], in1[1], in2[1] ]
   out2 = [ in1[2], in2[2], in1[3], in2[3] ]
*/
#  define INTERLEAVE2(in1, in2, out1, out2) {                                  \
    v4sf tmp__ = vsx_v4sf(vec_mergeh((in1).lo, (in2).lo),                      \
                          vec_mergel((in1).lo, (in2).lo));                     \
    out2 = vsx_v4sf(vec_mergeh((in1).hi, (in2).hi),                            \
                    vec_mergel((in1).hi, (in2).hi));                           \
    out1 = tmp__;                                                              \
  }

/* UNINTERLEAVE2(in1, in2, out1, out2) pseudo code:
   out1 = [ in1[0], in1[2], in2[0], in2[2] ]
   out2 = [ in1[1], in1[3], in2[1], in2[3] ]
*/
#  define UNINTERLEAVE2(in1, in2, out1, out2) {                                \
    v4sf tmp__ = vsx_v4sf(vec_mergeh((in1).lo, (in1).hi),                      \
                          vec_mergeh((in2).lo, (in2).hi));                     \
    out2 = vsx_v4sf(vec_mergel((in1).lo, (in1).hi),                            \
                    vec_mergel((in2).lo, (in2).hi));                           \
    out1 = tmp__;                                                              \
  }

/* VTRANSPOSE4 pseudo code: 4x4 matrix transpose */
#  define VTRANSPOSE4(row0, row1, row2, row3) {                                \
    __vector double r0lo__ = (row0).lo, r0hi__ = (row0).hi;                    \
    __vector double r1lo__ = (row1).lo, r1hi__ = (row1).hi;                    \
    __vector double r2lo__ = (row2).lo, r2hi__ = (row2).hi;                    \
    __vector double r3lo__ = (row3).lo, r3hi__ = (row3).hi;                    \
    (row0) = vsx_v4sf(vec_mergeh(r0lo__, r1lo__), vec_mergeh(r2lo__, r3lo__)); \
    (row1) = vsx_v4sf(vec_mergel(r0lo__, r1lo__), vec_mergel(r2lo__, r3lo__)); \
    (row2) = vsx_v4sf(vec_mergeh(r0hi__, r1hi__), vec_mergeh(r2hi__, r3hi__)); \
    (row3) = vsx_v4sf(vec_mergel(r0hi__, r1hi__), vec_mergel(r2hi__, r3hi__)); \
  }

/* VSWAPHL(a, b) pseudo code:
   return [ b[0], b[1], a[2], a[3] ]
*/
#  define VSWAPHL(a,b) vsx_v4sf((b).lo, (a).hi)

/* swap the two doubles of a vector: [x0, x1] -> [x1, x0] */
#  define VSX_SWAP2_(x) vec_mergel((x), vec_mergeh((x), (x)))

/* reverse/flip all doubles: [0,1,2,3] -> [3,2,1,0] */
#  define VREV_S(a) vsx_v4sf(VSX_SWAP2_((a).hi), VSX_SWAP2_((a).lo))

/* reverse/flip complex doubles: [0,1,2,3] -> [2,3,0,1] */
#  define VREV_C(a) vsx_v4sf((a).hi, (a).lo)

/* the transforms access buffers as 'vector double', a 16-byte aligned type */
#  define VALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0)

#endif

#endif /* PF_VSX_DBL_H */
