/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )
   Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )

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

/* this file requires architecture specific preprocessor definitions
 * it's only for library internal use
 */

#ifndef PFFFT_PRIV_IMPL_H
#define PFFFT_PRIV_IMPL_H


/* define own constants required to turn off g++ extensions .. */
#ifndef M_PI
  #define M_PI    3.14159265358979323846  /* pi */
#endif

#ifndef M_SQRT2
  #define M_SQRT2 1.41421356237309504880  /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
  #define M_SQRT1_2 0.707106781186547524400844362104849039  /* 1/sqrt(2) */
#endif

#include "pffft_setup.h"  /* PFFFT_MAX_N, the setup structure */


int FUNC_SIMD_SIZE(void) { return SIMD_SZ; }

int FUNC_MIN_FFT_SIZE(pffft_transform_t transform) {
  /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
     and 32 for real FFTs -- a lot of stuff would need to be rewritten to
     handle other cases (or maybe just switch to a scalar fft, I don't know..) */
  int simdSz = FUNC_SIMD_SIZE();
  if (transform == PFFFT_REAL)
    return ( 2 * simdSz * simdSz );
  else if (transform == PFFFT_COMPLEX)
    return ( simdSz * simdSz );
  else
    return 1;
}

int FUNC_IS_VALID_SIZE(int N, pffft_transform_t cplx) {
  const int N_min = FUNC_MIN_FFT_SIZE(cplx);
  int R = N;
  if (N <= 0 || N > PFFFT_MAX_N)
    return 0;
  while (R >= 5*N_min && (R % 5) == 0)  R /= 5;
  while (R >= 3*N_min && (R % 3) == 0)  R /= 3;
  while (R >= 2*N_min && (R % 2) == 0)  R /= 2;
  return (R == N_min) ? 1 : 0;
}

int FUNC_NEAREST_SIZE(int N, pffft_transform_t cplx, int higher) {
  int d;
  const int N_min = FUNC_MIN_FFT_SIZE(cplx);
  if (N < N_min)
    N = N_min;
  if (N > PFFFT_MAX_N) {
    N = PFFFT_MAX_N;  /* also keeps the rounding below free of int overflow */
    higher = 0;
  }
  d = (higher) ? N_min : -N_min;
  if (d > 0)
    N = N_min * ((N+N_min-1) / N_min);  /* round up */
  else
    N = N_min * (N / N_min);  /* round down */

  for (; N <= PFFFT_MAX_N; N += d)
    if (FUNC_IS_VALID_SIZE(N, cplx))
      return N;
  /* no valid size above: N_min * 2^k <= PFFFT_MAX_N always is one below */
  return FUNC_NEAREST_SIZE(PFFFT_MAX_N, cplx, 0);
}

const char * FUNC_SIMD_ARCH(void) { return VARCH; }


#if ( SIMD_SZ == 4 )
/* fuse the complex (de)interleaving into the first/last passes */
#  define PF_FUSED_LAYOUT 1
#  define PF_UNINTERLEAVE2 UNINTERLEAVE2
#  define PF_INTERLEAVE2 INTERLEAVE2
#endif
#include "pffft_passes_impl.h"
#undef PF_FUSED_LAYOUT
#undef PF_UNINTERLEAVE2
#undef PF_INTERLEAVE2


struct SETUP_STRUCT {
  PFFFT_SETUP_FIELDS(v4sf, float)
};

SETUP_STRUCT *FUNC_NEW_SETUP(int N, pffft_transform_t transform) {
  SETUP_STRUCT *s = 0;
  int k, m;
  /* validate N for negative values or potential int overflow */
  if (N <= 0 || N > PFFFT_MAX_N) {
    return s;
  }
  if (transform != PFFFT_REAL && transform != PFFFT_COMPLEX) {
    return s;
  }
  /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
     and 32 for real FFTs -- a lot of stuff would need to be rewritten to
     handle other cases (or maybe just switch to a scalar fft, I don't know..) */
  if (transform == PFFFT_REAL)    { if ((N%(2*SIMD_SZ*SIMD_SZ)) || N<=0) return s; }
  if (transform == PFFFT_COMPLEX) { if ((N%(  SIMD_SZ*SIMD_SZ)) || N<=0) return s; }
  s = (SETUP_STRUCT*)malloc(sizeof(SETUP_STRUCT));
  if (!s) return NULL;
  s->N = N;
  s->transform = transform;
  s->kind = PFFFT_SETUP_KIND_NATIVE;
  /* nb of complex simd vectors */
  s->Ncvec = (transform == PFFFT_REAL ? N/2 : N)/SIMD_SZ;
  s->data = (v4sf*)FUNC_ALIGNED_MALLOC(2*s->Ncvec * sizeof(v4sf));
  if (!s->data) {
    free(s);
    return NULL;
  }
  s->e = (float*)s->data;
  s->twiddle = (float*)(s->data + (2*s->Ncvec*(SIMD_SZ-1))/SIMD_SZ);

#if ( SIMD_SZ > 1 )  /* scalar builds have no per-lane twiddles */
  for (k=0; k < s->Ncvec; ++k) {
    int i = k/SIMD_SZ;
    int j = k%SIMD_SZ;
    for (m=0; m < SIMD_SZ-1; ++m) {
      /* exp(-2*pi*i*(m+1)*k/N), with (m+1)*k < N */
      float c, sn;
      twiddle_cos_sin((m+1)*k, N, &c, &sn);
      s->e[(2*(i*3 + m) + 0) * SIMD_SZ + j] = c;
      s->e[(2*(i*3 + m) + 1) * SIMD_SZ + j] = -sn;
    }
  }
#endif

  if (transform == PFFFT_REAL) {
    rffti1_ps(N/SIMD_SZ, s->twiddle, s->ifac);
  } else {
    cffti1_ps(N/SIMD_SZ, s->twiddle, s->ifac);
  }

  /* check that N is decomposable with allowed prime factors */
  for (k=0, m=1; k < s->ifac[1]; ++k) { m *= s->ifac[2+k]; }
  if (m != N/SIMD_SZ) {
    FUNC_DESTROY(s); s = 0;
  }

  return s;
}


void FUNC_DESTROY(SETUP_STRUCT *s) {
  if (!s)
    return;
  FUNC_ALIGNED_FREE(s->data);
  free(s);
}

#if ( SIMD_SZ == 4 )    /* !defined(PFFFT_SIMD_DISABLE) */

/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
static void reversed_copy(int N, const v4sf *in, int in_stride, v4sf *out) {
  v4sf g0, g1;
  int k;
  INTERLEAVE2(in[0], in[1], g0, g1); in += in_stride;

  *--out = VSWAPHL(g0, g1); /* [g0l, g0h], [g1l g1h] -> [g1l, g0h] */
  for (k=1; k < N; ++k) {
    v4sf h0, h1;
    INTERLEAVE2(in[0], in[1], h0, h1); in += in_stride;
    *--out = VSWAPHL(g1, h0);
    *--out = VSWAPHL(h0, h1);
    g1 = h1;
  }
  *--out = VSWAPHL(g1, g0);
}

static void unreversed_copy(int N, const v4sf *in, v4sf *out, int out_stride) {
  v4sf g0, g1, h0, h1;
  int k;
  g0 = g1 = in[0]; ++in;
  for (k=1; k < N; ++k) {
    h0 = *in++; h1 = *in++;
    g1 = VSWAPHL(g1, h0);
    h0 = VSWAPHL(h0, h1);
    UNINTERLEAVE2(h0, g1, out[0], out[1]); out += out_stride;
    g1 = h1;
  }
  h0 = *in++; h1 = g0;
  g1 = VSWAPHL(g1, h0);
  h0 = VSWAPHL(h0, h1);
  UNINTERLEAVE2(h0, g1, out[0], out[1]);
}

void FUNC_ZREORDER(const SETUP_STRUCT *setup, const float *in, float *out, pffft_direction_t direction) {
  int k, N = setup->N, Ncvec = setup->Ncvec;
  const v4sf *vin = (const v4sf*)in;
  v4sf *vout = (v4sf*)out;
  assert(in != out);
  if (setup->transform == PFFFT_REAL) {
    int dk = N/32;
    if (direction == PFFFT_FORWARD) {
      for (k=0; k < dk; ++k) {
        INTERLEAVE2(vin[k*8 + 0], vin[k*8 + 1], vout[2*(0*dk + k) + 0], vout[2*(0*dk + k) + 1]);
        INTERLEAVE2(vin[k*8 + 4], vin[k*8 + 5], vout[2*(2*dk + k) + 0], vout[2*(2*dk + k) + 1]);
      }
      reversed_copy(dk, (const v4sf*)(in + 2*SIMD_SZ), 8, (v4sf*)(out + N/2));
      reversed_copy(dk, (const v4sf*)(in + 6*SIMD_SZ), 8, (v4sf*)(out + N));
    } else {
      for (k=0; k < dk; ++k) {
        UNINTERLEAVE2(vin[2*(0*dk + k) + 0], vin[2*(0*dk + k) + 1], vout[k*8 + 0], vout[k*8 + 1]);
        UNINTERLEAVE2(vin[2*(2*dk + k) + 0], vin[2*(2*dk + k) + 1], vout[k*8 + 4], vout[k*8 + 5]);
      }
      unreversed_copy(dk, (v4sf*)(in + N/4), (v4sf*)(out + N - 6*SIMD_SZ), -8);
      unreversed_copy(dk, (v4sf*)(in + 3*N/4), (v4sf*)(out + N - 2*SIMD_SZ), -8);
    }
  } else {
    if (direction == PFFFT_FORWARD) {
      for (k=0; k < Ncvec; ++k) {
        int kk = (k/4) + (k%4)*(Ncvec/4);
        INTERLEAVE2(vin[k*2], vin[k*2+1], vout[kk*2], vout[kk*2+1]);
      }
    } else {
      for (k=0; k < Ncvec; ++k) {
        int kk = (k/4) + (k%4)*(Ncvec/4);
        UNINTERLEAVE2(vin[kk*2], vin[kk*2+1], vout[k*2], vout[k*2+1]);
      }
    }
  }
}

#define PFFFT_CPLX_ORDERED 0
#define PFFFT_CPLX_SCOPE
#define PFFFT_CPLX_FINALIZE FUNC_CPLX_FINALIZE
#define PFFFT_CPLX_PREPROCESS FUNC_CPLX_PREPROCESS
#include "pffft_cplx_impl.h"
#undef PFFFT_CPLX_ORDERED
#undef PFFFT_CPLX_SCOPE
#undef PFFFT_CPLX_FINALIZE
#undef PFFFT_CPLX_PREPROCESS

#define PFFFT_CPLX_ORDERED 1
#define PFFFT_CPLX_SCOPE static
#define PFFFT_CPLX_FINALIZE pffft_cplx_finalize_ordered
#define PFFFT_CPLX_PREPROCESS pffft_cplx_preprocess_ordered
#include "pffft_cplx_impl.h"
#undef PFFFT_CPLX_ORDERED
#undef PFFFT_CPLX_SCOPE
#undef PFFFT_CPLX_FINALIZE
#undef PFFFT_CPLX_PREPROCESS


static ALWAYS_INLINE(void) FUNC_REAL_FINALIZE_4X4(const v4sf *in0, const v4sf *in1, const v4sf *in,
                            const v4sf *e, v4sf *out) {
  v4sf r0, i0, r1, i1, r2, i2, r3, i3;
  v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
  r0 = *in0; i0 = *in1;
  r1 = *in++; i1 = *in++; r2 = *in++; i2 = *in++; r3 = *in++; i3 = *in++;
  VTRANSPOSE4(r0,r1,r2,r3);
  VTRANSPOSE4(i0,i1,i2,i3);

  /*
    transformation for each column is:

    [1   1   1   1   0   0   0   0]   [r0]
    [1   0  -1   0   0  -1   0   1]   [r1]
    [1   0  -1   0   0   1   0  -1]   [r2]
    [1  -1   1  -1   0   0   0   0]   [r3]
    [0   0   0   0   1   1   1   1] * [i0]
    [0  -1   0   1  -1   0   1   0]   [i1]
    [0  -1   0   1   1   0  -1   0]   [i2]
    [0   0   0   0  -1   1  -1   1]   [i3]
  */

  VCPLXMUL(r1,i1,e[0],e[1]);
  VCPLXMUL(r2,i2,e[2],e[3]);
  VCPLXMUL(r3,i3,e[4],e[5]);

  sr0 = VADD(r0,r2); dr0 = VSUB(r0,r2);
  sr1 = VADD(r1,r3); dr1 = VSUB(r3,r1);
  si0 = VADD(i0,i2); di0 = VSUB(i0,i2);
  si1 = VADD(i1,i3); di1 = VSUB(i3,i1);

  r0 = VADD(sr0, sr1);
  r3 = VSUB(sr0, sr1);
  i0 = VADD(si0, si1);
  i3 = VSUB(si1, si0);
  r1 = VADD(dr0, di1);
  r2 = VSUB(dr0, di1);
  i1 = VSUB(dr1, di0);
  i2 = VADD(dr1, di0);

  *out++ = r0;
  *out++ = i0;
  *out++ = r1;
  *out++ = i1;
  *out++ = r2;
  *out++ = i2;
  *out++ = r3;
  *out++ = i3;

}

static ALWAYS_INLINE(void) FUNC_REAL_PREPROCESS_4X4(const v4sf *in,
                                             const v4sf *e, v4sf *out, int first) {
  v4sf r0=in[0], i0=in[1], r1=in[2], i1=in[3], r2=in[4], i2=in[5], r3=in[6], i3=in[7];
  /*
    transformation for each column is:

    [1   1   1   1   0   0   0   0]   [r0]
    [1   0   0  -1   0  -1  -1   0]   [r1]
    [1  -1  -1   1   0   0   0   0]   [r2]
    [1   0   0  -1   0   1   1   0]   [r3]
    [0   0   0   0   1  -1   1  -1] * [i0]
    [0  -1   1   0   1   0   0   1]   [i1]
    [0   0   0   0   1   1  -1  -1]   [i2]
    [0   1  -1   0   1   0   0   1]   [i3]
  */

  v4sf sr0 = VADD(r0,r3), dr0 = VSUB(r0,r3);
  v4sf sr1 = VADD(r1,r2), dr1 = VSUB(r1,r2);
  v4sf si0 = VADD(i0,i3), di0 = VSUB(i0,i3);
  v4sf si1 = VADD(i1,i2), di1 = VSUB(i1,i2);

  r0 = VADD(sr0, sr1);
  r2 = VSUB(sr0, sr1);
  r1 = VSUB(dr0, si1);
  r3 = VADD(dr0, si1);
  i0 = VSUB(di0, di1);
  i2 = VADD(di0, di1);
  i1 = VSUB(si0, dr1);
  i3 = VADD(si0, dr1);

  VCPLXMULCONJ(r1,i1,e[0],e[1]);
  VCPLXMULCONJ(r2,i2,e[2],e[3]);
  VCPLXMULCONJ(r3,i3,e[4],e[5]);

  VTRANSPOSE4(r0,r1,r2,r3);
  VTRANSPOSE4(i0,i1,i2,i3);

  if (!first) {
    *out++ = r0;
    *out++ = i0;
  }
  *out++ = r1;
  *out++ = i1;
  *out++ = r2;
  *out++ = i2;
  *out++ = r3;
  *out++ = i3;
}

/* The second and fourth frequency quarters are reversed and rotated by one
   complex bin. Carry the previous block's high pair to keep every vector
   store aligned. Each quarter's first vector is completed after the last block. */
static ALWAYS_INLINE(void) pffft_real_store_reverse(v4sf re, v4sf im,
                                     v4sf *out, int dk, int k, v4sf *previous) {
  v4sf a, b;
  INTERLEAVE2(re, im, a, b);
  if (k != 0) {
    out[2*dk-2*k] = VSWAPHL(*previous, a);
  }
  out[2*dk-2*k-1] = VSWAPHL(a, b);
  *previous = b;
}

static ALWAYS_INLINE(void) pffft_real_store_ordered(const v4sf *block, v4sf *out,
                               int dk, int k, v4sf *previous1, v4sf *previous3) {
  INTERLEAVE2(block[0], block[1], out[2*k], out[2*k+1]);
  INTERLEAVE2(block[4], block[5], out[4*dk+2*k], out[4*dk+2*k+1]);
  pffft_real_store_reverse(block[2], block[3], out + 2*dk, dk, k, previous1);
  pffft_real_store_reverse(block[6], block[7], out + 6*dk, dk, k, previous3);
}

static ALWAYS_INLINE(void) pffft_real_load_ordered(const v4sf *in, v4sf *block,
                                                   int dk, int k) {
  const int j1 = 4*dk - 2*k, j3 = 8*dk - 2*k;
  v4sf a, b;
  UNINTERLEAVE2(in[2*k], in[2*k+1], block[0], block[1]);
  UNINTERLEAVE2(in[4*dk+2*k], in[4*dk+2*k+1], block[4], block[5]);
  /* Three aligned loads cover each reversed block. At k=0 the highest
     pair wraps to the quarter's first vector, never beyond the buffer. */
  a = VSWAPHL(in[j1-1], in[k ? j1 : 2*dk]);
  b = VSWAPHL(in[j1-2], in[j1-1]);
  UNINTERLEAVE2(a, b, block[2], block[3]);
  a = VSWAPHL(in[j3-1], in[k ? j3 : 6*dk]);
  b = VSWAPHL(in[j3-2], in[j3-1]);
  UNINTERLEAVE2(a, b, block[6], block[7]);
}

#define PFFFT_REAL_ORDERED 0
#define PFFFT_REAL_FINALIZE FUNC_REAL_FINALIZE
#define PFFFT_REAL_PREPROCESS FUNC_REAL_PREPROCESS
#include "pffft_real_impl.h"
#undef PFFFT_REAL_ORDERED
#undef PFFFT_REAL_FINALIZE
#undef PFFFT_REAL_PREPROCESS

#define PFFFT_REAL_ORDERED 1
#define PFFFT_REAL_FINALIZE pffft_real_finalize_ordered
#define PFFFT_REAL_PREPROCESS pffft_real_preprocess_ordered
#include "pffft_real_impl.h"
#undef PFFFT_REAL_ORDERED
#undef PFFFT_REAL_FINALIZE
#undef PFFFT_REAL_PREPROCESS


void FUNC_TRANSFORM_INTERNAL(const SETUP_STRUCT *setup, const float *finput, float *foutput, v4sf *scratch,
                             pffft_direction_t direction, int ordered) {
  int k, Ncvec   = setup->Ncvec;
  int nf_odd = (setup->ifac[1] & 1);

  /* temporary buffer is allocated on the stack if the scratch pointer is NULL */
  int stack_allocate = (scratch == 0 ? Ncvec*2 : 1);
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);

  const v4sf *vinput = (const v4sf*)finput;
  v4sf *voutput      = (v4sf*)foutput;
  v4sf *buff[2]      = { voutput, scratch ? scratch : scratch_on_stack };
  int ib = (nf_odd ^ ordered ? 1 : 0);

  assert(VALIGNED(finput) && VALIGNED(foutput));

  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    if (setup->transform == PFFFT_REAL) {
      ib = (rfftf1_ps(Ncvec*2, vinput, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
      FUNC_REAL_FINALIZE(Ncvec, buff[ib], buff[!ib], (v4sf*)setup->e);
    } else {
      ib = (cfftf1_il_ps(Ncvec, vinput, buff[!ib], buff[ib],
                         setup->twiddle, &setup->ifac[0], -1) == buff[0] ? 0 : 1);
      FUNC_CPLX_FINALIZE(Ncvec, buff[ib], buff[!ib], (v4sf*)setup->e);
    }
    if (ordered) {
      FUNC_ZREORDER(setup, (float*)buff[!ib], (float*)buff[ib], PFFFT_FORWARD);
    } else ib = !ib;
  } else {
    if (vinput == buff[ib]) {
      ib = !ib; /* may happen when finput == foutput */
    }
    if (ordered) {
      FUNC_ZREORDER(setup, (float*)vinput, (float*)buff[ib], PFFFT_BACKWARD);
      vinput = buff[ib]; ib = !ib;
    }
    if (setup->transform == PFFFT_REAL) {
      FUNC_REAL_PREPROCESS(Ncvec, vinput, buff[ib], (v4sf*)setup->e);
      ib = (rfftb1_ps(Ncvec*2, buff[ib], buff[0], buff[1],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
    } else {
      FUNC_CPLX_PREPROCESS(Ncvec, vinput, buff[ib], (v4sf*)setup->e);
      /* The last pass interleaves directly into the caller's buffer,
         including when the passes run in scratch during an in-place
         transform. */
      cfftf1_tail_ps(Ncvec, buff[ib], buff[!ib], 2, 1, 0,
                     setup->twiddle, &setup->ifac[0], +1, voutput);
      ib = 0;
    }
  }

  if (buff[ib] != voutput) {
    /* extra copy required -- this situation should only happen when finput == foutput */
    assert(finput==foutput);
    for (k=0; k < Ncvec; ++k) {
      v4sf a = buff[ib][2*k], b = buff[ib][2*k+1];
      voutput[2*k] = a; voutput[2*k+1] = b;
    }
    ib = !ib;
  }
  assert(buff[ib] == voutput);
}

/* One layout pass is enough for ordered real transforms. The caller keeps
   in-place transforms with even pass counts on the two-pass path, where
   the reorder does useful buffer switching instead of adding a final copy. */
static NEVER_INLINE(void) pffft_transform_real_ordered(const SETUP_STRUCT *setup,
                                          const float *input, float *output,
                                          v4sf *scratch, pffft_direction_t direction) {
  int Ncvec = setup->Ncvec;
  int ib = setup->ifac[1] & 1;
  int stack_allocate = scratch ? 1 : Ncvec*2;
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  const v4sf *vinput = (const v4sf*)input;
  v4sf *voutput = (v4sf*)output;
  v4sf *buff[2] = { voutput, scratch ? scratch : scratch_on_stack };

  assert(VALIGNED(input) && VALIGNED(output));
  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    ib = (rfftf1_ps(Ncvec*2, vinput, buff[ib], buff[!ib],
                    setup->twiddle, setup->ifac) == buff[0] ? 0 : 1);
    pffft_real_finalize_ordered(Ncvec, buff[ib], buff[!ib], (v4sf*)setup->e);
    ib = !ib;
  } else {
    if (vinput == buff[ib]) ib = !ib;
    pffft_real_preprocess_ordered(Ncvec, vinput, buff[ib], (v4sf*)setup->e);
    ib = (rfftb1_ps(Ncvec*2, buff[ib], buff[0], buff[1],
                    setup->twiddle, setup->ifac) == buff[0] ? 0 : 1);
  }
  assert(buff[ib] == voutput);
}

/* Complex ordered transforms fuse zreorder into the final/preprocessing pass.
   Keep their buffer selection separate from the real and unordered paths.
   Do not inline: otherwise the compiler can impose this function's stack
   alignment/prologue on the real transform's small dispatch wrapper too. */
static NEVER_INLINE(void) pffft_transform_complex_ordered(const SETUP_STRUCT *setup,
                                          const float *input, float *output,
                                          v4sf *scratch, pffft_direction_t direction) {
  int k, Ncvec = setup->Ncvec;
  int ib = setup->ifac[1] & 1;
  int stack_allocate = scratch ? 1 : Ncvec*2;
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  const v4sf *vinput = (const v4sf*)input;
  v4sf *voutput = (v4sf*)output;
  v4sf *buff[2] = { voutput, scratch ? scratch : scratch_on_stack };

  assert(VALIGNED(input) && VALIGNED(output));
  if (direction == PFFFT_FORWARD) {
    ib = !ib;
    ib = (cfftf1_il_ps(Ncvec, vinput, buff[!ib], buff[ib],
                       setup->twiddle, setup->ifac, -1) == buff[0] ? 0 : 1);
    pffft_cplx_finalize_ordered(Ncvec, buff[ib], buff[!ib], (v4sf*)setup->e);
    ib = !ib;
  } else {
    if (vinput == buff[ib]) ib = !ib;
    pffft_cplx_preprocess_ordered(Ncvec, vinput, buff[ib], (v4sf*)setup->e);
    cfftf1_tail_ps(Ncvec, buff[ib], buff[!ib], 2, 1, 0,
                   setup->twiddle, setup->ifac, +1, voutput);
    ib = 0;
  }
  if (buff[ib] != voutput) {
    assert(input == output);
    for (k=0; k < 2*Ncvec; ++k) voutput[k] = buff[ib][k];
  }
}

/* Unordered complex transforms in place with an even number of passes. At
   that parity the fused first (forward) or last (backward) pass of
   FUNC_TRANSFORM_INTERNAL would have to write into its own input and falls
   back to a separate (de)interleave sweep. Here the forward passes start in
   the scratch buffer and the backward preprocess runs in place: the
   unordered finalize and preprocess read each block before writing it.
   Separate, so that FUNC_TRANSFORM_INTERNAL compiles as before. */
static NEVER_INLINE(void) pffft_transform_complex_inplace(const SETUP_STRUCT *setup, float *data,
                                          v4sf *scratch, pffft_direction_t direction) {
  int Ncvec = setup->Ncvec;
  int stack_allocate = scratch ? 1 : Ncvec*2;
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  v4sf *vdata = (v4sf*)data;
  v4sf *work = scratch ? scratch : scratch_on_stack;

  assert(VALIGNED(data) && !(setup->ifac[1] & 1));
  if (direction == PFFFT_FORWARD) {
    v4sf *res = cfftf1_il_ps(Ncvec, vdata, work, vdata, setup->twiddle, setup->ifac, -1);
    FUNC_CPLX_FINALIZE(Ncvec, res, vdata, (v4sf*)setup->e);
  } else {
    FUNC_CPLX_PREPROCESS(Ncvec, vdata, vdata, (v4sf*)setup->e);
    cfftf1_tail_ps(Ncvec, vdata, work, 2, 1, 0, setup->twiddle, setup->ifac, +1, vdata);
  }
}

void FUNC_ZCONVOLVE_ACCUMULATE(const SETUP_STRUCT *s, const float *a, const float *b, float *ab, float scaling) {
  int Ncvec = s->Ncvec;
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;

#ifdef __arm__
  __builtin_prefetch(va);
  __builtin_prefetch(vb);
  __builtin_prefetch(vab);
  __builtin_prefetch(va+2);
  __builtin_prefetch(vb+2);
  __builtin_prefetch(vab+2);
  __builtin_prefetch(va+4);
  __builtin_prefetch(vb+4);
  __builtin_prefetch(vab+4);
  __builtin_prefetch(va+6);
  __builtin_prefetch(vb+6);
  __builtin_prefetch(vab+6);
#endif

  float ar0, ai0, br0, bi0, abr0, abi0;
  v4sf vscal = LD_PS1(scaling);
  int i;

  assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
  ar0 = ((v4sf_union*)va)[0].f[0];
  ai0 = ((v4sf_union*)va)[1].f[0];
  br0 = ((v4sf_union*)vb)[0].f[0];
  bi0 = ((v4sf_union*)vb)[1].f[0];
  abr0 = ((v4sf_union*)vab)[0].f[0];
  abi0 = ((v4sf_union*)vab)[1].f[0];

  for (i=0; i < Ncvec; i += 2) {
    v4sf ar, ai, br, bi;
    ar = va[2*i+0]; ai = va[2*i+1];
    br = vb[2*i+0]; bi = vb[2*i+1];
    VCPLXMUL(ar, ai, br, bi);
    vab[2*i+0] = VMADD(ar, vscal, vab[2*i+0]);
    vab[2*i+1] = VMADD(ai, vscal, vab[2*i+1]);
    ar = va[2*i+2]; ai = va[2*i+3];
    br = vb[2*i+2]; bi = vb[2*i+3];
    VCPLXMUL(ar, ai, br, bi);
    vab[2*i+2] = VMADD(ar, vscal, vab[2*i+2]);
    vab[2*i+3] = VMADD(ai, vscal, vab[2*i+3]);
  }
  if (s->transform == PFFFT_REAL) {
    ((v4sf_union*)vab)[0].f[0] = abr0 + ar0*br0*scaling;
    ((v4sf_union*)vab)[1].f[0] = abi0 + ai0*bi0*scaling;
  }
}

void FUNC_ZCONVOLVE_SCALE(const SETUP_STRUCT *s, const float *a, const float *b, float *ab, float scaling) {
  v4sf vscal = LD_PS1(scaling);
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;
  float sar, sai, sbr, sbi;
  const int NcvecMulTwo = 2*s->Ncvec;
  int k; /* was i -- but always used "2*i" - except at for() */

#ifdef __arm__
  __builtin_prefetch(va);
  __builtin_prefetch(vb);
  __builtin_prefetch(vab);
  __builtin_prefetch(va+2);
  __builtin_prefetch(vb+2);
  __builtin_prefetch(vab+2);
  __builtin_prefetch(va+4);
  __builtin_prefetch(vb+4);
  __builtin_prefetch(vab+4);
  __builtin_prefetch(va+6);
  __builtin_prefetch(vb+6);
  __builtin_prefetch(vab+6);
#endif

  assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
  sar = ((v4sf_union*)va)[0].f[0];
  sai = ((v4sf_union*)va)[1].f[0];
  sbr = ((v4sf_union*)vb)[0].f[0];
  sbi = ((v4sf_union*)vb)[1].f[0];

  /* default routine, works fine for non-arm cpus with current compilers */
  for (k=0; k < NcvecMulTwo; k += 4) {
    v4sf var, vai, vbr, vbi;
    var = va[k+0]; vai = va[k+1];
    vbr = vb[k+0]; vbi = vb[k+1];
    VCPLXMUL(var, vai, vbr, vbi);
    vab[k+0] = VMUL(var, vscal);
    vab[k+1] = VMUL(vai, vscal);
    var = va[k+2]; vai = va[k+3];
    vbr = vb[k+2]; vbi = vb[k+3];
    VCPLXMUL(var, vai, vbr, vbi);
    vab[k+2] = VMUL(var, vscal);
    vab[k+3] = VMUL(vai, vscal);
  }

  if (s->transform == PFFFT_REAL) {
    ((v4sf_union*)vab)[0].f[0] = sar*sbr*scaling;
    ((v4sf_union*)vab)[1].f[0] = sai*sbi*scaling;
  }
}


void FUNC_ZCONVOLVE(const SETUP_STRUCT *s, const float *a, const float *b, float *ab) {
  const v4sf *va = (const v4sf*)a;
  const v4sf *vb = (const v4sf*)b;
  v4sf *vab = (v4sf*)ab;
  float sar, sai, sbr, sbi;
  const int NcvecMulTwo = 2*s->Ncvec;
  int k; /* was i -- but always used "2*i" - except at for() */

#ifdef __arm__
  __builtin_prefetch(va);
  __builtin_prefetch(vb);
  __builtin_prefetch(vab);
  __builtin_prefetch(va+2);
  __builtin_prefetch(vb+2);
  __builtin_prefetch(vab+2);
  __builtin_prefetch(va+4);
  __builtin_prefetch(vb+4);
  __builtin_prefetch(vab+4);
  __builtin_prefetch(va+6);
  __builtin_prefetch(vb+6);
  __builtin_prefetch(vab+6);
#endif

  assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
  sar = ((v4sf_union*)va)[0].f[0];
  sai = ((v4sf_union*)va)[1].f[0];
  sbr = ((v4sf_union*)vb)[0].f[0];
  sbi = ((v4sf_union*)vb)[1].f[0];

  /* default routine, works fine for non-arm cpus with current compilers */
  for (k=0; k < NcvecMulTwo; k += 4) {
    v4sf var, vai, vbr, vbi;
    var = va[k+0]; vai = va[k+1];
    vbr = vb[k+0]; vbi = vb[k+1];
    VCPLXMUL(var, vai, vbr, vbi);
    vab[k+0] = var;
    vab[k+1] = vai;
    var = va[k+2]; vai = va[k+3];
    vbr = vb[k+2]; vbi = vb[k+3];
    VCPLXMUL(var, vai, vbr, vbi);
    vab[k+2] = var;
    vab[k+3] = vai;
  }

  if (s->transform == PFFFT_REAL) {
    ((v4sf_union*)vab)[0].f[0] = sar*sbr;
    ((v4sf_union*)vab)[1].f[0] = sai*sbi;
  }
}


int FUNC_ZCONVERT_ZP(const SETUP_STRUCT *s, const float *in, float *out, float scaling) {
  int j, nv = s->Ncvec * 2;  /* number of simd vectors: alternating real/imag vector pairs */
  /* no RESTRICT: in may alias out by contract */
  const v4sf *vin = (const v4sf*)in;
  v4sf *vout = (v4sf*)out;
  const float nscaling = scaling / (float)s->N;  /* unit-gain normalization */
  v4sf vs = LD_PS1(nscaling);
  if (s->transform != PFFFT_REAL) return -1;
  assert(VALIGNED(in) && VALIGNED(out));
  /*
    unordered forward REAL layout: even-indexed vectors hold the real
    parts of 4 coefficients each, odd-indexed vectors their imaginary
    parts -- except the very first odd lane, which packs the real
    Nyquist coefficient F(N/2) next to F(0) in vector 0 (same packing
    that pffft_zconvolve_accumulate() treats component-wise).
    Zero-phase form: keep/scale the real vectors, zero the imag
    vectors, keep/scale the Nyquist lane.

    The conversion applies a 1/N factor (times 'scaling'): with
    scaling == 1, transform(x, FORWARD) -> zconvolve_zp() ->
    transform(.., BACKWARD) yields the circular convolution of x
    with the filter at unit gain. */
  vout[0] = VMUL(vin[0], vs);
  {
    /* latch the scaled Nyquist coefficient before zeroing: with
       in == out the vector store below would destroy vin[1].f[0] */
    const float nyquist = ((const v4sf_union*)vin)[1].f[0] * nscaling;
    const v4sf vz = VZERO();
    vout[1] = vz;
    ((v4sf_union*)vout)[1].f[0] = nyquist;
    for (j = 2; j < nv; j += 2) {
      vout[j] = VMUL(vin[j], vs);
      vout[j+1] = vz;
    }
  }
  return 0;
}

int FUNC_ZCONVOLVE_ZP(const SETUP_STRUCT *s, const float *x, const float *hzp, float *ab) {
  int j, np = s->Ncvec;  /* number of real/imag vector pairs */
  /* no RESTRICT: any pair of pointers may alias by contract */
  const v4sf *vx = (const v4sf*)x;
  const v4sf *vh = (const v4sf*)hzp;
  v4sf *vab = (v4sf*)ab;
  if (s->transform != PFFFT_REAL) return -1;
  assert(VALIGNED(x) && VALIGNED(hzp) && VALIGNED(ab));
  /*
    with a zero-phase filter (imag vectors all zero):
    Y_re = X_re * H_re   and   Y_im = X_im * H_re.
    The first two vectors additionally carry the real DC and Nyquist
    coefficients in their f[0] lanes (see FUNC_ZCONVERT_ZP), so that
    pair is finished scalar-wise, component-wise, exactly like
    FUNC_ZCONVOLVE_ACCUMULATE() does. */
  {
    /* note: the Nyquist product reads hv1.f[0], not hv0.f[0] */
    v4sf_union xv1, hv0, hv1;
    xv1.v = vx[1];
    hv0.v = vh[0]; hv1.v = vh[1];
    vab[0] = VMUL(vx[0], vh[0]);
    ((v4sf_union*)vab)[1].f[0] = xv1.f[0] * hv1.f[0];
    ((v4sf_union*)vab)[1].f[1] = xv1.f[1] * hv0.f[1];
    ((v4sf_union*)vab)[1].f[2] = xv1.f[2] * hv0.f[2];
    ((v4sf_union*)vab)[1].f[3] = xv1.f[3] * hv0.f[3];
  }
  for (j = 1; j < np; ++j) {
    v4sf hr = vh[2*j];
    vab[2*j] = VMUL(vx[2*j], hr);
    vab[2*j+1] = VMUL(vx[2*j+1], hr);
  }
  return 0;
}

#else  /* #if ( SIMD_SZ == 4 )   * !defined(PFFFT_SIMD_DISABLE) */

/* standard routine using scalar floats, without SIMD stuff. */

#define pffft_zreorder_nosimd FUNC_ZREORDER
void pffft_zreorder_nosimd(const SETUP_STRUCT *setup, const float *in, float *out, pffft_direction_t direction) {
  int k, N = setup->N;
  if (setup->transform == PFFFT_COMPLEX) {
    for (k=0; k < 2*N; ++k) out[k] = in[k];
    return;
  }
  else if (direction == PFFFT_FORWARD) {
    float x_N = in[N-1];
    for (k=N-1; k > 1; --k) out[k] = in[k-1];
    out[0] = in[0];
    out[1] = x_N;
  } else {
    float x_N = in[1];
    for (k=1; k < N-1; ++k) out[k] = in[k+1];
    out[0] = in[0];
    out[N-1] = x_N;
  }
}

#define pffft_transform_internal_nosimd FUNC_TRANSFORM_INTERNAL
void pffft_transform_internal_nosimd(const SETUP_STRUCT *setup, const float *input, float *output, float *scratch,
                                    pffft_direction_t direction, int ordered) {
  int Ncvec   = setup->Ncvec;
  int nf_odd = (setup->ifac[1] & 1);

  /* temporary buffer is allocated on the stack if the scratch pointer is NULL */
  int stack_allocate = (scratch == 0 ? Ncvec*2 : 1);
  VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
  float *buff[2];
  int ib;
  if (setup->ifac[1] == 0) {
    /* N == 1 complex: no radix passes, the DFT is the identity. cfftf1_ps()
       would hand back 'input' itself, which the buffer bookkeeping below
       mistakes for the scratch buffer. */
    output[0] = input[0]; output[1] = input[1];
    return;
  }
  if (scratch == 0) scratch = scratch_on_stack;
  buff[0] = output; buff[1] = scratch;

  if (setup->transform == PFFFT_COMPLEX) ordered = 0; /* it is always ordered. */
  ib = (nf_odd ^ ordered ? 1 : 0);

  if (direction == PFFFT_FORWARD) {
    if (setup->transform == PFFFT_REAL) {
      ib = (rfftf1_ps(Ncvec*2, input, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
    } else {
      ib = (cfftf1_ps(Ncvec, input, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0], -1) == buff[0] ? 0 : 1);
    }
    if (ordered) {
      FUNC_ZREORDER(setup, buff[ib], buff[!ib], PFFFT_FORWARD); ib = !ib;
    }
  } else {
    if (input == buff[ib]) {
      ib = !ib; /* may happen when finput == foutput */
    }
    if (ordered) {
      FUNC_ZREORDER(setup, input, buff[!ib], PFFFT_BACKWARD);
      input = buff[!ib];
    }
    if (setup->transform == PFFFT_REAL) {
      ib = (rfftb1_ps(Ncvec*2, input, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0]) == buff[0] ? 0 : 1);
    } else {
      ib = (cfftf1_ps(Ncvec, input, buff[ib], buff[!ib],
                      setup->twiddle, &setup->ifac[0], +1) == buff[0] ? 0 : 1);
    }
  }
  if (buff[ib] != output) {
    int k;
    /* extra copy required -- this situation should happens only when finput == foutput */
    assert(input==output);
    for (k=0; k < Ncvec; ++k) {
      float a = buff[ib][2*k], b = buff[ib][2*k+1];
      output[2*k] = a; output[2*k+1] = b;
    }
    ib = !ib;
  }
  assert(buff[ib] == output);
}

#define pffft_zconvolve_accumulate_nosimd FUNC_ZCONVOLVE_ACCUMULATE
void pffft_zconvolve_accumulate_nosimd(const SETUP_STRUCT *s, const float *a, const float *b,
                                       float *ab, float scaling) {
  int NcvecMulTwo = 2*s->Ncvec;
  int k; /* was i -- but always used "2*i" - except at for() */

  if (s->transform == PFFFT_REAL) {
    /* take care of the fftpack ordering */
    ab[0] += a[0]*b[0]*scaling;
    ab[NcvecMulTwo-1] += a[NcvecMulTwo-1]*b[NcvecMulTwo-1]*scaling;
    ++ab; ++a; ++b; NcvecMulTwo -= 2;
  }
  for (k=0; k < NcvecMulTwo; k += 2) {
    float ar, ai, br, bi;
    ar = a[k+0]; ai = a[k+1];
    br = b[k+0]; bi = b[k+1];
    VCPLXMUL(ar, ai, br, bi);
    ab[k+0] += ar*scaling;
    ab[k+1] += ai*scaling;
  }
}

#define pffft_zconvolve_scale_nosimd FUNC_ZCONVOLVE_SCALE
void pffft_zconvolve_scale_nosimd(const SETUP_STRUCT *s, const float *a, const float *b,
                                    float *ab, float scaling) {
  int NcvecMulTwo = 2*s->Ncvec;
  int k; /* was i -- but always used "2*i" - except at for() */

  if (s->transform == PFFFT_REAL) {
    /* take care of the fftpack ordering */
    ab[0] = a[0]*b[0]*scaling;
    ab[NcvecMulTwo-1] = a[NcvecMulTwo-1]*b[NcvecMulTwo-1]*scaling;
    ++ab; ++a; ++b; NcvecMulTwo -= 2;
  }
  for (k=0; k < NcvecMulTwo; k += 2) {
    float ar, ai, br, bi;
    ar = a[k+0]; ai = a[k+1];
    br = b[k+0]; bi = b[k+1];
    VCPLXMUL(ar, ai, br, bi);
    ab[k+0] = ar*scaling;
    ab[k+1] = ai*scaling;
  }
}


#define pffft_zconvolve_nosimd FUNC_ZCONVOLVE
void pffft_zconvolve_nosimd(const SETUP_STRUCT *s, const float *a, const float *b,
                            float *ab) {
  int NcvecMulTwo = 2*s->Ncvec;
  int k; /* was i -- but always used "2*i" - except at for() */

  if (s->transform == PFFFT_REAL) {
    /* take care of the fftpack ordering */
    ab[0] = a[0]*b[0];
    ab[NcvecMulTwo-1] = a[NcvecMulTwo-1]*b[NcvecMulTwo-1];
    ++ab; ++a; ++b; NcvecMulTwo -= 2;
  }
  for (k=0; k < NcvecMulTwo; k += 2) {
    float ar, ai, br, bi;
    ar = a[k+0]; ai = a[k+1];
    br = b[k+0]; bi = b[k+1];
    VCPLXMUL(ar, ai, br, bi);
    ab[k+0] = ar;
    ab[k+1] = ai;
  }
}


#define pffft_zconvert_zp_nosimd FUNC_ZCONVERT_ZP
int pffft_zconvert_zp_nosimd(const SETUP_STRUCT *s, const float *in, float *out, float scaling) {
  int k, N2 = s->Ncvec * 2;
  float nscaling = scaling / (float)s->N;  /* unit-gain normalization */
  if (s->transform != PFFFT_REAL) return -1;
  out[0] = in[0] * nscaling;
  out[N2-1] = in[N2-1] * nscaling;
  for (k = 1; k < N2-1; k += 2) {
    out[k] = in[k] * nscaling;
    out[k+1] = 0.f;
  }
  return 0;
}

#define pffft_zconvolve_zp_nosimd FUNC_ZCONVOLVE_ZP
int pffft_zconvolve_zp_nosimd(const SETUP_STRUCT *s, const float *x, const float *hzp, float *ab) {
  int k, N2 = s->Ncvec * 2;
  if (s->transform != PFFFT_REAL) return -1;
  ab[0] = x[0] * hzp[0];
  ab[N2-1] = x[N2-1] * hzp[N2-1];
  for (k = 1; k < N2-1; k += 2) {
    float hr = hzp[k];
    ab[k] = x[k] * hr;
    ab[k+1] = x[k+1] * hr;
  }
  return 0;
}

#endif /* #if ( SIMD_SZ == 4 )    * !defined(PFFFT_SIMD_DISABLE) */


void FUNC_TRANSFORM_UNORDRD(const SETUP_STRUCT *setup, const float *input, float *output, float *work, pffft_direction_t direction) {
#if SIMD_SZ == 4
  if (input == output && setup->transform == PFFFT_COMPLEX && !(setup->ifac[1] & 1)) {
    pffft_transform_complex_inplace(setup, output, (v4sf*)work, direction);
    return;
  }
#endif
  FUNC_TRANSFORM_INTERNAL(setup, input, output, (v4sf*)work, direction, 0);
}

void FUNC_TRANSFORM_ORDERED(const SETUP_STRUCT *setup, const float *input, float *output, float *work, pffft_direction_t direction) {
#if SIMD_SZ == 4
  if (setup->transform == PFFFT_COMPLEX) {
    pffft_transform_complex_ordered(setup, input, output, (v4sf*)work, direction);
  } else if (input == output && !(setup->ifac[1] & 1)) {
    FUNC_TRANSFORM_INTERNAL(setup, input, output, (v4sf*)work, direction, 1);
  } else {
    pffft_transform_real_ordered(setup, input, output, (v4sf*)work, direction);
  }
#else
  FUNC_TRANSFORM_INTERNAL(setup, input, output, (v4sf*)work, direction, 1);
#endif
}


#if ( SIMD_SZ == 4 )

#define assertv4(v,f0,f1,f2,f3) assert(v.f[0] == (f0) && v.f[1] == (f1) && v.f[2] == (f2) && v.f[3] == (f3))

/* detect bugs with the vector support macros */
PFFFT_EXPORT void FUNC_VALIDATE_SIMD_A(void) {
  float f[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
  v4sf_union a0, a1, a2, a3, t, u;
  memcpy(a0.f, f, 4*sizeof(float));
  memcpy(a1.f, f+4, 4*sizeof(float));
  memcpy(a2.f, f+8, 4*sizeof(float));
  memcpy(a3.f, f+12, 4*sizeof(float));

  t = a0; u = a1; t.v = VZERO();
  printf("VZERO=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]); assertv4(t, 0, 0, 0, 0);
  t.v = VADD(a1.v, a2.v);
  printf("VADD(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]); assertv4(t, 12, 14, 16, 18);
  t.v = VMUL(a1.v, a2.v);
  printf("VMUL(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]); assertv4(t, 32, 45, 60, 77);
  t.v = VMADD(a1.v, a2.v,a0.v);
  printf("VMADD(4:7,8:11,0:3)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]); assertv4(t, 32, 46, 62, 80);

  INTERLEAVE2(a1.v,a2.v,t.v,u.v);
  printf("INTERLEAVE2(4:7,8:11)=[%2g %2g %2g %2g] [%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3], u.f[0], u.f[1], u.f[2], u.f[3]);
  assertv4(t, 4, 8, 5, 9); assertv4(u, 6, 10, 7, 11);
  UNINTERLEAVE2(a1.v,a2.v,t.v,u.v);
  printf("UNINTERLEAVE2(4:7,8:11)=[%2g %2g %2g %2g] [%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3], u.f[0], u.f[1], u.f[2], u.f[3]);
  assertv4(t, 4, 6, 8, 10); assertv4(u, 5, 7, 9, 11);

  t.v=LD_PS1(f[15]);
  printf("LD_PS1(15)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
  assertv4(t, 15, 15, 15, 15);
  t.v = VSWAPHL(a1.v, a2.v);
  printf("VSWAPHL(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
  assertv4(t, 8, 9, 6, 7);
  VTRANSPOSE4(a0.v, a1.v, a2.v, a3.v);
  printf("VTRANSPOSE4(0:3,4:7,8:11,12:15)=[%2g %2g %2g %2g] [%2g %2g %2g %2g] [%2g %2g %2g %2g] [%2g %2g %2g %2g]\n",
         a0.f[0], a0.f[1], a0.f[2], a0.f[3], a1.f[0], a1.f[1], a1.f[2], a1.f[3],
         a2.f[0], a2.f[1], a2.f[2], a2.f[3], a3.f[0], a3.f[1], a3.f[2], a3.f[3]);
  assertv4(a0, 0, 4, 8, 12); assertv4(a1, 1, 5, 9, 13); assertv4(a2, 2, 6, 10, 14); assertv4(a3, 3, 7, 11, 15);
}


static void pffft_assert1( float result, float ref, const char * vartxt, const char * functxt, int * numErrs, const char * f, int lineNo )
{
  if ( !( fabs( result - ref ) < 0.01F ) )
  {
    fprintf(stderr, "%s: assert for %s at %s(%d)\n  expected %f  value %f\n", functxt, vartxt, f, lineNo, ref, result);
    ++(*numErrs);
  }
}

static void pffft_assert4(  vsfscalar v0, vsfscalar v1, vsfscalar v2, vsfscalar v3,
  float a, float b, float c, float d, const char * functxt, int * numErrs, const char * f, int lineNo )
{
  pffft_assert1( v0, a, "[0]", functxt, numErrs, f, lineNo );
  pffft_assert1( v1, b, "[1]", functxt, numErrs, f, lineNo );
  pffft_assert1( v2, c, "[2]", functxt, numErrs, f, lineNo );
  pffft_assert1( v3, d, "[3]", functxt, numErrs, f, lineNo );
}

#define PFFFT_ASSERT4( V, a, b, c, d, FUNCTXT )  pffft_assert4( (V).f[0], (V).f[1], (V).f[2], (V).f[3], a, b, c, d, FUNCTXT, &numErrs, __FILE__, __LINE__ )


PFFFT_EXPORT int FUNC_VALIDATE_SIMD_EX(FILE * DbgOut)
{
  int numErrs = 0;

  {
    v4sf_union C;
    int k;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: { }\n" );
    }
    C.v = VZERO();
    if (DbgOut) {
      fprintf(DbgOut, "VZERO(a) => C) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( C, 0.0F, 0.0F, 0.0F, 0.0F, "VZERO() Out C" );
  }

  {
    v4sf_union C;
    float a = 42.0F;
    int k;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: a = {\n" );
      fprintf(DbgOut, "  Inp a:  %f\n", a );
      fprintf(DbgOut, "}\n" );
    }
    C.v = LD_PS1(a);
    if (DbgOut) {
      fprintf(DbgOut, "LD_PS1(a) => C) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( C, 42.0F, 42.0F, 42.0F, 42.0F, "LD_PS1() Out C" );
  }

  {
    v4sf_union C;
    float a[16];
    int numAligned = 0, numUnaligned = 0;
    int k;
    const char * pUn;
    for ( k = 0; k < 16; ++k ) a[k] = k+1;

    for ( k = 0; k + 3 < 16; ++k )
    {
      const float * ptr = &a[k];
      if (DbgOut)
        fprintf(DbgOut, "\ninput: a = [ %f, %f, %f, %f ]\n", ptr[0], ptr[1], ptr[2], ptr[3] );
      if ( VALIGNED(ptr) )
      {
        C.v = VLOAD_ALIGNED( ptr );
        pUn = "";
        ++numAligned;
      }
      else
      {
        C.v = VLOAD_UNALIGNED( ptr );
        pUn = "UN";
        ++numUnaligned;
      }
      if (DbgOut) {
        fprintf(DbgOut, "C = VLOAD_%sALIGNED(&a[%d]) => {\n", pUn, k );
        fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
        fprintf(DbgOut, "}\n" );
      }

      if ( numAligned >= 1 && numUnaligned >= 4 )
        break;
    }
    if ( numAligned < 1 ) {
      fprintf(stderr, "VALIGNED() should have found at least 1 occurence!");
      ++numErrs;
    }
    if ( numUnaligned < 4 ) {
      fprintf(stderr, "!VALIGNED() should have found at least 4 occurences!");
      ++numErrs;
    }
  }

  {
    v4sf_union A, B, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 20 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VADD(A.v, B.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VADD(A,B) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VADD(): Inp A" );
    PFFFT_ASSERT4( B, 21.0F, 22.0F, 23.0F, 24.0F, "VADD(): Inp B" );
    PFFFT_ASSERT4( C, 32.0F, 34.0F, 36.0F, 38.0F, "VADD(): Out C" );
  }

  {
    v4sf_union A, B, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 20 + 2*k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VSUB(A.v, B.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VSUB(A,B) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 21.0F, 23.0F, 25.0F, 27.0F, "VSUB(): Inp A" );
    PFFFT_ASSERT4( B, 11.0F, 12.0F, 13.0F, 14.0F, "VSUB(): Inp B" );
    PFFFT_ASSERT4( C, 10.0F, 11.0F, 12.0F, 13.0F, "VSUB(): Out C" );
  }

  {
    v4sf_union A, B, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VMUL(A.v, B.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VMUL(A,B) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VMUL(): Inp A" );
    PFFFT_ASSERT4( B,  1.0F,  2.0F,  3.0F,  4.0F, "VMUL(): Inp B" );
    PFFFT_ASSERT4( C, 11.0F, 24.0F, 39.0F, 56.0F, "VMUL(): Out C" );
  }

  {
    v4sf_union A, B, C, D;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 10 + k;
    for ( k = 0; k < 4; ++k )  D.f[k] = 40 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B,C = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "  Inp C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    D.v = VMADD(A.v, B.v, C.v);
    if (DbgOut) {
      fprintf(DbgOut, "D = VMADD(A,B,C) => {\n" );
      fprintf(DbgOut, "  Out D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VMADD(): Inp A" );
    PFFFT_ASSERT4( B,  1.0F,  2.0F,  3.0F,  4.0F, "VMADD(): Inp B" );
    PFFFT_ASSERT4( C, 10.0F, 11.0F, 12.0F, 13.0F, "VMADD(): Inp C" );
    PFFFT_ASSERT4( D, 21.0F, 35.0F, 51.0F, 69.0F, "VMADD(): Out D" );
  }

  {
    v4sf_union A, B, C, D;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 100 + k;
    for ( k = 0; k < 4; ++k )  D.f[k] = 40 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B,C = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "  Inp C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    D.v = VMSUB(A.v, B.v, C.v);
    if (DbgOut) {
      fprintf(DbgOut, "D = VMSUB(A,B,C) => {\n" );
      fprintf(DbgOut, "  Out D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A,  11.0F,  12.0F,  13.0F,  14.0F, "VMSUB(): Inp A" );
    PFFFT_ASSERT4( B,   1.0F,   2.0F,   3.0F,   4.0F, "VMSUB(): Inp B" );
    PFFFT_ASSERT4( C, 100.0F, 101.0F, 102.0F, 103.0F, "VMSUB(): Inp C" );
    PFFFT_ASSERT4( D,  89.0F,  77.0F,  63.0F,  47.0F, "VMSUB(): Out D" );
  }

  {
    v4sf_union A, B, C, D;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 20 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;
    for ( k = 0; k < 4; ++k )  D.f[k] = 40 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    INTERLEAVE2(A.v, B.v, C.v, D.v);
    if (DbgOut) {
      fprintf(DbgOut, "INTERLEAVE2(A,B, => C,D) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "  Out D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "INTERLEAVE2() Inp A" );
    PFFFT_ASSERT4( B, 21.0F, 22.0F, 23.0F, 24.0F, "INTERLEAVE2() Inp B" );
    PFFFT_ASSERT4( C, 11.0F, 21.0F, 12.0F, 22.0F, "INTERLEAVE2() Out C" );
    PFFFT_ASSERT4( D, 13.0F, 23.0F, 14.0F, 24.0F, "INTERLEAVE2() Out D" );
  }

  {
    v4sf_union A, B, C, D;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 20 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;
    for ( k = 0; k < 4; ++k )  D.f[k] = 40 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    UNINTERLEAVE2(A.v, B.v, C.v, D.v);
    if (DbgOut) {
      fprintf(DbgOut, "UNINTERLEAVE2(A,B, => C,D) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "  Out D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "UNINTERLEAVE2() Inp A" );
    PFFFT_ASSERT4( B, 21.0F, 22.0F, 23.0F, 24.0F, "UNINTERLEAVE2() Inp B" );
    PFFFT_ASSERT4( C, 11.0F, 13.0F, 21.0F, 23.0F, "UNINTERLEAVE2() Out C" );
    PFFFT_ASSERT4( D, 12.0F, 14.0F, 22.0F, 24.0F, "UNINTERLEAVE2() Out D" );
  }

  {
    v4sf_union A, B, C, D;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 20 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;
    for ( k = 0; k < 4; ++k )  D.f[k] = 40 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B,C,D = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "  Inp C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "  Inp D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    VTRANSPOSE4(A.v, B.v, C.v, D.v);
    if (DbgOut) {
      fprintf(DbgOut, "VTRANSPOSE4(A,B,C,D) => {\n" );
      fprintf(DbgOut, "  Out A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Out B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "  Out D:  %f, %f, %f, %f\n", D.f[0], D.f[1], D.f[2], D.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 21.0F, 31.0F, 41.0F, "VTRANSPOSE4(): Out A" );
    PFFFT_ASSERT4( B, 12.0F, 22.0F, 32.0F, 42.0F, "VTRANSPOSE4(): Out B" );
    PFFFT_ASSERT4( C, 13.0F, 23.0F, 33.0F, 43.0F, "VTRANSPOSE4(): Out C" );
    PFFFT_ASSERT4( D, 14.0F, 24.0F, 34.0F, 44.0F, "VTRANSPOSE4(): Out D" );
  }

  {
    v4sf_union A, B, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  B.f[k] = 20 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A,B = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "  Inp B:  %f, %f, %f, %f\n", B.f[0], B.f[1], B.f[2], B.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VSWAPHL(A.v, B.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VSWAPHL(A,B) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VSWAPHL(): Inp A" );
    PFFFT_ASSERT4( B, 21.0F, 22.0F, 23.0F, 24.0F, "VSWAPHL(): Inp B" );
    PFFFT_ASSERT4( C, 21.0F, 22.0F, 13.0F, 14.0F, "VSWAPHL(): Out C" );
  }

  {
    v4sf_union A, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;
    for ( k = 0; k < 4; ++k )  C.f[k] = 30 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VREV_S(A.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VREV_S(A) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VREV_S(): Inp A" );
    PFFFT_ASSERT4( C, 14.0F, 13.0F, 12.0F, 11.0F, "VREV_S(): Out C" );
  }

  {
    v4sf_union A, C;
    int k;
    for ( k = 0; k < 4; ++k )  A.f[k] = 10 + k+1;

    if (DbgOut) {
      fprintf(DbgOut, "\ninput: A = {\n" );
      fprintf(DbgOut, "  Inp A:  %f, %f, %f, %f\n", A.f[0], A.f[1], A.f[2], A.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    C.v = VREV_C(A.v);
    if (DbgOut) {
      fprintf(DbgOut, "C = VREV_C(A) => {\n" );
      fprintf(DbgOut, "  Out C:  %f, %f, %f, %f\n", C.f[0], C.f[1], C.f[2], C.f[3] );
      fprintf(DbgOut, "}\n" );
    }
    PFFFT_ASSERT4( A, 11.0F, 12.0F, 13.0F, 14.0F, "VREV_C(): Inp A" );
    PFFFT_ASSERT4( C, 13.0F, 14.0F, 11.0F, 12.0F, "VREV_C(): Out A" );
  }

  return numErrs;
}

#else  /* if ( SIMD_SZ == 4 ) */

PFFFT_EXPORT void FUNC_VALIDATE_SIMD_A(void)
{
}

PFFFT_EXPORT int FUNC_VALIDATE_SIMD_EX(FILE * DbgOut)
{
  (void)DbgOut;
  return -1;
}

#endif  /* end if ( SIMD_SZ == 4 ) */

#endif /* PFFFT_PRIV_IMPL_H */
