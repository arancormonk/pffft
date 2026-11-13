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

/* FFTPACK pass codelets, their drivers and the twiddle/factor setup.

   Included by pffft_priv_impl.h for the 4-lane backends (float, double) and
   again by pffft_wide_impl.h for the 8-lane AVX2 float backend, which renames
   every function defined here before the inclusion. The code only uses the
   vector macros (v4sf, VADD, VMUL, LD_PS1, VCPLXMUL, ...), so it works at any
   vector width. Deliberately no include guard.

   With PFFFT_INPLACE_FIRST_PASS and the fused layout, the file also
   includes itself once, with PF_INPLACE_CODELETS defined. That inclusion
   compiles the eight complex codelets passf2/3/4/5_ps and
   passf2/3/4/5_il_ps a second time, as the always-inline bodies
   passf2_body, ..., passf5_il_body without 'restrict', for the in-place
   first-pass wrappers (passf2_ip_ps, ..., passf5_il_ip_ps); only the
   function header changes, through PF_CODELET, PF_CODELET_NAME and
   PF_CODELET_RESTRICT. Only the eight codelets (with their local macros),
   the definitions of those three macros and the #ifdef PF_FUSED_LAYOUT
   around the four _il codelets are outside #ifndef PF_INPLACE_CODELETS. */

#ifndef PF_INPLACE_CODELETS
#include "pffft_pass_config.h"  /* PFFFT_TERMINAL_RADIX8, PFFFT_INPLACE_FIRST_PASS */
#endif

/* the header of the eight codelets compiled twice: the regular codelet,
   or the body of an in-place wrapper */
#undef PF_CODELET
#undef PF_CODELET_NAME
#undef PF_CODELET_RESTRICT
#ifdef PF_INPLACE_CODELETS
#  define PF_CODELET(t) ALWAYS_INLINE(t)
#  define PF_CODELET_NAME(name) name##_body
#  define PF_CODELET_RESTRICT
#else
#  define PF_CODELET(t) NEVER_INLINE(t)
#  define PF_CODELET_NAME(name) name##_ps
#  define PF_CODELET_RESTRICT RESTRICT
#endif

/*
  passf2 and passb2 has been merged here, fsign = -1 for passf2, +1 for passb2
*/
static PF_CODELET(void) PF_CODELET_NAME(passf2)(int ido, int l1, const v4sf * PF_CODELET_RESTRICT cc,
                                                v4sf * PF_CODELET_RESTRICT ch,
                                                const float *wa1, float fsign) {
  int k, i;
  int l1ido = l1*ido;
  if (ido <= 2) {
    for (k=0; k < l1ido; k += ido, ch += ido, cc+= 2*ido) {
      VST(ch, 0, VADD(VLD(cc, 0), VLD(cc, ido+0)));
      VST(ch, l1ido, VSUB(VLD(cc, 0), VLD(cc, ido+0)));
      VST(ch, 1, VADD(VLD(cc, 1), VLD(cc, ido+1)));
      VST(ch, l1ido + 1, VSUB(VLD(cc, 1), VLD(cc, ido+1)));
    }
  } else {
    for (k=0; k < l1ido; k += ido, ch += ido, cc += 2*ido) {
      for (i=0; i<ido-1; i+=2) {
        v4sf tr2 = VSUB(VLD(cc, i+0), VLD(cc, i+ido+0));
        v4sf ti2 = VSUB(VLD(cc, i+1), VLD(cc, i+ido+1));
        v4sf wr = LD_PS1(wa1[i]), wi = VMUL(LD_PS1(fsign), LD_PS1(wa1[i+1]));
        VST(ch, i, VADD(VLD(cc, i+0), VLD(cc, i+ido+0)));
        VST(ch, i+1, VADD(VLD(cc, i+1), VLD(cc, i+ido+1)));
        VCPLXMUL(tr2, ti2, wr, wi);
        VST(ch, i+l1ido, tr2);
        VST(ch, i+l1ido+1, ti2);
      }
    }
  }
}

/*
  passf3 and passb3 has been merged here, fsign = -1 for passf3, +1 for passb3

  The radix-3/5 twiddle constants are written as full-precision double literals
  without an 'f' suffix, for when this file is compiled with '#define float double'.
  In the float build, the values round to the same floats as the old suffixed literals.
*/
static PF_CODELET(void) PF_CODELET_NAME(passf3)(int ido, int l1, const v4sf * PF_CODELET_RESTRICT cc,
                                                v4sf * PF_CODELET_RESTRICT ch,
                                                const float *wa1, const float *wa2, float fsign) {
  static const float taur = (float)-0.5;                     /* cos(2*pi/3)   */
  float taui = (float)0.86602540378443864676*fsign;          /* sin(2*pi/3) = sqrt(3)/2 */
  int i, k;
  v4sf tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
  int l1ido = l1*ido;
  float wr1, wi1, wr2, wi2;
  assert(ido >= 2);  /* ido == 2 (one complex element) is reached when SIMD_SZ == 1 */
  for (k=0; k< l1ido; k += ido, cc+= 3*ido, ch +=ido) {
    for (i=0; i<ido-1; i+=2) {
      tr2 = VADD(VLD(cc, i+ido), VLD(cc, i+2*ido));
      cr2 = SVMADD(taur, tr2, VLD(cc, i));
      VST(ch, i, VADD(VLD(cc, i), tr2));
      ti2 = VADD(VLD(cc, i+ido+1), VLD(cc, i+2*ido+1));
      ci2 = SVMADD(taur, ti2, VLD(cc, i    +1));
      VST(ch, i+1, VADD(VLD(cc, i+1), ti2));
      cr3 = SVMUL(taui, VSUB(VLD(cc, i+ido), VLD(cc, i+2*ido)));
      ci3 = SVMUL(taui, VSUB(VLD(cc, i+ido+1), VLD(cc, i+2*ido+1)));
      dr2 = VSUB(cr2, ci3);
      dr3 = VADD(cr2, ci3);
      di2 = VADD(ci2, cr3);
      di3 = VSUB(ci2, cr3);
      wr1=wa1[i]; wi1=fsign*wa1[i+1]; wr2=wa2[i]; wi2=fsign*wa2[i+1];
      VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
      VST(ch, i+l1ido, dr2);
      VST(ch, i+l1ido + 1, di2);
      VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
      VST(ch, i+2*l1ido, dr3);
      VST(ch, i+2*l1ido+1, di3);
    }
  }
} /* passf3 */

static PF_CODELET(void) PF_CODELET_NAME(passf4)(int ido, int l1, const v4sf * PF_CODELET_RESTRICT cc,
                                                v4sf * PF_CODELET_RESTRICT ch,
                                                const float *wa1, const float *wa2, const float *wa3,
                                                float fsign) {
  /* isign == -1 for forward transform and +1 for backward transform */

  int i, k;
  v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
  int l1ido = l1*ido;
  if (ido == 2) {
    for (k=0; k < l1ido; k += ido, ch += ido, cc += 4*ido) {
      tr1 = VSUB(VLD(cc, 0), VLD(cc, 2*ido + 0));
      tr2 = VADD(VLD(cc, 0), VLD(cc, 2*ido + 0));
      ti1 = VSUB(VLD(cc, 1), VLD(cc, 2*ido + 1));
      ti2 = VADD(VLD(cc, 1), VLD(cc, 2*ido + 1));
      ti4 = VMUL(VSUB(VLD(cc, 1*ido + 0), VLD(cc, 3*ido + 0)), LD_PS1(fsign));
      tr4 = VMUL(VSUB(VLD(cc, 3*ido + 1), VLD(cc, 1*ido + 1)), LD_PS1(fsign));
      tr3 = VADD(VLD(cc, ido + 0), VLD(cc, 3*ido + 0));
      ti3 = VADD(VLD(cc, ido + 1), VLD(cc, 3*ido + 1));

      VST(ch, 0*l1ido + 0, VADD(tr2, tr3));
      VST(ch, 0*l1ido + 1, VADD(ti2, ti3));
      VST(ch, 1*l1ido + 0, VADD(tr1, tr4));
      VST(ch, 1*l1ido + 1, VADD(ti1, ti4));
      VST(ch, 2*l1ido + 0, VSUB(tr2, tr3));
      VST(ch, 2*l1ido + 1, VSUB(ti2, ti3));
      VST(ch, 3*l1ido + 0, VSUB(tr1, tr4));
      VST(ch, 3*l1ido + 1, VSUB(ti1, ti4));
    }
  } else {
    for (k=0; k < l1ido; k += ido, ch+=ido, cc += 4*ido) {
      for (i=0; i<ido-1; i+=2) {
        float wr1, wi1, wr2, wi2, wr3, wi3;
        tr1 = VSUB(VLD(cc, i + 0), VLD(cc, i + 2*ido + 0));
        tr2 = VADD(VLD(cc, i + 0), VLD(cc, i + 2*ido + 0));
        ti1 = VSUB(VLD(cc, i + 1), VLD(cc, i + 2*ido + 1));
        ti2 = VADD(VLD(cc, i + 1), VLD(cc, i + 2*ido + 1));
        tr4 = VMUL(VSUB(VLD(cc, i + 3*ido + 1), VLD(cc, i + 1*ido + 1)), LD_PS1(fsign));
        ti4 = VMUL(VSUB(VLD(cc, i + 1*ido + 0), VLD(cc, i + 3*ido + 0)), LD_PS1(fsign));
        tr3 = VADD(VLD(cc, i + ido + 0), VLD(cc, i + 3*ido + 0));
        ti3 = VADD(VLD(cc, i + ido + 1), VLD(cc, i + 3*ido + 1));

        VST(ch, i, VADD(tr2, tr3));
        cr3    = VSUB(tr2, tr3);
        VST(ch, i + 1, VADD(ti2, ti3));
        ci3 = VSUB(ti2, ti3);

        cr2 = VADD(tr1, tr4);
        cr4 = VSUB(tr1, tr4);
        ci2 = VADD(ti1, ti4);
        ci4 = VSUB(ti1, ti4);
        wr1=wa1[i]; wi1=fsign*wa1[i+1];
        VCPLXMUL(cr2, ci2, LD_PS1(wr1), LD_PS1(wi1));
        wr2=wa2[i]; wi2=fsign*wa2[i+1];
        VST(ch, i + l1ido, cr2);
        VST(ch, i + l1ido + 1, ci2);

        VCPLXMUL(cr3, ci3, LD_PS1(wr2), LD_PS1(wi2));
        wr3=wa3[i]; wi3=fsign*wa3[i+1];
        VST(ch, i + 2*l1ido, cr3);
        VST(ch, i + 2*l1ido + 1, ci3);

        VCPLXMUL(cr4, ci4, LD_PS1(wr3), LD_PS1(wi3));
        VST(ch, i + 3*l1ido, cr4);
        VST(ch, i + 3*l1ido + 1, ci4);
      }
    }
  }
} /* passf4 */

/*
  passf5 and passb5 has been merged here, fsign = -1 for passf5, +1 for passb5
*/
static PF_CODELET(void) PF_CODELET_NAME(passf5)(int ido, int l1, const v4sf * PF_CODELET_RESTRICT cc,
                                                v4sf * PF_CODELET_RESTRICT ch,
                                                const float *wa1, const float *wa2,
                                                const float *wa3, const float *wa4, float fsign) {
  static const float tr11 = (float)0.30901699437494742410;   /* cos(2*pi/5) = (sqrt(5)-1)/4 */
  const float ti11 = (float)0.95105651629515357212*fsign;    /* sin(2*pi/5) */
  static const float tr12 = (float)-0.80901699437494742410;  /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
  const float ti12 = (float)0.58778525229247312917*fsign;    /* sin(4*pi/5) */

  /* Local variables */
  int i, k;
  v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
    ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

  float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

#define cc_idx(a_1,a_2) ((a_2-1)*ido + a_1 + 1)
#define ch_idx(a_1,a_3) ((a_3-1)*l1*ido + a_1 + 1)
/* leg 1, read again after the first two stores; at l1 == 1 these store to
   the positions of leg 1, so in place it is loaded before them */
#ifdef PF_INPLACE_CODELETS
#  define cc_leg1r c1r
#  define cc_leg1i c1i
#else
#  define cc_leg1r VLD(cc, cc_idx(i-1, 1))
#  define cc_leg1i VLD(cc, cc_idx(i  , 1))
#endif

  assert(ido >= 2);  /* ido == 2 (one complex element) is reached when SIMD_SZ == 1 */
  for (k = 0; k < l1; ++k, cc += 5*ido, ch += ido) {
    for (i = 0; i < ido-1; i += 2) {
      ti5 = VSUB(VLD(cc, cc_idx(i  , 2)), VLD(cc, cc_idx(i  , 5)));
      ti2 = VADD(VLD(cc, cc_idx(i  , 2)), VLD(cc, cc_idx(i  , 5)));
      ti4 = VSUB(VLD(cc, cc_idx(i  , 3)), VLD(cc, cc_idx(i  , 4)));
      ti3 = VADD(VLD(cc, cc_idx(i  , 3)), VLD(cc, cc_idx(i  , 4)));
      tr5 = VSUB(VLD(cc, cc_idx(i-1, 2)), VLD(cc, cc_idx(i-1, 5)));
      tr2 = VADD(VLD(cc, cc_idx(i-1, 2)), VLD(cc, cc_idx(i-1, 5)));
      tr4 = VSUB(VLD(cc, cc_idx(i-1, 3)), VLD(cc, cc_idx(i-1, 4)));
      tr3 = VADD(VLD(cc, cc_idx(i-1, 3)), VLD(cc, cc_idx(i-1, 4)));
#ifdef PF_INPLACE_CODELETS
      const v4sf c1r = VLD(cc, cc_idx(i-1, 1)), c1i = VLD(cc, cc_idx(i  , 1));
#endif
      VST(ch, ch_idx(i-1, 1), VADD(VLD(cc, cc_idx(i-1, 1)), VADD(tr2, tr3)));
      VST(ch, ch_idx(i  , 1), VADD(VLD(cc, cc_idx(i  , 1)), VADD(ti2, ti3)));
      cr2 = VADD(cc_leg1r, SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
      ci2 = VADD(cc_leg1i, SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
      cr3 = VADD(cc_leg1r, SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
      ci3 = VADD(cc_leg1i, SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
      cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
      ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
      cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
      ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
      dr3 = VSUB(cr3, ci4);
      dr4 = VADD(cr3, ci4);
      di3 = VADD(ci3, cr4);
      di4 = VSUB(ci3, cr4);
      dr5 = VADD(cr2, ci5);
      dr2 = VSUB(cr2, ci5);
      di5 = VSUB(ci2, cr5);
      di2 = VADD(ci2, cr5);
      wr1=wa1[i]; wi1=fsign*wa1[i+1]; wr2=wa2[i]; wi2=fsign*wa2[i+1];
      wr3=wa3[i]; wi3=fsign*wa3[i+1]; wr4=wa4[i]; wi4=fsign*wa4[i+1];
      VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
      VST(ch, ch_idx(i - 1, 2), dr2);
      VST(ch, ch_idx(i, 2), di2);
      VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
      VST(ch, ch_idx(i - 1, 3), dr3);
      VST(ch, ch_idx(i, 3), di3);
      VCPLXMUL(dr4, di4, LD_PS1(wr3), LD_PS1(wi3));
      VST(ch, ch_idx(i - 1, 4), dr4);
      VST(ch, ch_idx(i, 4), di4);
      VCPLXMUL(dr5, di5, LD_PS1(wr4), LD_PS1(wi4));
      VST(ch, ch_idx(i - 1, 5), dr5);
      VST(ch, ch_idx(i, 5), di5);
    }
  }
#undef cc_leg1i
#undef cc_leg1r
#undef ch_idx
#undef cc_idx
}

#ifndef PF_INPLACE_CODELETS
#if PFFFT_TERMINAL_RADIX8
/*
  passf8 and passb8 merged here, fsign = -1 for passf8, +1 for passb8.

  The radix-8 pass of terminal_radix8_plan: the last pass of a complex
  transform, with one complex element per leg (ido == 2) and therefore no
  twiddles, for any l1. It replaces the plan's radix-2 pass and its last
  radix-4 pass: the radix-2 pass is dropped and the last radix-4 pass
  becomes this one, so every pass in between runs with other l1 and ido;
  the transform has one pass over the data fewer.

  The radix-8 DFT is evaluated half by half: the even outputs are the DFT-4
  of (x_j + x_{j+4}), the odd ones the DFT-4 of (x_j - x_{j+4}) w8^j. The
  inputs are loaded again for the odd half rather than kept in registers,
  so that at most about twelve vector values are live (x86-64 has sixteen
  vector registers); cc and ch are deliberately not 'restrict', which makes
  the compiler reload them after the first half's stores. The 1/sqrt(2) of
  w8 and w8^3 is folded into the last butterfly as a multiply-add.

  The backward transform is the forward one with real and imaginary parts
  exchanged on input and output: R, I are the vector offsets of the parts
  the arithmetic treats as real and imaginary (0,1 forward; 1,0 backward),
  and ST(m, a, b) stores output m, a being the part read at offset R.
  k8 is the vector 1/sqrt(2) in the caller.
*/
#define PF_PASS8(cc, R, I, ST)                                                                 \
  {                                                                                             \
    v4sf s0r = VADD(VLD(cc, 0+R), VLD(cc, 8+R)),  s0i = VADD(VLD(cc, 0+I), VLD(cc, 8+I));       \
    v4sf s1r = VADD(VLD(cc, 2+R), VLD(cc, 10+R)), s1i = VADD(VLD(cc, 2+I), VLD(cc, 10+I));      \
    v4sf s2r = VADD(VLD(cc, 4+R), VLD(cc, 12+R)), s2i = VADD(VLD(cc, 4+I), VLD(cc, 12+I));      \
    v4sf s3r = VADD(VLD(cc, 6+R), VLD(cc, 14+R)), s3i = VADD(VLD(cc, 6+I), VLD(cc, 14+I));      \
    v4sf t1r = VSUB(s0r, s2r), t1i = VSUB(s0i, s2i);                                            \
    v4sf t2r = VADD(s0r, s2r), t2i = VADD(s0i, s2i);                                            \
    v4sf t3r = VADD(s1r, s3r), t3i = VADD(s1i, s3i);                                            \
    v4sf t4r = VSUB(s1i, s3i), t4i = VSUB(s3r, s1r);                                            \
    ST(0, VADD(t2r, t3r), VADD(t2i, t3i))                                                       \
    ST(2, VADD(t1r, t4r), VADD(t1i, t4i))                                                       \
    ST(4, VSUB(t2r, t3r), VSUB(t2i, t3i))                                                       \
    ST(6, VSUB(t1r, t4r), VSUB(t1i, t4i))                                                       \
  }                                                                                             \
  {                                                                                             \
    v4sf d0r = VSUB(VLD(cc, 0+R), VLD(cc, 8+R)),  d0i = VSUB(VLD(cc, 0+I), VLD(cc, 8+I));       \
    v4sf d1r = VSUB(VLD(cc, 2+R), VLD(cc, 10+R)), d1i = VSUB(VLD(cc, 2+I), VLD(cc, 10+I));      \
    v4sf d2r = VSUB(VLD(cc, 4+R), VLD(cc, 12+R)), d2i = VSUB(VLD(cc, 4+I), VLD(cc, 12+I));      \
    v4sf d3r = VSUB(VLD(cc, 6+R), VLD(cc, 14+R)), d3i = VSUB(VLD(cc, 6+I), VLD(cc, 14+I));      \
    v4sf p1r = VADD(d1r, d1i), p1i = VSUB(d1i, d1r);                                            \
    v4sf s3_ = VADD(d3r, d3i), p3r = VSUB(d3i, d3r);                                            \
    v4sf t1r = VSUB(d0r, d2i), t1i = VADD(d0i, d2r);                                            \
    v4sf t2r = VADD(d0r, d2i), t2i = VSUB(d0i, d2r);                                            \
    v4sf q3r = VADD(p1r, p3r), q3i = VSUB(p1i, s3_);                                            \
    v4sf er_ = VSUB(p1r, p3r), ei_ = VADD(p1i, s3_);                                            \
    ST(1, VMADD(k8, q3r, t2r), VMADD(k8, q3i, t2i))                                             \
    ST(3, VMADD(k8, ei_, t1r), VMSUB(k8, er_, t1i))                                             \
    ST(5, VMSUB(k8, q3r, t2r), VMSUB(k8, q3i, t2i))                                             \
    ST(7, VMSUB(k8, ei_, t1r), VMADD(k8, er_, t1i))                                             \
  }

static NEVER_INLINE(void) passf8_ps(int ido, int l1, const v4sf *cc, v4sf *ch, float fsign) {
  const float c8 = (float)M_SQRT1_2;   /* cos(pi/4) = sin(pi/4) = 1/sqrt(2) */
  const v4sf k8 = LD_PS1(c8);          /* LD_PS1 may take its argument's address */
  const int l1ido = l1*ido;
  int k;
  assert(ido == 2);
  if (fsign < 0) {
#define PF_ST8(m, a, b) { VST(ch, (m)*l1ido, a); VST(ch, (m)*l1ido + 1, b); }
    for (k=0; k < l1ido; k += ido, ch += ido, cc += 8*ido) {
      PF_PASS8(cc, 0, 1, PF_ST8)
    }
#undef PF_ST8
  } else {
#define PF_ST8(m, a, b) { VST(ch, (m)*l1ido + 1, a); VST(ch, (m)*l1ido, b); }
    for (k=0; k < l1ido; k += ido, ch += ido, cc += 8*ido) {
      PF_PASS8(cc, 1, 0, PF_ST8)
    }
#undef PF_ST8
  }
} /* passf8 */
#else  /* PFFFT_TERMINAL_RADIX8 */
/*
  passf8 and passb8 merged here, fsign = -1 for passf8, +1 for passb8

  Radix-8 DFT via two radix-4 DFTs:
    even outputs from DFT-4 of (x0+x4, x1+x5, x2+x6, x3+x7)
    odd  outputs from DFT-4 of (x0-x4, x1-x5, x2-x6, x3-x7)*W8^k
*/
/*
  cffti1_ps only ever selects factor 8 when n == 8 (the Ncvec == 8,
  N == 32 case); at the sole dispatch site below this forces ido == 2
  and l1 == 1. This codelet is therefore specialized to that single
  reachable shape instead of the general ido/l1 loop: the assert
  enforces the invariant at the point of use.
*/
static NEVER_INLINE(void) passf8_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch, float fsign) {
  static const float K8 = (float)M_SQRT1_2;   /* cos(pi/4) = sin(pi/4) = 1/sqrt(2) */
  const float K8s = fsign * K8;
  const float nK8 = -K8;
  const float nfsign = -fsign;

  assert(ido == 2 && l1 == 1);
  (void)l1; /* otherwise unused when assertions are disabled */

  /* Load 8 inputs: cc[leg*ido + 0] = real, cc[leg*ido + 1] = imag */
  v4sf x0r = VLD(cc, 0*ido+0), x0i = VLD(cc, 0*ido+1);
  v4sf x1r = VLD(cc, 1*ido+0), x1i = VLD(cc, 1*ido+1);
  v4sf x2r = VLD(cc, 2*ido+0), x2i = VLD(cc, 2*ido+1);
  v4sf x3r = VLD(cc, 3*ido+0), x3i = VLD(cc, 3*ido+1);
  v4sf x4r = VLD(cc, 4*ido+0), x4i = VLD(cc, 4*ido+1);
  v4sf x5r = VLD(cc, 5*ido+0), x5i = VLD(cc, 5*ido+1);
  v4sf x6r = VLD(cc, 6*ido+0), x6i = VLD(cc, 6*ido+1);
  v4sf x7r = VLD(cc, 7*ido+0), x7i = VLD(cc, 7*ido+1);

  /* Step 1: sums and diffs at stride 4 */
  v4sf s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
  v4sf s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
  v4sf s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
  v4sf s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
  v4sf d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
  v4sf d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
  v4sf d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
  v4sf d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);

  /* Step 2: twiddle diffs by W8^k */
  /* d0 *= W8^0 = 1 (no-op) */
  /* d1 *= W8^1: VCPLXMUL with (K8, fsign*K8) */
  VCPLXMUL(d1r, d1i, LD_PS1(K8), LD_PS1(K8s));
  /* d2 *= W8^2 = -j rotation */
  { v4sf tr = VMUL(LD_PS1(nfsign), d2i);
    v4sf ti = VMUL(LD_PS1(fsign), d2r);
    d2r = tr; d2i = ti; }
  /* d3 *= W8^3: VCPLXMUL with (-K8, fsign*K8) */
  VCPLXMUL(d3r, d3i, LD_PS1(nK8), LD_PS1(K8s));

  /* Step 3: DFT-4 on sums -> even outputs (0,2,4,6) */
  {
    v4sf t1r = VSUB(s0r, s2r), t1i = VSUB(s0i, s2i);
    v4sf t2r = VADD(s0r, s2r), t2i = VADD(s0i, s2i);
    v4sf t4i = VMUL(VSUB(s1r, s3r), LD_PS1(fsign));
    v4sf t4r = VMUL(VSUB(s3i, s1i), LD_PS1(fsign));
    v4sf t3r = VADD(s1r, s3r), t3i = VADD(s1i, s3i);

    VST(ch, 0*ido + 0, VADD(t2r, t3r));
    VST(ch, 0*ido + 1, VADD(t2i, t3i));
    VST(ch, 2*ido + 0, VADD(t1r, t4r));
    VST(ch, 2*ido + 1, VADD(t1i, t4i));
    VST(ch, 4*ido + 0, VSUB(t2r, t3r));
    VST(ch, 4*ido + 1, VSUB(t2i, t3i));
    VST(ch, 6*ido + 0, VSUB(t1r, t4r));
    VST(ch, 6*ido + 1, VSUB(t1i, t4i));
  }

  /* Step 4: DFT-4 on diffs -> odd outputs (1,3,5,7) */
  {
    v4sf t1r = VSUB(d0r, d2r), t1i = VSUB(d0i, d2i);
    v4sf t2r = VADD(d0r, d2r), t2i = VADD(d0i, d2i);
    v4sf t4i = VMUL(VSUB(d1r, d3r), LD_PS1(fsign));
    v4sf t4r = VMUL(VSUB(d3i, d1i), LD_PS1(fsign));
    v4sf t3r = VADD(d1r, d3r), t3i = VADD(d1i, d3i);

    VST(ch, 1*ido + 0, VADD(t2r, t3r));
    VST(ch, 1*ido + 1, VADD(t2i, t3i));
    VST(ch, 3*ido + 0, VADD(t1r, t4r));
    VST(ch, 3*ido + 1, VADD(t1i, t4i));
    VST(ch, 5*ido + 0, VSUB(t2r, t3r));
    VST(ch, 5*ido + 1, VSUB(t2i, t3i));
    VST(ch, 7*ido + 0, VSUB(t1r, t4r));
    VST(ch, 7*ido + 1, VSUB(t1i, t4i));
  }
} /* passf8 */
#endif /* PFFFT_TERMINAL_RADIX8 */


static NEVER_INLINE(void) radf2_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch, const float *wa1) {
  static const float minus_one = -1.f;
  int i, k, l1ido = l1*ido;
  for (k=0; k < l1ido; k += ido) {
    v4sf a = VLD(cc, k), b = VLD(cc, k + l1ido);
    VST(ch, 2*k, VADD(a, b));
    VST(ch, 2*(k+ido)-1, VSUB(a, b));
  }
  if (ido < 2) return;
  if (ido != 2) {
    for (k=0; k < l1ido; k += ido) {
      for (i=2; i<ido; i+=2) {
        v4sf tr2 = VLD(cc, i - 1 + k + l1ido), ti2 = VLD(cc, i + k + l1ido);
        v4sf br = VLD(cc, i - 1 + k), bi = VLD(cc, i + k);
        VCPLXMULCONJ(tr2, ti2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
        VST(ch, i + 2*k, VADD(bi, ti2));
        VST(ch, 2*(k+ido) - i, VSUB(ti2, bi));
        VST(ch, i - 1 + 2*k, VADD(br, tr2));
        VST(ch, 2*(k+ido) - i -1, VSUB(br, tr2));
      }
    }
    if (ido % 2 != 0) return;
  }
  for (k=0; k < l1ido; k += ido) {
    VST(ch, 2*k + ido, SVMUL(minus_one, VLD(cc, ido-1 + k + l1ido)));
    VST(ch, 2*k + ido-1, VLD(cc, k + ido-1));
  }
} /* radf2 */


static NEVER_INLINE(void) radb2_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch, const float *wa1) {
  static const float minus_two=-2;
  int i, k, l1ido = l1*ido;
  v4sf a,b,c,d, tr2, ti2;
  for (k=0; k < l1ido; k += ido) {
    a = VLD(cc, 2*k); b = VLD(cc, 2*(k+ido) - 1);
    VST(ch, k, VADD(a, b));
    VST(ch, k + l1ido, VSUB(a, b));
  }
  if (ido < 2) return;
  if (ido != 2) {
    for (k = 0; k < l1ido; k += ido) {
      for (i = 2; i < ido; i += 2) {
        a = VLD(cc, i-1 + 2*k); b = VLD(cc, 2*(k + ido) - i - 1);
        c = VLD(cc, i+0 + 2*k); d = VLD(cc, 2*(k + ido) - i + 0);
        VST(ch, i-1 + k, VADD(a, b));
        tr2 = VSUB(a, b);
        VST(ch, i+0 + k, VSUB(c, d));
        ti2 = VADD(c, d);
        VCPLXMUL(tr2, ti2, LD_PS1(wa1[i - 2]), LD_PS1(wa1[i - 1]));
        VST(ch, i-1 + k + l1ido, tr2);
        VST(ch, i+0 + k + l1ido, ti2);
      }
    }
    if (ido % 2 != 0) return;
  }
  for (k = 0; k < l1ido; k += ido) {
    a = VLD(cc, 2*k + ido-1); b = VLD(cc, 2*k + ido);
    VST(ch, k + ido-1, VADD(a,a));
    VST(ch, k + ido-1 + l1ido, SVMUL(minus_two, b));
  }
} /* radb2 */

static void radf3_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch,
                     const float *wa1, const float *wa2) {
  static const float taur = (float)-0.5;                     /* cos(2*pi/3)   */
  static const float taui = (float)0.86602540378443864676;   /* sin(2*pi/3) = sqrt(3)/2 */
  int i, k, ic;
  v4sf ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3, wr1, wi1, wr2, wi2;
  for (k=0; k<l1; k++) {
    cr2 = VADD(VLD(cc, (k + l1)*ido), VLD(cc, (k + 2*l1)*ido));
    VST(ch, 3*k*ido, VADD(VLD(cc, k*ido), cr2));
    VST(ch, (3*k+2)*ido, SVMUL(taui, VSUB(VLD(cc, (k + l1*2)*ido), VLD(cc, (k + l1)*ido))));
    VST(ch, ido-1 + (3*k + 1)*ido, SVMADD(taur, cr2, VLD(cc, k*ido)));
  }
  if (ido == 1) return;
  if (ido == 3) {
    /* Fast path: inner loop executes exactly once with i=2, ic=1 */
    wr1 = LD_PS1(wa1[0]); wi1 = LD_PS1(wa1[1]);
    wr2 = LD_PS1(wa2[0]); wi2 = LD_PS1(wa2[1]);
    for (k=0; k<l1; k++) {
      dr2 = VLD(cc, 1 + (k + l1)*ido); di2 = VLD(cc, 2 + (k + l1)*ido);
      VCPLXMULCONJ(dr2, di2, wr1, wi1);
      dr3 = VLD(cc, 1 + (k + l1*2)*ido); di3 = VLD(cc, 2 + (k + l1*2)*ido);
      VCPLXMULCONJ(dr3, di3, wr2, wi2);
      cr2 = VADD(dr2, dr3);
      ci2 = VADD(di2, di3);
      VST(ch, 1 + 3*k*ido, VADD(VLD(cc, 1 + k*ido), cr2));
      VST(ch, 2 + 3*k*ido, VADD(VLD(cc, 2 + k*ido), ci2));
      tr2 = SVMADD(taur, cr2, VLD(cc, 1 + k*ido));
      ti2 = SVMADD(taur, ci2, VLD(cc, 2 + k*ido));
      tr3 = SVMUL(taui, VSUB(di2, di3));
      ti3 = SVMUL(taui, VSUB(dr3, dr2));
      VST(ch, 1 + (3*k + 2)*ido, VADD(tr2, tr3));
      VST(ch, 0 + (3*k + 1)*ido, VSUB(tr2, tr3));
      VST(ch, 2 + (3*k + 2)*ido, VADD(ti2, ti3));
      VST(ch, 1 + (3*k + 1)*ido, VSUB(ti3, ti2));
    }
    return;
  }
  for (k=0; k<l1; k++) {
    for (i=2; i<ido; i+=2) {
      ic = ido - i;
      wr1 = LD_PS1(wa1[i - 2]); wi1 = LD_PS1(wa1[i - 1]);
      dr2 = VLD(cc, i - 1 + (k + l1)*ido); di2 = VLD(cc, i + (k + l1)*ido);
      VCPLXMULCONJ(dr2, di2, wr1, wi1);

      wr2 = LD_PS1(wa2[i - 2]); wi2 = LD_PS1(wa2[i - 1]);
      dr3 = VLD(cc, i - 1 + (k + l1*2)*ido); di3 = VLD(cc, i + (k + l1*2)*ido);
      VCPLXMULCONJ(dr3, di3, wr2, wi2);

      cr2 = VADD(dr2, dr3);
      ci2 = VADD(di2, di3);
      VST(ch, i - 1 + 3*k*ido, VADD(VLD(cc, i - 1 + k*ido), cr2));
      VST(ch, i + 3*k*ido, VADD(VLD(cc, i + k*ido), ci2));
      tr2 = SVMADD(taur, cr2, VLD(cc, i - 1 + k*ido));
      ti2 = SVMADD(taur, ci2, VLD(cc, i + k*ido));
      tr3 = SVMUL(taui, VSUB(di2, di3));
      ti3 = SVMUL(taui, VSUB(dr3, dr2));
      VST(ch, i - 1 + (3*k + 2)*ido, VADD(tr2, tr3));
      VST(ch, ic - 1 + (3*k + 1)*ido, VSUB(tr2, tr3));
      VST(ch, i + (3*k + 2)*ido, VADD(ti2, ti3));
      VST(ch, ic + (3*k + 1)*ido, VSUB(ti3, ti2));
    }
  }
} /* radf3 */


static void radb3_ps(int ido, int l1, const v4sf *RESTRICT cc, v4sf *RESTRICT ch,
                     const float *wa1, const float *wa2)
{
  static const float taur = (float)-0.5;                     /* cos(2*pi/3)   */
  static const float taui = (float)0.86602540378443864676;   /* sin(2*pi/3) = sqrt(3)/2 */
  static const float taui_2 = (float)1.7320508075688772935;  /* 2*sin(2*pi/3) = sqrt(3) */
  int i, k, ic;
  v4sf ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
  for (k=0; k<l1; k++) {
    tr2 = VLD(cc, ido-1 + (3*k + 1)*ido); tr2 = VADD(tr2,tr2);
    cr2 = SVMADD(taur, tr2, VLD(cc, 3*k*ido));
    VST(ch, k*ido, VADD(VLD(cc, 3*k*ido), tr2));
    ci3 = SVMUL(taui_2, VLD(cc, (3*k + 2)*ido));
    VST(ch, (k + l1)*ido, VSUB(cr2, ci3));
    VST(ch, (k + 2*l1)*ido, VADD(cr2, ci3));
  }
  if (ido == 1) return;
  if (ido == 3) {
    /* Fast path: inner loop executes exactly once with i=2, ic=1 */
    v4sf wa1_r = LD_PS1(wa1[0]), wa1_i = LD_PS1(wa1[1]);
    v4sf wa2_r = LD_PS1(wa2[0]), wa2_i = LD_PS1(wa2[1]);
    for (k=0; k<l1; k++) {
      tr2 = VADD(VLD(cc, 1 + (3*k + 2)*ido), VLD(cc, 0 + (3*k + 1)*ido));
      cr2 = SVMADD(taur, tr2, VLD(cc, 1 + 3*k*ido));
      VST(ch, 1 + k*ido, VADD(VLD(cc, 1 + 3*k*ido), tr2));
      ti2 = VSUB(VLD(cc, 2 + (3*k + 2)*ido), VLD(cc, 1 + (3*k + 1)*ido));
      ci2 = SVMADD(taur, ti2, VLD(cc, 2 + 3*k*ido));
      VST(ch, 2 + k*ido, VADD(VLD(cc, 2 + 3*k*ido), ti2));
      cr3 = SVMUL(taui, VSUB(VLD(cc, 1 + (3*k + 2)*ido), VLD(cc, 0 + (3*k + 1)*ido)));
      ci3 = SVMUL(taui, VADD(VLD(cc, 2 + (3*k + 2)*ido), VLD(cc, 1 + (3*k + 1)*ido)));
      dr2 = VSUB(cr2, ci3);
      dr3 = VADD(cr2, ci3);
      di2 = VADD(ci2, cr3);
      di3 = VSUB(ci2, cr3);
      VCPLXMUL(dr2, di2, wa1_r, wa1_i);
      VST(ch, 1 + (k + l1)*ido, dr2);
      VST(ch, 2 + (k + l1)*ido, di2);
      VCPLXMUL(dr3, di3, wa2_r, wa2_i);
      VST(ch, 1 + (k + 2*l1)*ido, dr3);
      VST(ch, 2 + (k + 2*l1)*ido, di3);
    }
    return;
  }
  for (k=0; k<l1; k++) {
    for (i=2; i<ido; i+=2) {
      ic = ido - i;
      tr2 = VADD(VLD(cc, i - 1 + (3*k + 2)*ido), VLD(cc, ic - 1 + (3*k + 1)*ido));
      cr2 = SVMADD(taur, tr2, VLD(cc, i - 1 + 3*k*ido));
      VST(ch, i - 1 + k*ido, VADD(VLD(cc, i - 1 + 3*k*ido), tr2));
      ti2 = VSUB(VLD(cc, i + (3*k + 2)*ido), VLD(cc, ic + (3*k + 1)*ido));
      ci2 = SVMADD(taur, ti2, VLD(cc, i + 3*k*ido));
      VST(ch, i + k*ido, VADD(VLD(cc, i + 3*k*ido), ti2));
      cr3 = SVMUL(taui, VSUB(VLD(cc, i - 1 + (3*k + 2)*ido), VLD(cc, ic - 1 + (3*k + 1)*ido)));
      ci3 = SVMUL(taui, VADD(VLD(cc, i + (3*k + 2)*ido), VLD(cc, ic + (3*k + 1)*ido)));
      dr2 = VSUB(cr2, ci3);
      dr3 = VADD(cr2, ci3);
      di2 = VADD(ci2, cr3);
      di3 = VSUB(ci2, cr3);
      VCPLXMUL(dr2, di2, LD_PS1(wa1[i-2]), LD_PS1(wa1[i-1]));
      VST(ch, i - 1 + (k + l1)*ido, dr2);
      VST(ch, i + (k + l1)*ido, di2);
      VCPLXMUL(dr3, di3, LD_PS1(wa2[i-2]), LD_PS1(wa2[i-1]));
      VST(ch, i - 1 + (k + 2*l1)*ido, dr3);
      VST(ch, i + (k + 2*l1)*ido, di3);
    }
  }
} /* radb3 */

static NEVER_INLINE(void) radf4_ps(int ido, int l1, const v4sf *RESTRICT cc, v4sf * RESTRICT ch,
                                   const float * RESTRICT wa1, const float * RESTRICT wa2, const float * RESTRICT wa3)
{
  static const float minus_hsqt2 = (float)-M_SQRT1_2;  /* -sin(pi/4) = -1/sqrt(2) */
  int i, k, l1ido = l1*ido;
  {
    const v4sf *RESTRICT cc_ = cc, * RESTRICT cc_end = cc + l1ido;
    v4sf * RESTRICT ch_ = ch;
    while (cc < cc_end) {
      /* this loop represents between 25% and 40% of total radf4_ps cost ! */
      v4sf a0 = VLD(cc, 0), a1 = VLD(cc, l1ido);
      v4sf a2 = VLD(cc, 2*l1ido), a3 = VLD(cc, 3*l1ido);
      v4sf tr1 = VADD(a1, a3);
      v4sf tr2 = VADD(a0, a2);
      VST(ch, 2*ido-1, VSUB(a0, a2));
      VST(ch, 2*ido, VSUB(a3, a1));
      VST(ch, 0, VADD(tr1, tr2));
      VST(ch, 4*ido-1, VSUB(tr2, tr1));
      cc += ido; ch += 4*ido;
    }
    cc = cc_; ch = ch_;
  }
  if (ido < 2) return;
  if (ido == 3) {
    /* Fast path: inner loop executes exactly once with i=2, ic=ido-i=1 */
    v4sf wr, wi, cr2, ci2, cr3, ci3, cr4, ci4;
    v4sf tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;
    for (k = 0; k < l1ido; k += ido) {
      cr2 = VLD(cc, 1 + k + 1*l1ido);
      ci2 = VLD(cc, 1 + k + 1*l1ido + 1);
      wr = LD_PS1(wa1[0]); wi = LD_PS1(wa1[1]);
      VCPLXMULCONJ(cr2, ci2, wr, wi);

      cr3 = VLD(cc, 1 + k + 2*l1ido);
      ci3 = VLD(cc, 1 + k + 2*l1ido + 1);
      wr = LD_PS1(wa2[0]); wi = LD_PS1(wa2[1]);
      VCPLXMULCONJ(cr3, ci3, wr, wi);

      cr4 = VLD(cc, 1 + k + 3*l1ido);
      ci4 = VLD(cc, 1 + k + 3*l1ido + 1);
      wr = LD_PS1(wa3[0]); wi = LD_PS1(wa3[1]);
      VCPLXMULCONJ(cr4, ci4, wr, wi);

      tr1 = VADD(cr2, cr4);
      tr4 = VSUB(cr4, cr2);
      tr2 = VADD(VLD(cc, 1 + k), cr3);
      tr3 = VSUB(VLD(cc, 1 + k), cr3);
      VST(ch, 1 + 4*k, VADD(tr1, tr2));
      VST(ch, 0 + 4*k + 3*ido, VSUB(tr2, tr1));
      ti1 = VADD(ci2, ci4);
      ti4 = VSUB(ci2, ci4);
      VST(ch, 1 + 4*k + 2*ido, VADD(ti4, tr3));
      VST(ch, 0 + 4*k + 1*ido, VSUB(tr3, ti4));
      ti2 = VADD(VLD(cc, 1 + k + 1), ci3);
      ti3 = VSUB(VLD(cc, 1 + k + 1), ci3);
      VST(ch, 2 + 4*k, VADD(ti1, ti2));
      VST(ch, 1 + 4*k + 3*ido, VSUB(ti1, ti2));
      VST(ch, 2 + 4*k + 2*ido, VADD(tr4, ti3));
      VST(ch, 1 + 4*k + 1*ido, VSUB(tr4, ti3));
    }
    return;
  }
  if (ido != 2) {
    for (k = 0; k < l1ido; k += ido) {
      const v4sf * RESTRICT pc = (v4sf*)(cc + 1 + k);
      for (i=2; i<ido; i += 2, pc += 2) {
        int ic = ido - i;
        v4sf wr, wi, cr2, ci2, cr3, ci3, cr4, ci4;
        v4sf tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;

        cr2 = VLD(pc, 1*l1ido+0);
        ci2 = VLD(pc, 1*l1ido+1);
        wr=LD_PS1(wa1[i - 2]);
        wi=LD_PS1(wa1[i - 1]);
        VCPLXMULCONJ(cr2,ci2,wr,wi);

        cr3 = VLD(pc, 2*l1ido+0);
        ci3 = VLD(pc, 2*l1ido+1);
        wr = LD_PS1(wa2[i-2]);
        wi = LD_PS1(wa2[i-1]);
        VCPLXMULCONJ(cr3, ci3, wr, wi);

        cr4 = VLD(pc, 3*l1ido);
        ci4 = VLD(pc, 3*l1ido+1);
        wr = LD_PS1(wa3[i-2]);
        wi = LD_PS1(wa3[i-1]);
        VCPLXMULCONJ(cr4, ci4, wr, wi);

        /* at this point, on SSE, five of "cr2 cr3 cr4 ci2 ci3 ci4" should be loaded in registers */

        tr1 = VADD(cr2,cr4);
        tr4 = VSUB(cr4,cr2);
        tr2 = VADD(VLD(pc, 0),cr3);
        tr3 = VSUB(VLD(pc, 0),cr3);
        VST(ch, i - 1 + 4*k, VADD(tr1,tr2));
        VST(ch, ic - 1 + 4*k + 3*ido, VSUB(tr2,tr1)); /* at this point tr1 and tr2 can be disposed */
        ti1 = VADD(ci2,ci4);
        ti4 = VSUB(ci2,ci4);
        VST(ch, i - 1 + 4*k + 2*ido, VADD(ti4,tr3));
        VST(ch, ic - 1 + 4*k + 1*ido, VSUB(tr3,ti4)); /* dispose tr3, ti4 */
        ti2 = VADD(VLD(pc, 1),ci3);
        ti3 = VSUB(VLD(pc, 1),ci3);
        VST(ch, i + 4*k, VADD(ti1, ti2));
        VST(ch, ic + 4*k + 3*ido, VSUB(ti1, ti2));
        VST(ch, i + 4*k + 2*ido, VADD(tr4, ti3));
        VST(ch, ic + 4*k + 1*ido, VSUB(tr4, ti3));
      }
    }
    if (ido % 2 != 0) return;
  }
  for (k=0; k<l1ido; k += ido) {
    v4sf a = VLD(cc, ido-1 + k + l1ido), b = VLD(cc, ido-1 + k + 3*l1ido);
    v4sf c = VLD(cc, ido-1 + k), d = VLD(cc, ido-1 + k + 2*l1ido);
    v4sf ti1 = SVMUL(minus_hsqt2, VADD(a, b));
    v4sf tr1 = SVMUL(minus_hsqt2, VSUB(b, a));
    VST(ch, ido-1 + 4*k, VADD(tr1, c));
    VST(ch, ido-1 + 4*k + 2*ido, VSUB(c, tr1));
    VST(ch, 4*k + 1*ido, VSUB(ti1, d));
    VST(ch, 4*k + 3*ido, VADD(ti1, d));
  }
} /* radf4 */


static NEVER_INLINE(void) radb4_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch,
                                   const float * RESTRICT wa1, const float * RESTRICT wa2, const float *RESTRICT wa3)
{
  static const float minus_sqrt2 = (float)-M_SQRT2;     /* -2*sin(pi/4) = -sqrt(2) */
  static const float two = 2.f;
  int i, k, l1ido = l1*ido;
  v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
  {
    const v4sf *RESTRICT cc_ = cc, * RESTRICT ch_end = ch + l1ido;
    v4sf *ch_ = ch;
    while (ch < ch_end) {
      v4sf a = VLD(cc, 0), b = VLD(cc, 4*ido-1);
      v4sf c = VLD(cc, 2*ido), d = VLD(cc, 2*ido-1);
      tr3 = SVMUL(two,d);
      tr2 = VADD(a,b);
      tr1 = VSUB(a,b);
      tr4 = SVMUL(two,c);
      VST(ch, 0*l1ido, VADD(tr2, tr3));
      VST(ch, 2*l1ido, VSUB(tr2, tr3));
      VST(ch, 1*l1ido, VSUB(tr1, tr4));
      VST(ch, 3*l1ido, VADD(tr1, tr4));

      cc += 4*ido; ch += ido;
    }
    cc = cc_; ch = ch_;
  }
  if (ido < 2) return;
  if (ido == 3) {
    /* Fast path: inner loop executes exactly once with i=2, ic=ido-i=1 */
    v4sf wa1_r = LD_PS1(wa1[0]), wa1_i = LD_PS1(wa1[1]);
    v4sf wa2_r = LD_PS1(wa2[0]), wa2_i = LD_PS1(wa2[1]);
    v4sf wa3_r = LD_PS1(wa3[0]), wa3_i = LD_PS1(wa3[1]);
    for (k = 0; k < l1ido; k += ido) {
      tr1 = VSUB(VLD(cc, 1 + 4*k), VLD(cc, 9 + 4*k));
      tr2 = VADD(VLD(cc, 1 + 4*k), VLD(cc, 9 + 4*k));
      ti4 = VSUB(VLD(cc, 7 + 4*k), VLD(cc, 3 + 4*k));
      tr3 = VADD(VLD(cc, 7 + 4*k), VLD(cc, 3 + 4*k));
      VST(ch, k + 1, VADD(tr2, tr3));
      cr3 = VSUB(tr2, tr3);

      ti3 = VSUB(VLD(cc, 8 + 4*k), VLD(cc, 4 + 4*k));
      tr4 = VADD(VLD(cc, 8 + 4*k), VLD(cc, 4 + 4*k));
      cr2 = VSUB(tr1, tr4);
      cr4 = VADD(tr1, tr4);

      ti1 = VADD(VLD(cc, 2 + 4*k), VLD(cc, 10 + 4*k));
      ti2 = VSUB(VLD(cc, 2 + 4*k), VLD(cc, 10 + 4*k));

      VST(ch, k + 2, VADD(ti2, ti3));
      ci3 = VSUB(ti2, ti3);
      ci2 = VADD(ti1, ti4);
      ci4 = VSUB(ti1, ti4);
      VCPLXMUL(cr2, ci2, wa1_r, wa1_i);
      VST(ch, k + 1 + l1ido, cr2);
      VST(ch, k + 2 + l1ido, ci2);
      VCPLXMUL(cr3, ci3, wa2_r, wa2_i);
      VST(ch, k + 1 + 2*l1ido, cr3);
      VST(ch, k + 2 + 2*l1ido, ci3);
      VCPLXMUL(cr4, ci4, wa3_r, wa3_i);
      VST(ch, k + 1 + 3*l1ido, cr4);
      VST(ch, k + 2 + 3*l1ido, ci4);
    }
    return;
  }
  if (ido != 2) {
    for (k = 0; k < l1ido; k += ido) {
      const v4sf * RESTRICT pc = (v4sf*)(cc - 1 + 4*k);
      v4sf * RESTRICT ph = (v4sf*)(ch + k + 1);
      for (i = 2; i < ido; i += 2) {

        tr1 = VSUB(VLD(pc, i), VLD(pc, 4*ido - i));
        tr2 = VADD(VLD(pc, i), VLD(pc, 4*ido - i));
        ti4 = VSUB(VLD(pc, 2*ido + i), VLD(pc, 2*ido - i));
        tr3 = VADD(VLD(pc, 2*ido + i), VLD(pc, 2*ido - i));
        VST(ph, 0, VADD(tr2, tr3));
        cr3 = VSUB(tr2, tr3);

        ti3 = VSUB(VLD(pc, 2*ido + i + 1), VLD(pc, 2*ido - i + 1));
        tr4 = VADD(VLD(pc, 2*ido + i + 1), VLD(pc, 2*ido - i + 1));
        cr2 = VSUB(tr1, tr4);
        cr4 = VADD(tr1, tr4);

        ti1 = VADD(VLD(pc, i + 1), VLD(pc, 4*ido - i + 1));
        ti2 = VSUB(VLD(pc, i + 1), VLD(pc, 4*ido - i + 1));

        VST(ph, 1, VADD(ti2, ti3)); ph += l1ido;
        ci3 = VSUB(ti2, ti3);
        ci2 = VADD(ti1, ti4);
        ci4 = VSUB(ti1, ti4);
        VCPLXMUL(cr2, ci2, LD_PS1(wa1[i-2]), LD_PS1(wa1[i-1]));
        VST(ph, 0, cr2);
        VST(ph, 1, ci2); ph += l1ido;
        VCPLXMUL(cr3, ci3, LD_PS1(wa2[i-2]), LD_PS1(wa2[i-1]));
        VST(ph, 0, cr3);
        VST(ph, 1, ci3); ph += l1ido;
        VCPLXMUL(cr4, ci4, LD_PS1(wa3[i-2]), LD_PS1(wa3[i-1]));
        VST(ph, 0, cr4);
        VST(ph, 1, ci4); ph = ph - 3*l1ido + 2;
      }
    }
    if (ido % 2 != 0) return;
  }
  for (k=0; k < l1ido; k+=ido) {
    int i0 = 4*k + ido;
    v4sf c = VLD(cc, i0-1), d = VLD(cc, i0 + 2*ido-1);
    v4sf a = VLD(cc, i0+0), b = VLD(cc, i0 + 2*ido+0);
    tr1 = VSUB(c,d);
    tr2 = VADD(c,d);
    ti1 = VADD(b,a);
    ti2 = VSUB(b,a);
    VST(ch, ido-1 + k + 0*l1ido, VADD(tr2,tr2));
    VST(ch, ido-1 + k + 1*l1ido, SVMUL(minus_sqrt2, VSUB(ti1, tr1)));
    VST(ch, ido-1 + k + 2*l1ido, VADD(ti2, ti2));
    VST(ch, ido-1 + k + 3*l1ido, SVMUL(minus_sqrt2, VADD(ti1, tr1)));
  }
} /* radb4 */

static void radf5_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch,
                     const float *wa1, const float *wa2, const float *wa3, const float *wa4)
{
  static const float tr11 = (float)0.30901699437494742410;   /* cos(2*pi/5) = (sqrt(5)-1)/4 */
  static const float ti11 = (float)0.95105651629515357212;   /* sin(2*pi/5) */
  static const float tr12 = (float)-0.80901699437494742410;  /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
  static const float ti12 = (float)0.58778525229247312917;   /* sin(4*pi/5) */

  /* System generated locals */
  int cc_offset, ch_offset;

  /* Local variables */
  int i, k, ic;
  v4sf ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5,
    cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
  int idp2;


#define cc_idx(a_1,a_2,a_3) (((a_3)*l1 + (a_2))*ido + a_1)
#define ch_idx(a_1,a_2,a_3) (((a_3)*5 + (a_2))*ido + a_1)

  /* Parameter adjustments */
  ch_offset = 1 + ido * 6;
  ch -= ch_offset;
  cc_offset = 1 + ido * (1 + l1);
  cc -= cc_offset;

  /* Function Body */
  for (k = 1; k <= l1; ++k) {
    cr2 = VADD(VLD(cc, cc_idx(1, k, 5)), VLD(cc, cc_idx(1, k, 2)));
    ci5 = VSUB(VLD(cc, cc_idx(1, k, 5)), VLD(cc, cc_idx(1, k, 2)));
    cr3 = VADD(VLD(cc, cc_idx(1, k, 4)), VLD(cc, cc_idx(1, k, 3)));
    ci4 = VSUB(VLD(cc, cc_idx(1, k, 4)), VLD(cc, cc_idx(1, k, 3)));
    VST(ch, ch_idx(1, 1, k), VADD(VLD(cc, cc_idx(1, k, 1)), VADD(cr2, cr3)));
    VST(ch, ch_idx(ido, 2, k), VADD(VLD(cc, cc_idx(1, k, 1)), SVMADD(tr11, cr2, SVMUL(tr12, cr3))));
    VST(ch, ch_idx(1, 3, k), SVMADD(ti11, ci5, SVMUL(ti12, ci4)));
    VST(ch, ch_idx(ido, 4, k), VADD(VLD(cc, cc_idx(1, k, 1)), SVMADD(tr12, cr2, SVMUL(tr11, cr3))));
    VST(ch, ch_idx(1, 5, k), SVMSUB(ti11, ci4, SVMUL(ti12, ci5)));
  }
  if (ido == 1) {
    return;
  }
  idp2 = ido + 2;
  for (k = 1; k <= l1; ++k) {
    for (i = 3; i <= ido; i += 2) {
      ic = idp2 - i;
      dr2 = LD_PS1(wa1[i-3]); di2 = LD_PS1(wa1[i-2]);
      dr3 = LD_PS1(wa2[i-3]); di3 = LD_PS1(wa2[i-2]);
      dr4 = LD_PS1(wa3[i-3]); di4 = LD_PS1(wa3[i-2]);
      dr5 = LD_PS1(wa4[i-3]); di5 = LD_PS1(wa4[i-2]);
      VCPLXMULCONJ(dr2, di2, VLD(cc, cc_idx(i-1, k, 2)), VLD(cc, cc_idx(i, k, 2)));
      VCPLXMULCONJ(dr3, di3, VLD(cc, cc_idx(i-1, k, 3)), VLD(cc, cc_idx(i, k, 3)));
      VCPLXMULCONJ(dr4, di4, VLD(cc, cc_idx(i-1, k, 4)), VLD(cc, cc_idx(i, k, 4)));
      VCPLXMULCONJ(dr5, di5, VLD(cc, cc_idx(i-1, k, 5)), VLD(cc, cc_idx(i, k, 5)));
      cr2 = VADD(dr2, dr5);
      ci5 = VSUB(dr5, dr2);
      cr5 = VSUB(di2, di5);
      ci2 = VADD(di2, di5);
      cr3 = VADD(dr3, dr4);
      ci4 = VSUB(dr4, dr3);
      cr4 = VSUB(di3, di4);
      ci3 = VADD(di3, di4);
      VST(ch, ch_idx(i - 1, 1, k), VADD(VLD(cc, cc_idx(i - 1, k, 1)), VADD(cr2, cr3)));
      VST(ch, ch_idx(i, 1, k), VSUB(VLD(cc, cc_idx(i, k, 1)), VADD(ci2, ci3)));
      tr2 = VADD(VLD(cc, cc_idx(i - 1, k, 1)), SVMADD(tr11, cr2, SVMUL(tr12, cr3)));
      ti2 = VSUB(VLD(cc, cc_idx(i, k, 1)), SVMADD(tr11, ci2, SVMUL(tr12, ci3)));
      tr3 = VADD(VLD(cc, cc_idx(i - 1, k, 1)), SVMADD(tr12, cr2, SVMUL(tr11, cr3)));
      ti3 = VSUB(VLD(cc, cc_idx(i, k, 1)), SVMADD(tr12, ci2, SVMUL(tr11, ci3)));
      tr5 = SVMADD(ti11, cr5, SVMUL(ti12, cr4));
      ti5 = SVMADD(ti11, ci5, SVMUL(ti12, ci4));
      tr4 = SVMSUB(ti11, cr4, SVMUL(ti12, cr5));
      ti4 = SVMSUB(ti11, ci4, SVMUL(ti12, ci5));
      VST(ch, ch_idx(i - 1, 3, k), VSUB(tr2, tr5));
      VST(ch, ch_idx(ic - 1, 2, k), VADD(tr2, tr5));
      VST(ch, ch_idx(i, 3, k), VADD(ti2, ti5));
      VST(ch, ch_idx(ic, 2, k), VSUB(ti5, ti2));
      VST(ch, ch_idx(i - 1, 5, k), VSUB(tr3, tr4));
      VST(ch, ch_idx(ic - 1, 4, k), VADD(tr3, tr4));
      VST(ch, ch_idx(i, 5, k), VADD(ti3, ti4));
      VST(ch, ch_idx(ic, 4, k), VSUB(ti4, ti3));
    }
  }
#undef cc_idx
#undef ch_idx
} /* radf5 */

static void radb5_ps(int ido, int l1, const v4sf *RESTRICT cc, v4sf *RESTRICT ch,
                  const float *wa1, const float *wa2, const float *wa3, const float *wa4)
{
  static const float tr11 = (float)0.30901699437494742410;   /* cos(2*pi/5) = (sqrt(5)-1)/4 */
  static const float ti11 = (float)0.95105651629515357212;   /* sin(2*pi/5) */
  static const float tr12 = (float)-0.80901699437494742410;  /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
  static const float ti12 = (float)0.58778525229247312917;   /* sin(4*pi/5) */

  int cc_offset, ch_offset;

  /* Local variables */
  int i, k, ic;
  v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
    ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
  int idp2;

#define cc_idx(a_1,a_2,a_3) (((a_3)*5 + (a_2))*ido + a_1)
#define ch_idx(a_1,a_2,a_3) (((a_3)*l1 + (a_2))*ido + a_1)

  /* Parameter adjustments */
  ch_offset = 1 + ido * (1 + l1);
  ch -= ch_offset;
  cc_offset = 1 + ido * 6;
  cc -= cc_offset;

  /* Function Body */
  for (k = 1; k <= l1; ++k) {
    ti5 = VADD(VLD(cc, cc_idx(1, 3, k)), VLD(cc, cc_idx(1, 3, k)));
    ti4 = VADD(VLD(cc, cc_idx(1, 5, k)), VLD(cc, cc_idx(1, 5, k)));
    tr2 = VADD(VLD(cc, cc_idx(ido, 2, k)), VLD(cc, cc_idx(ido, 2, k)));
    tr3 = VADD(VLD(cc, cc_idx(ido, 4, k)), VLD(cc, cc_idx(ido, 4, k)));
    VST(ch, ch_idx(1, k, 1), VADD(VLD(cc, cc_idx(1, 1, k)), VADD(tr2, tr3)));
    cr2 = VADD(VLD(cc, cc_idx(1, 1, k)), SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
    cr3 = VADD(VLD(cc, cc_idx(1, 1, k)), SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
    ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
    ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
    VST(ch, ch_idx(1, k, 2), VSUB(cr2, ci5));
    VST(ch, ch_idx(1, k, 3), VSUB(cr3, ci4));
    VST(ch, ch_idx(1, k, 4), VADD(cr3, ci4));
    VST(ch, ch_idx(1, k, 5), VADD(cr2, ci5));
  }
  if (ido == 1) {
    return;
  }
  idp2 = ido + 2;
  for (k = 1; k <= l1; ++k) {
    for (i = 3; i <= ido; i += 2) {
      ic = idp2 - i;
      ti5 = VADD(VLD(cc, cc_idx(i  , 3, k)), VLD(cc, cc_idx(ic  , 2, k)));
      ti2 = VSUB(VLD(cc, cc_idx(i  , 3, k)), VLD(cc, cc_idx(ic  , 2, k)));
      ti4 = VADD(VLD(cc, cc_idx(i  , 5, k)), VLD(cc, cc_idx(ic  , 4, k)));
      ti3 = VSUB(VLD(cc, cc_idx(i  , 5, k)), VLD(cc, cc_idx(ic  , 4, k)));
      tr5 = VSUB(VLD(cc, cc_idx(i-1, 3, k)), VLD(cc, cc_idx(ic-1, 2, k)));
      tr2 = VADD(VLD(cc, cc_idx(i-1, 3, k)), VLD(cc, cc_idx(ic-1, 2, k)));
      tr4 = VSUB(VLD(cc, cc_idx(i-1, 5, k)), VLD(cc, cc_idx(ic-1, 4, k)));
      tr3 = VADD(VLD(cc, cc_idx(i-1, 5, k)), VLD(cc, cc_idx(ic-1, 4, k)));
      VST(ch, ch_idx(i - 1, k, 1), VADD(VLD(cc, cc_idx(i-1, 1, k)), VADD(tr2, tr3)));
      VST(ch, ch_idx(i, k, 1), VADD(VLD(cc, cc_idx(i, 1, k)), VADD(ti2, ti3)));
      cr2 = VADD(VLD(cc, cc_idx(i-1, 1, k)), SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
      ci2 = VADD(VLD(cc, cc_idx(i  , 1, k)), SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
      cr3 = VADD(VLD(cc, cc_idx(i-1, 1, k)), SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
      ci3 = VADD(VLD(cc, cc_idx(i  , 1, k)), SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
      cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
      ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
      cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
      ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
      dr3 = VSUB(cr3, ci4);
      dr4 = VADD(cr3, ci4);
      di3 = VADD(ci3, cr4);
      di4 = VSUB(ci3, cr4);
      dr5 = VADD(cr2, ci5);
      dr2 = VSUB(cr2, ci5);
      di5 = VSUB(ci2, cr5);
      di2 = VADD(ci2, cr5);
      VCPLXMUL(dr2, di2, LD_PS1(wa1[i-3]), LD_PS1(wa1[i-2]));
      VCPLXMUL(dr3, di3, LD_PS1(wa2[i-3]), LD_PS1(wa2[i-2]));
      VCPLXMUL(dr4, di4, LD_PS1(wa3[i-3]), LD_PS1(wa3[i-2]));
      VCPLXMUL(dr5, di5, LD_PS1(wa4[i-3]), LD_PS1(wa4[i-2]));

      VST(ch, ch_idx(i-1, k, 2), dr2); VST(ch, ch_idx(i, k, 2), di2);
      VST(ch, ch_idx(i-1, k, 3), dr3); VST(ch, ch_idx(i, k, 3), di3);
      VST(ch, ch_idx(i-1, k, 4), dr4); VST(ch, ch_idx(i, k, 4), di4);
      VST(ch, ch_idx(i-1, k, 5), dr5); VST(ch, ch_idx(i, k, 5), di5);
    }
  }
#undef cc_idx
#undef ch_idx
} /* radb5 */

#if PFFFT_TERMINAL_RADIX8
/*
  Radix-8 real passes at ido == 1 (no twiddles), from terminal_radix8_plan:
  the first pass of the forward transform and the last pass of the backward
  one. They replace the radix-2 pass, the one with the longest twiddle
  table, and the radix-4 pass at ido == 1: the radix-2 pass is dropped and
  that radix-4 pass becomes the radix-8 pass, so every pass in between runs
  with other l1 and ido; the transform has one pass over the data fewer.

  forward:  X_m = sum_j a_j exp(-2 pi i j m/8), stored half-complex as
            R0 R1 I1 R2 I2 R3 I3 R4
  backward: 8 a_j from the same layout (unscaled, as FFTPACK's passes)
*/
static NEVER_INLINE(void) radf8_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch) {
  const float c8 = (float)M_SQRT1_2;   /* 1/sqrt(2) */
  const v4sf k8 = LD_PS1(c8);          /* named: see passf8_ps */
  int k;
  assert(ido == 1);
  (void)ido; /* otherwise unused when assertions are disabled */
  for (k=0; k < l1; ++k, ++cc, ch += 8) {
    v4sf a0 = VLD(cc, 0),    a4 = VLD(cc, 4*l1);
    v4sf a1 = VLD(cc, l1),   a5 = VLD(cc, 5*l1);
    v4sf a2 = VLD(cc, 2*l1), a6 = VLD(cc, 6*l1);
    v4sf a3 = VLD(cc, 3*l1), a7 = VLD(cc, 7*l1);
    v4sf s0 = VADD(a0, a4), d0 = VSUB(a0, a4);
    v4sf s1 = VADD(a1, a5), d1 = VSUB(a1, a5);
    v4sf s2 = VADD(a2, a6), d2 = VSUB(a2, a6), nd2 = VSUB(a6, a2);
    v4sf s3 = VADD(a3, a7), d3 = VSUB(a3, a7);
    v4sf e0 = VADD(s0, s2), e1 = VADD(s1, s3);
    v4sf u = VSUB(d1, d3), v = VADD(d1, d3);
    VST(ch, 0, VADD(e0, e1));            /* R0 */
    VST(ch, 1, VMADD(k8, u, d0));        /* R1 = d0 + u/sqrt2 */
    VST(ch, 2, VMSUB(k8, v, nd2));       /* I1 = -d2 - v/sqrt2 */
    VST(ch, 3, VSUB(s0, s2));            /* R2 */
    VST(ch, 4, VSUB(s3, s1));            /* I2 */
    VST(ch, 5, VMSUB(k8, u, d0));        /* R3 = d0 - u/sqrt2 */
    VST(ch, 6, VMSUB(k8, v, d2));        /* I3 = d2 - v/sqrt2 */
    VST(ch, 7, VSUB(e0, e1));            /* R4 */
  }
} /* radf8 */

static NEVER_INLINE(void) radb8_ps(int ido, int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT ch) {
  const float c2 = 2.f, csq2 = (float)M_SQRT2;
  const v4sf two = LD_PS1(c2), sq2 = LD_PS1(csq2);  /* named: see passf8_ps */
  int k;
  assert(ido == 1);
  (void)ido; /* otherwise unused when assertions are disabled */
  for (k=0; k < l1; ++k, cc += 8, ++ch) {
    v4sf r0 = VLD(cc, 0), r1 = VLD(cc, 1), i1 = VLD(cc, 2), r2 = VLD(cc, 3);
    v4sf i2 = VLD(cc, 4), r3 = VLD(cc, 5), i3 = VLD(cc, 6), r4 = VLD(cc, 7);
    v4sf p = VADD(r0, r4), q = VSUB(r0, r4);
    v4sf a = VADD(r1, r3), b = VSUB(r1, r3), c = VSUB(i3, i1), d = VADD(i1, i3);
    v4sf e0 = VMADD(two, r2, p), e2 = VMSUB(two, r2, p);
    v4sf e1 = VMSUB(two, i2, q), e3 = VMADD(two, i2, q);
    v4sf f1 = VSUB(b, d), f3 = VADD(b, d);
    VST(ch, 0,    VMADD(two, a, e0));
    VST(ch, 4*l1, VMSUB(two, a, e0));
    VST(ch, 2*l1, VMADD(two, c, e2));
    VST(ch, 6*l1, VMSUB(two, c, e2));
    VST(ch, 1*l1, VMADD(sq2, f1, e1));
    VST(ch, 5*l1, VMSUB(sq2, f1, e1));
    VST(ch, 3*l1, VMSUB(sq2, f3, e3));
    VST(ch, 7*l1, VMADD(sq2, f3, e3));
  }
} /* radb8 */
#endif /* PFFFT_TERMINAL_RADIX8 */

static NEVER_INLINE(v4sf *) rfftf1_ps(int n, const v4sf *input_readonly, v4sf *work1, v4sf *work2,
                                      const float *wa, const int *ifac) {
  v4sf *in  = (v4sf*)input_readonly;
  v4sf *out = (in == work2 ? work1 : work2);
  int nf = ifac[1], k1;
  int l2 = n;
  int iw = n-1;
  assert(in != out && work1 != work2);
  for (k1 = 1; k1 <= nf; ++k1) {
    int kh = nf - k1;
    int ip = ifac[kh + 2];
    int l1 = l2 / ip;
    int ido = n / l2;
    iw -= (ip - 1)*ido;
#if PFFFT_TERMINAL_RADIX8
    /* radix 8 is dispatched outside the switch: a fifth case makes GCC
       compile it to a jump table, an indirect jump. check_isa_baseline
       rejects an indirect jump in 8-lane code while the upper ymm halves
       may be dirty, and they are here: the radix-3 and radix-5 passes are
       inlined into this driver. The complex drivers dispatch by if-chain
       for the same reason. */
    if (ip == 8) {
      radf8_ps(ido, l1, in, out);
    } else
#endif
    switch (ip) {
      case 5: {
        int ix2 = iw + ido;
        int ix3 = ix2 + ido;
        int ix4 = ix3 + ido;
        radf5_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
      } break;
      case 4: {
        int ix2 = iw + ido;
        int ix3 = ix2 + ido;
        radf4_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
      } break;
      case 3: {
        int ix2 = iw + ido;
        radf3_ps(ido, l1, in, out, &wa[iw], &wa[ix2]);
      } break;
      case 2:
        radf2_ps(ido, l1, in, out, &wa[iw]);
        break;
      default:
        assert(0);
        break;
    }
    l2 = l1;
    if (out == work2) {
      out = work1; in = work2;
    } else {
      out = work2; in = work1;
    }
  }
  return in; /* this is in fact the output .. */
} /* rfftf1 */

static NEVER_INLINE(v4sf *) rfftb1_ps(int n, const v4sf *input_readonly, v4sf *work1, v4sf *work2,
                                      const float *wa, const int *ifac) {
  v4sf *in  = (v4sf*)input_readonly;
  v4sf *out = (in == work2 ? work1 : work2);
  int nf = ifac[1], k1;
  int l1 = 1;
  int iw = 0;
  assert(in != out);
  for (k1=1; k1<=nf; k1++) {
    int ip = ifac[k1 + 1];
    int l2 = ip*l1;
    int ido = n / l2;
#if PFFFT_TERMINAL_RADIX8
    /* outside the switch: no jump table, see rfftf1_ps */
    if (ip == 8) {
      radb8_ps(ido, l1, in, out);
    } else
#endif
    switch (ip) {
      case 5: {
        int ix2 = iw + ido;
        int ix3 = ix2 + ido;
        int ix4 = ix3 + ido;
        radb5_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
      } break;
      case 4: {
        int ix2 = iw + ido;
        int ix3 = ix2 + ido;
        radb4_ps(ido, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3]);
      } break;
      case 3: {
        int ix2 = iw + ido;
        radb3_ps(ido, l1, in, out, &wa[iw], &wa[ix2]);
      } break;
      case 2:
        radb2_ps(ido, l1, in, out, &wa[iw]);
        break;
      default:
        assert(0);
        break;
    }
    l1 = l2;
    iw += (ip - 1)*ido;

    if (out == work2) {
      out = work1; in = work2;
    } else {
      out = work2; in = work1;
    }
  }
  return in; /* this is in fact the output .. */
}

#ifndef IFAC_MAX_SIZE
#define IFAC_MAX_SIZE 25 /* max number of integer factors for the decomposition, +2 */
#endif
static int decompose(int n, int *ifac, const int *ntryh) {
  int nl = n, nf = 0, i, j = 0;
  for (j=0; ntryh[j]; ++j) {
    int ntry = ntryh[j];
    while (nl != 1) {
      int nq = nl / ntry;
      int nr = nl - ntry * nq;
      if (nr == 0) {
        assert(2 + nf < IFAC_MAX_SIZE);
        ifac[2+nf++] = ntry;
        nl = nq;
        if (ntry == 2 && nf != 1) {
          for (i = 2; i <= nf; ++i) {
            int ib = nf - i + 2;
            ifac[ib + 1] = ifac[ib];
          }
          ifac[2] = 2;
        }
      } else break;
    }
  }
  ifac[0] = n;
  ifac[1] = nf;
  return nf;
}



/*
  cos and sin of 2*pi*m/n for 0 <= m < n. The angle is reduced exactly in
  integers to [0, pi/4] and evaluated in long double, so every factor is
  correctly rounded or nearly so (at most ~1 ulp where long double is
  double), and the symmetric factors (+-1, 0, cos = sin at pi/4, ...) are
  exact. Repeated rounding of 2*pi/n * m in the working precision instead
  costs up to ~5 eps per factor in float.
*/
static void twiddle_cos_sin(int m, int n, float *c, float *s)
{
  const long double pi_2 = 1.570796326794896619231321691639751442L;
  long q = (4L*m) / n;          /* quadrant */
  long r = 4L*m - q*n;          /* angle in the quadrant = pi/2 * r/n */
  long double x, cx, sx, t;
  if (2*r <= n) {
    x = pi_2 * (long double)r / n;
    cx = cosl(x);
    sx = sinl(x);
  } else {                      /* pi/2 - angle is in [0, pi/4) */
    x = pi_2 * (long double)(n - r) / n;
    cx = sinl(x);
    sx = cosl(x);
  }
  switch (q & 3) {
    case 1: t = cx; cx = -sx; sx = t; break;
    case 2: cx = -cx; sx = -sx; break;
    case 3: t = cx; cx = sx; sx = -t; break;
    default: break;
  }
  *c = (float)cx;
  *s = (float)sx;
}

#if PFFFT_TERMINAL_RADIX8
/*
  The plan rule of PFFFT_TERMINAL_RADIX8, applied by rffti1_ps and
  cffti1_ps to the factor list decompose() made: no stand-alone radix-2
  pass. If the list starts with a 2 (decompose puts a 2 first) and ends
  with a 4, the 2 is dropped and that last 4 becomes an 8; ifac[1] is
  updated. Returns the number of factors.

  The last factor is the pass with one element per leg: the last pass of
  a complex transform in both directions, the first pass of the real
  forward transform and the last of the real backward one. There the
  radix-8 pass (passf8_ps, radf8_ps, radb8_ps) needs no twiddles.
  Complex lists 2, ..., 4 become ..., 8. Real lists end with a 4 only for
  powers of two, 2, 4, ..., 4, which become 4, ..., 8: FFTPACK's odd-radix
  real passes need an odd ido, so the factors 3 and 5 come last in a real
  list and their passes run before the even-radix ones in the forward
  transform; a radix-8 pass after them would need twiddles, and the list
  keeps its radix-2 pass.
*/
static int terminal_radix8_plan(int *ifac) {
  int nf = ifac[1], k;
  if (nf >= 2 && ifac[2] == 2 && ifac[nf+1] == 4) {
    for (k = 2; k <= nf; ++k) ifac[k] = ifac[k+1];
    ifac[nf] = 8;
    ifac[1] = --nf;
  }
  return nf;
}
#endif /* PFFFT_TERMINAL_RADIX8 */

static void rffti1_ps(int n, float *wa, int *ifac)
{
  static const int ntryh[] = { 4,2,3,5,0 };
  int k1, j, ii;

#if PFFFT_TERMINAL_RADIX8
  int nf;
  decompose(n,ifac,ntryh);
  nf = terminal_radix8_plan(ifac);
#else
  int nf = decompose(n,ifac,ntryh);
#endif
  int is = 0;
  int nfm1 = nf - 1;
  int l1 = 1;
  for (k1 = 1; k1 <= nfm1; k1++) {
    int ip = ifac[k1 + 1];
    int ld = 0;
    int l2 = l1*ip;
    int ido = n / l2;
    int ipm = ip - 1;
    for (j = 1; j <= ipm; ++j) {
      int i = is, fi=0;
      ld += l1;
      for (ii = 3; ii <= ido; ii += 2) {
        i += 2;
        fi += 1;
        twiddle_cos_sin(fi*ld, n, &wa[i - 2], &wa[i - 1]);  /* fi*ld < n */
      }
      is += ido;
    }
    l1 = l2;
  }
} /* rffti1 */

static void cffti1_ps(int n, float *wa, int *ifac)
{
  /* Radix 8 is used only as the pass with one element per leg, the last
     one, where it needs no twiddles. As a twiddled pass in an earlier
     position it is slower: on x86-64, the passes alone (without the
     cross-lane step) of an 8-lane complex transform of N = 16384, per-lane
     length n = 2048, took 1.47 times as long with 8,8,8,4 as with
     2,4,4,4,4,4, and the arm64 measurements below found the same. Two
     regimes:

     PFFFT_TERMINAL_RADIX8 == 0: radix 8 only for n == 8 (the Ncvec==8
     case, N=32 complex on 4 lanes), where a single dedicated pass is
     guaranteed. Measured on Mac/arm64, Android/arm64-v8a and iPhone/arm64
     (20-rep sustained, Mann-Whitney U + Hodges-Lehmann shift): Ncvec==8
     is a clear win (-2% to -21%) on every platform/precision; Ncvec==16
     regressed double precision by +4% to +6% on every platform even when
     restricted-in, and was mixed-to-negative for float, so it was
     excluded and the radix-16 codelet (passf16_ps) removed entirely --
     it would otherwise be unreachable at every N. Larger multi-pass
     sizes regress severely (+15% to +27% at N=16384) if this gate is
     lifted; see git history for the controlled A/B data.

     PFFFT_TERMINAL_RADIX8 == 1: the default try-list for every n, then
     terminal_radix8_plan: a list 2, ..., 4 becomes ..., 8 (n == 8 still
     gives the single pass 8), which saves one sweep over the data. The
     measurements are in benchmarks/perf_analysis/README.md. */
#if PFFFT_TERMINAL_RADIX8
  static const int ntryh[] = { 5,3,4,2,0 };
  int k1, j, ii;

  int nf;
  decompose(n,ifac,ntryh);
  nf = terminal_radix8_plan(ifac);
#else
  static const int ntryh_large[] = { 5,3,8,4,2,0 };
  static const int ntryh_default[] = { 5,3,4,2,0 };
  const int *ntryh = (n == 8) ? ntryh_large : ntryh_default;
  int k1, j, ii;

  int nf = decompose(n,ifac,ntryh);
#endif

  int i = 1;
  int l1 = 1;
  for (k1=1; k1<=nf; k1++) {
    int ip = ifac[k1+1];
    int ld = 0;
    int l2 = l1*ip;
    int ido = n / l2;
    int idot = ido + ido + 2;
    int ipm = ip - 1;
    for (j=1; j<=ipm; j++) {
      int i1 = i, fi = 0;
      wa[i-1] = 1;
      wa[i] = 0;
      ld += l1;
      for (ii = 4; ii <= idot; ii += 2) {
        i += 2;
        fi += 1;
        twiddle_cos_sin(fi*ld, n, &wa[i-1], &wa[i]);  /* fi*ld < n */
      }
      if (ip > 16) {
        wa[i1-1] = wa[i-1];
        wa[i1] = wa[i];
      }
    }
    l1 = l2;
  }
} /* cffti1 */


#ifndef PF_FUSED_LAYOUT   /* superseded by cfftf1_il_ps / cfftf1_tail_ol_ps below */
static v4sf *cfftf1_ps(int n, const v4sf *input_readonly, v4sf *work1, v4sf *work2, const float *wa, const int *ifac, int isign) {
  v4sf *in  = (v4sf*)input_readonly;
  v4sf *out = (in == work2 ? work1 : work2);
  int nf = ifac[1], k1;
  int l1 = 1;
  int iw = 0;
  assert(in != out && work1 != work2);
  for (k1=2; k1<=nf+1; k1++) {
    int ip = ifac[k1];
    int l2 = ip*l1;
    int ido = n / l2;
    int idot = ido + ido;
    switch (ip) {
      case 8: {
        passf8_ps(idot, l1, in, out, isign);
      } break;
      case 5: {
        int ix2 = iw + idot;
        int ix3 = ix2 + idot;
        int ix4 = ix3 + idot;
        passf5_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
      } break;
      case 4: {
        int ix2 = iw + idot;
        int ix3 = ix2 + idot;
        passf4_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], isign);
      } break;
      case 2: {
        passf2_ps(idot, l1, in, out, &wa[iw], isign);
      } break;
      case 3: {
        int ix2 = iw + idot;
        passf3_ps(idot, l1, in, out, &wa[iw], &wa[ix2], isign);
      } break;
      default:
        assert(0);
    }
    l1 = l2;
    iw += (ip - 1)*idot;
    if (out == work2) {
      out = work1; in = work2;
    } else {
      out = work2; in = work1;
    }
  }

  return in; /* this is in fact the output .. */
}
#endif
#endif /* PF_INPLACE_CODELETS */

#ifdef PF_FUSED_LAYOUT
#ifndef PF_INPLACE_CODELETS

/*
  Complex transforms deinterleave their input before the first pass and
  interleave the result of the last backward pass. The variants below do
  that while loading (first pass, l1 == 1) or storing (last pass, ido == 2):
  the element pair cc[i], cc[i+1] becomes PF_UNINTERLEAVE2(in[i], in[i+1])
  and ch[j], ch[j+1] becomes PF_INTERLEAVE2(..., out[j], out[j+1]). The
  arithmetic is that of passf2/3/4/5_ps, so the results are bit-identical
  and one full pass over the data disappears. The including file defines
  PF_FUSED_LAYOUT, PF_UNINTERLEAVE2 and PF_INTERLEAVE2.
*/

/* PF_(UN)INTERLEAVE2 with the outputs in p[i1], p[i2]; a backend whose VST
   is not a plain assignment defines these too */
#ifndef PF_INTERLEAVE2_ST
#  define PF_INTERLEAVE2_ST(in1, in2, p, i1, i2) PF_INTERLEAVE2(in1, in2, (p)[i1], (p)[i2])
#  define PF_UNINTERLEAVE2_ST(in1, in2, p, i1, i2) PF_UNINTERLEAVE2(in1, in2, (p)[i1], (p)[i2])
#endif
#endif /* PF_INPLACE_CODELETS */

static PF_CODELET(void) PF_CODELET_NAME(passf2_il)(int ido, const v4sf * PF_CODELET_RESTRICT in,
                                                   v4sf * PF_CODELET_RESTRICT ch,
                                                   const float *wa1, float fsign) {
  int i;
  v4sf c0r, c0i, c1r, c1i;
  if (ido <= 2) {
    PF_UNINTERLEAVE2(VLD(in, 0), VLD(in, 1), c0r, c0i);
    PF_UNINTERLEAVE2(VLD(in, ido), VLD(in, ido+1), c1r, c1i);
    VST(ch, 0, VADD(c0r, c1r));
    VST(ch, ido, VSUB(c0r, c1r));
    VST(ch, 1, VADD(c0i, c1i));
    VST(ch, ido+1, VSUB(c0i, c1i));
  } else {
    for (i=0; i<ido-1; i+=2) {
      v4sf tr2, ti2, wr, wi;
      PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i+1), c0r, c0i);
      PF_UNINTERLEAVE2(VLD(in, i+ido), VLD(in, i+ido+1), c1r, c1i);
      tr2 = VSUB(c0r, c1r);
      ti2 = VSUB(c0i, c1i);
      wr = LD_PS1(wa1[i]); wi = VMUL(LD_PS1(fsign), LD_PS1(wa1[i+1]));
      VST(ch, i, VADD(c0r, c1r));
      VST(ch, i+1, VADD(c0i, c1i));
      VCPLXMUL(tr2, ti2, wr, wi);
      VST(ch, i+ido, tr2);
      VST(ch, i+ido+1, ti2);
    }
  }
}

static PF_CODELET(void) PF_CODELET_NAME(passf3_il)(int ido, const v4sf * PF_CODELET_RESTRICT in,
                                                   v4sf * PF_CODELET_RESTRICT ch,
                                                   const float *wa1, const float *wa2, float fsign) {
  static const float taur = (float)-0.5;                     /* cos(2*pi/3)   */
  float taui = (float)0.86602540378443864676*fsign;          /* sin(2*pi/3) = sqrt(3)/2 */
  int i;
  v4sf tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
  v4sf c0r, c0i, c1r, c1i, c2r, c2i;
  float wr1, wi1, wr2, wi2;
  assert(ido >= 2);
  for (i=0; i<ido-1; i+=2) {
    PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i+1), c0r, c0i);
    PF_UNINTERLEAVE2(VLD(in, i+ido), VLD(in, i+ido+1), c1r, c1i);
    PF_UNINTERLEAVE2(VLD(in, i+2*ido), VLD(in, i+2*ido+1), c2r, c2i);
    tr2 = VADD(c1r, c2r);
    cr2 = SVMADD(taur, tr2, c0r);
    VST(ch, i, VADD(c0r, tr2));
    ti2 = VADD(c1i, c2i);
    ci2 = SVMADD(taur, ti2, c0i);
    VST(ch, i+1, VADD(c0i, ti2));
    cr3 = SVMUL(taui, VSUB(c1r, c2r));
    ci3 = SVMUL(taui, VSUB(c1i, c2i));
    dr2 = VSUB(cr2, ci3);
    dr3 = VADD(cr2, ci3);
    di2 = VADD(ci2, cr3);
    di3 = VSUB(ci2, cr3);
    wr1=wa1[i]; wi1=fsign*wa1[i+1]; wr2=wa2[i]; wi2=fsign*wa2[i+1];
    VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
    VST(ch, i+ido, dr2);
    VST(ch, i+ido + 1, di2);
    VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
    VST(ch, i+2*ido, dr3);
    VST(ch, i+2*ido+1, di3);
  }
}

#ifndef PF_INPLACE_CODELETS
/* radix-4 butterfly of passf4_ps on one element pair of each leg; writes
   the four results to (y[2m], y[2m+1]) */
static ALWAYS_INLINE(void) passf4_il_butterfly(v4sf c0r, v4sf c0i, v4sf c1r, v4sf c1i,
                                               v4sf c2r, v4sf c2i, v4sf c3r, v4sf c3i,
                                               float fsign, v4sf *y) {
  v4sf tr1 = VSUB(c0r, c2r);
  v4sf tr2 = VADD(c0r, c2r);
  v4sf ti1 = VSUB(c0i, c2i);
  v4sf ti2 = VADD(c0i, c2i);
  v4sf ti4 = VMUL(VSUB(c1r, c3r), LD_PS1(fsign));
  v4sf tr4 = VMUL(VSUB(c3i, c1i), LD_PS1(fsign));
  v4sf tr3 = VADD(c1r, c3r);
  v4sf ti3 = VADD(c1i, c3i);
  y[0] = VADD(tr2, tr3);
  y[1] = VADD(ti2, ti3);
  y[2] = VADD(tr1, tr4);
  y[3] = VADD(ti1, ti4);
  y[4] = VSUB(tr2, tr3);
  y[5] = VSUB(ti2, ti3);
  y[6] = VSUB(tr1, tr4);
  y[7] = VSUB(ti1, ti4);
}
#endif /* PF_INPLACE_CODELETS */

static PF_CODELET(void) PF_CODELET_NAME(passf4_il)(int ido, const v4sf * PF_CODELET_RESTRICT in,
                                                   v4sf * PF_CODELET_RESTRICT ch,
                                                   const float *wa1, const float *wa2, const float *wa3,
                                                   float fsign) {
  int i;
  v4sf c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i;
  if (ido == 2) {
    v4sf y[8];
    PF_UNINTERLEAVE2(VLD(in, 0), VLD(in, 1), c0r, c0i);
    PF_UNINTERLEAVE2(VLD(in, ido), VLD(in, ido+1), c1r, c1i);
    PF_UNINTERLEAVE2(VLD(in, 2*ido), VLD(in, 2*ido+1), c2r, c2i);
    PF_UNINTERLEAVE2(VLD(in, 3*ido), VLD(in, 3*ido+1), c3r, c3i);
    passf4_il_butterfly(c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, fsign, y);
    VST(ch, 0*ido + 0, y[0]); VST(ch, 0*ido + 1, y[1]);
    VST(ch, 1*ido + 0, y[2]); VST(ch, 1*ido + 1, y[3]);
    VST(ch, 2*ido + 0, y[4]); VST(ch, 2*ido + 1, y[5]);
    VST(ch, 3*ido + 0, y[6]); VST(ch, 3*ido + 1, y[7]);
  } else {
    for (i=0; i<ido-1; i+=2) {
      v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      float wr1, wi1, wr2, wi2, wr3, wi3;
      PF_UNINTERLEAVE2(VLD(in, i), VLD(in, i+1), c0r, c0i);
      PF_UNINTERLEAVE2(VLD(in, i+ido), VLD(in, i+ido+1), c1r, c1i);
      PF_UNINTERLEAVE2(VLD(in, i+2*ido), VLD(in, i+2*ido+1), c2r, c2i);
      PF_UNINTERLEAVE2(VLD(in, i+3*ido), VLD(in, i+3*ido+1), c3r, c3i);
      tr1 = VSUB(c0r, c2r);
      tr2 = VADD(c0r, c2r);
      ti1 = VSUB(c0i, c2i);
      ti2 = VADD(c0i, c2i);
      tr4 = VMUL(VSUB(c3i, c1i), LD_PS1(fsign));
      ti4 = VMUL(VSUB(c1r, c3r), LD_PS1(fsign));
      tr3 = VADD(c1r, c3r);
      ti3 = VADD(c1i, c3i);

      VST(ch, i, VADD(tr2, tr3));
      cr3    = VSUB(tr2, tr3);
      VST(ch, i + 1, VADD(ti2, ti3));
      ci3 = VSUB(ti2, ti3);

      cr2 = VADD(tr1, tr4);
      cr4 = VSUB(tr1, tr4);
      ci2 = VADD(ti1, ti4);
      ci4 = VSUB(ti1, ti4);
      wr1=wa1[i]; wi1=fsign*wa1[i+1];
      VCPLXMUL(cr2, ci2, LD_PS1(wr1), LD_PS1(wi1));
      wr2=wa2[i]; wi2=fsign*wa2[i+1];
      VST(ch, i + ido, cr2);
      VST(ch, i + ido + 1, ci2);

      VCPLXMUL(cr3, ci3, LD_PS1(wr2), LD_PS1(wi2));
      wr3=wa3[i]; wi3=fsign*wa3[i+1];
      VST(ch, i + 2*ido, cr3);
      VST(ch, i + 2*ido + 1, ci3);

      VCPLXMUL(cr4, ci4, LD_PS1(wr3), LD_PS1(wi3));
      VST(ch, i + 3*ido, cr4);
      VST(ch, i + 3*ido + 1, ci4);
    }
  }
}

static PF_CODELET(void) PF_CODELET_NAME(passf5_il)(int ido, const v4sf * PF_CODELET_RESTRICT in,
                                                   v4sf * PF_CODELET_RESTRICT ch,
                                                   const float *wa1, const float *wa2,
                                                   const float *wa3, const float *wa4, float fsign) {
  static const float tr11 = (float)0.30901699437494742410;   /* cos(2*pi/5) = (sqrt(5)-1)/4 */
  const float ti11 = (float)0.95105651629515357212*fsign;    /* sin(2*pi/5) */
  static const float tr12 = (float)-0.80901699437494742410;  /* cos(4*pi/5) = -(sqrt(5)+1)/4 */
  const float ti12 = (float)0.58778525229247312917*fsign;    /* sin(4*pi/5) */
  int i, j;
  v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
    ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
  float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

  assert(ido >= 2);
  for (i = 0; i < ido-1; i += 2) {
    v4sf xr[6], xi[6];   /* legs 1..5 as in passf5_ps */
    for (j = 1; j <= 5; ++j) {
      PF_UNINTERLEAVE2(VLD(in, (j-1)*ido + i), VLD(in, (j-1)*ido + i + 1), xr[j], xi[j]);
    }
    ti5 = VSUB(xi[2], xi[5]);
    ti2 = VADD(xi[2], xi[5]);
    ti4 = VSUB(xi[3], xi[4]);
    ti3 = VADD(xi[3], xi[4]);
    tr5 = VSUB(xr[2], xr[5]);
    tr2 = VADD(xr[2], xr[5]);
    tr4 = VSUB(xr[3], xr[4]);
    tr3 = VADD(xr[3], xr[4]);
    VST(ch, i, VADD(xr[1], VADD(tr2, tr3)));
    VST(ch, i + 1, VADD(xi[1], VADD(ti2, ti3)));
    cr2 = VADD(xr[1], SVMADD(tr11, tr2, SVMUL(tr12, tr3)));
    ci2 = VADD(xi[1], SVMADD(tr11, ti2, SVMUL(tr12, ti3)));
    cr3 = VADD(xr[1], SVMADD(tr12, tr2, SVMUL(tr11, tr3)));
    ci3 = VADD(xi[1], SVMADD(tr12, ti2, SVMUL(tr11, ti3)));
    cr5 = SVMADD(ti11, tr5, SVMUL(ti12, tr4));
    ci5 = SVMADD(ti11, ti5, SVMUL(ti12, ti4));
    cr4 = SVMSUB(ti11, tr4, SVMUL(ti12, tr5));
    ci4 = SVMSUB(ti11, ti4, SVMUL(ti12, ti5));
    dr3 = VSUB(cr3, ci4);
    dr4 = VADD(cr3, ci4);
    di3 = VADD(ci3, cr4);
    di4 = VSUB(ci3, cr4);
    dr5 = VADD(cr2, ci5);
    dr2 = VSUB(cr2, ci5);
    di5 = VSUB(ci2, cr5);
    di2 = VADD(ci2, cr5);
    wr1=wa1[i]; wi1=fsign*wa1[i+1]; wr2=wa2[i]; wi2=fsign*wa2[i+1];
    wr3=wa3[i]; wi3=fsign*wa3[i+1]; wr4=wa4[i]; wi4=fsign*wa4[i+1];
    VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
    VST(ch, ido + i, dr2);
    VST(ch, ido + i + 1, di2);
    VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
    VST(ch, 2*ido + i, dr3);
    VST(ch, 2*ido + i + 1, di3);
    VCPLXMUL(dr4, di4, LD_PS1(wr3), LD_PS1(wi3));
    VST(ch, 3*ido + i, dr4);
    VST(ch, 3*ido + i + 1, di4);
    VCPLXMUL(dr5, di5, LD_PS1(wr4), LD_PS1(wi4));
    VST(ch, 4*ido + i, dr5);
    VST(ch, 4*ido + i + 1, di5);
  }
}

#ifndef PF_INPLACE_CODELETS
/* last pass (ido == 2: one complex element per leg) of passf4_ps, storing
   interleaved complex output */
static NEVER_INLINE(void) passf4_ol_ps(int l1, const v4sf * RESTRICT cc, v4sf * RESTRICT out, float fsign) {
  const int ido = 2, l1ido = l1*ido;
  int k;
  for (k=0; k < l1ido; k += ido, out += ido, cc += 4*ido) {
    v4sf y[8];
    passf4_il_butterfly(VLD(cc, 0), VLD(cc, 1), VLD(cc, ido + 0), VLD(cc, ido + 1), VLD(cc, 2*ido + 0), VLD(cc, 2*ido + 1),
                        VLD(cc, 3*ido + 0), VLD(cc, 3*ido + 1), fsign, y);
    PF_INTERLEAVE2_ST(y[0], y[1], out, 0*l1ido + 0, 0*l1ido + 1);
    PF_INTERLEAVE2_ST(y[2], y[3], out, 1*l1ido + 0, 1*l1ido + 1);
    PF_INTERLEAVE2_ST(y[4], y[5], out, 2*l1ido + 0, 2*l1ido + 1);
    PF_INTERLEAVE2_ST(y[6], y[7], out, 3*l1ido + 0, 3*l1ido + 1);
  }
}

#if PFFFT_TERMINAL_RADIX8
/* passf8_ps (ido == 2: the last pass), storing interleaved complex output;
   the same arithmetic, so the result has the bits of passf8_ps followed by
   PF_INTERLEAVE2 */
static NEVER_INLINE(void) passf8_ol_ps(int l1, const v4sf *cc, v4sf *out, float fsign) {
  const float c8 = (float)M_SQRT1_2;   /* 1/sqrt(2) */
  const v4sf k8 = LD_PS1(c8);          /* named: see passf8_ps */
  const int ido = 2, l1ido = l1*ido;
  int k;
  if (fsign < 0) {
#define PF_ST8(m, a, b) PF_INTERLEAVE2_ST(a, b, out, (m)*l1ido, (m)*l1ido + 1);
    for (k=0; k < l1ido; k += ido, out += ido, cc += 8*ido) {
      PF_PASS8(cc, 0, 1, PF_ST8)
    }
#undef PF_ST8
  } else {
#define PF_ST8(m, a, b) PF_INTERLEAVE2_ST(b, a, out, (m)*l1ido, (m)*l1ido + 1);
    for (k=0; k < l1ido; k += ido, out += ido, cc += 8*ido) {
      PF_PASS8(cc, 1, 0, PF_ST8)
    }
#undef PF_ST8
  }
}
#endif /* PFFFT_TERMINAL_RADIX8 */

#if PFFFT_INPLACE_FIRST_PASS
/*
  The first complex pass in place. In the first pass (l1 == 1) each
  butterfly reads the element pair at position i of every leg and writes
  its results to the same positions, so the pass can run with input and
  output the same buffer, provided that no butterfly reads a position after
  storing to it. The drivers below use that where a fused (de)interleaving
  pass would otherwise have to write into its own input: running the first
  pass in place exchanges the buffers of the later passes, and the separate
  (de)interleaving sweep over the data is not needed.

  The bodies meet the condition as follows. passf4_ps (ido > 2) and the
  four deinterleaving codelets passf2/3/4/5_il_ps load every leg before
  their first store. passf2_ps (ido > 2) and passf3_ps store the outputs of
  leg 0 between their loads, but each position is read only before the
  store to it. passf5_ps reads leg 1 again after storing to it; its in-place
  instantiation loads leg 1 into two locals before the first store
  (cc_leg1r, cc_leg1i). The plain radix-2 codelet with one element per leg
  (ido <= 2) stores ch[0] and then reads cc[0] again, so it must not run in
  place: passf2_ip_ps asserts ido > 2, and the driver takes the first pass
  in place only when another pass follows, which for a radix-2 first pass
  means at least two elements per leg. (Its deinterleaving form,
  passf2_il_ps, loads both legs first and is safe at any ido.)

  The bodies come from a second inclusion of this file, which compiles the
  eight codelets once more under the names passfR_body and passfR_il_body,
  force-inlined and without 'restrict'. The regular codelets keep
  'restrict' and their code.
*/
#define PF_INPLACE_CODELETS 1
#include "pffft_passes_impl.h"
#undef PF_INPLACE_CODELETS

static NEVER_INLINE(void) passf2_ip_ps(int ido, v4sf *c, const float *wa1, float fsign) {
  assert(ido > 2);   /* see above: one element per leg is not safe in place */
  passf2_body(ido, 1, c, c, wa1, fsign);
}

static NEVER_INLINE(void) passf3_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                       float fsign) {
  assert(ido > 2);
  passf3_body(ido, 1, c, c, wa1, wa2, fsign);
}

static NEVER_INLINE(void) passf4_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                       const float *wa3, float fsign) {
  assert(ido > 2);
  passf4_body(ido, 1, c, c, wa1, wa2, wa3, fsign);
}

static NEVER_INLINE(void) passf5_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                       const float *wa3, const float *wa4, float fsign) {
  assert(ido > 2);
  passf5_body(ido, 1, c, c, wa1, wa2, wa3, wa4, fsign);
}

static NEVER_INLINE(void) passf2_il_ip_ps(int ido, v4sf *c, const float *wa1, float fsign) {
  passf2_il_body(ido, c, c, wa1, fsign);
}

static NEVER_INLINE(void) passf3_il_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                          float fsign) {
  passf3_il_body(ido, c, c, wa1, wa2, fsign);
}

static NEVER_INLINE(void) passf4_il_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                          const float *wa3, float fsign) {
  passf4_il_body(ido, c, c, wa1, wa2, wa3, fsign);
}

static NEVER_INLINE(void) passf5_il_ip_ps(int ido, v4sf *c, const float *wa1, const float *wa2,
                                          const float *wa3, const float *wa4, float fsign) {
  passf5_il_body(ido, c, c, wa1, wa2, wa3, wa4, fsign);
}
#endif /* PFFFT_INPLACE_FIRST_PASS */

/* the passes of cfftf1_ps from factor index k1 on, l1 and iw as cfftf1_ps
   has them there; 'in' holds the data and 'out' is the other buffer. With
   'interleave' set, the last pass stores interleaved into il_out when it
   can (radix 4, or radix 8 with PFFFT_TERMINAL_RADIX8, not reading
   il_out); otherwise the result is interleaved into il_out afterwards, in
   a separate sweep. With PFFFT_INPLACE_FIRST_PASS, when the last pass
   would read il_out and the first pass (l1 == 1, not radix 8) is followed
   by another, the first pass runs in place in 'in' (passfR_ip_ps above),
   so that the last pass reads the other buffer. Returns the buffer holding
   the result. A flag rather than a NULL il_out selects this: callers pass
   one of their two buffers as il_out, and testing it for NULL would tell a
   static analyzer that that buffer, too, may be NULL. Callers use the
   wrappers below, which pass the flag as a constant. */
static ALWAYS_INLINE(v4sf *) cfftf1_passes_ps(int n, v4sf *in, v4sf *out, int k1, int l1, int iw,
                                              const float *wa, const int *ifac, int isign,
                                              int interleave, v4sf *il_out) {
  const int nf = ifac[1];
  assert(in != out);
#if PFFFT_INPLACE_FIRST_PASS
  /* the passes k1 .. nf+1 alternate between in and out, so the last one
     reads 'in' when their number is odd. An if-chain, not a switch: no
     jump table in the 8-lane code (check_isa_baseline) */
  if (interleave && l1 == 1 && k1 <= nf && ifac[k1] != 8 &&
      (((nf + 2 - k1) & 1) ? in : out) == il_out) {
    const int ip = ifac[k1], idot = 2*(n / ip);
    const float fsign = (float)isign;
    if (ip == 4) {
      passf4_ip_ps(idot, in, &wa[iw], &wa[iw + idot], &wa[iw + 2*idot], fsign);
    } else if (ip == 2) {
      passf2_ip_ps(idot, in, &wa[iw], fsign);
    } else if (ip == 3) {
      passf3_ip_ps(idot, in, &wa[iw], &wa[iw + idot], fsign);
    } else {
      assert(ip == 5);
      passf5_ip_ps(idot, in, &wa[iw], &wa[iw + idot], &wa[iw + 2*idot], &wa[iw + 3*idot], fsign);
    }
    l1 = ip;
    iw += (ip - 1)*idot;
    ++k1;
  }
#endif
  for (; k1<=nf+1; k1++) {
    int ip = ifac[k1];
    int l2 = ip*l1;
    int ido = n / l2;
    int idot = ido + ido;
    v4sf *tmp;
    if (interleave && k1 == nf+1 && ip == 4 && in != il_out) {
      passf4_ol_ps(l1, in, il_out, (float)isign);
      return il_out;
    }
#if PFFFT_TERMINAL_RADIX8
    if (interleave && k1 == nf+1 && ip == 8 && in != il_out) {
      passf8_ol_ps(l1, in, il_out, (float)isign);
      return il_out;
    }
#endif
    switch (ip) {
      case 8: {
        passf8_ps(idot, l1, in, out, isign);
      } break;
      case 5: {
        int ix2 = iw + idot;
        int ix3 = ix2 + idot;
        int ix4 = ix3 + idot;
        passf5_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
      } break;
      case 4: {
        int ix2 = iw + idot;
        int ix3 = ix2 + idot;
        passf4_ps(idot, l1, in, out, &wa[iw], &wa[ix2], &wa[ix3], isign);
      } break;
      case 2: {
        passf2_ps(idot, l1, in, out, &wa[iw], isign);
      } break;
      case 3: {
        int ix2 = iw + idot;
        passf3_ps(idot, l1, in, out, &wa[iw], &wa[ix2], isign);
      } break;
      default:
        assert(0);
    }
    l1 = l2;
    iw += (ip - 1)*idot;
    tmp = in; in = out; out = tmp;
  }
  if (interleave) {
    int k;
    for (k=0; k < n; ++k) {
      PF_INTERLEAVE2_ST(VLD(in, 2*k), VLD(in, 2*k+1), il_out, 2*k, 2*k+1);
    }
    return il_out;
  }
  return in;
}

/* cfftf1_passes_ps without interleaving: returns the buffer holding the
   result, in or out */
static v4sf *cfftf1_tail_ps(int n, v4sf *in, v4sf *out, int k1, int l1, int iw,
                            const float *wa, const int *ifac, int isign) {
  return cfftf1_passes_ps(n, in, out, k1, l1, iw, wa, ifac, isign, 0, NULL);
}

/* cfftf1_passes_ps with the result interleaved into il_out, which may be
   in or out */
static void cfftf1_tail_ol_ps(int n, v4sf *in, v4sf *out, int k1, int l1, int iw,
                              const float *wa, const int *ifac, int isign, v4sf *il_out) {
  cfftf1_passes_ps(n, in, out, k1, l1, iw, wa, ifac, isign, 1, il_out);
}

/* cfftf1_ps(n, work2, work1, work2, ...) for data that is still interleaved
   in 'input': the first pass deinterleaves it while loading and writes work1.
   input may be work2. If it is work1, the first pass deinterleaves in place
   in work1 with PFFFT_INPLACE_FIRST_PASS (passfR_il_ip_ps above); without
   it, and for the single radix-8 pass of n == 8, the deinterleaving is a
   separate pass into work2, as without fusion. Returns the buffer holding
   the result. */
static v4sf *cfftf1_il_ps(int n, const v4sf *input, v4sf *work1, v4sf *work2,
                          const float *wa, const int *ifac, int isign) {
  const int ip = ifac[2];
  const int ido = n / ip, idot = ido + ido;
  const float fsign = (float)isign;
  int k;
  assert(work1 != work2);
#if PFFFT_INPLACE_FIRST_PASS
  if (input == work1 && ip != 8) {
    /* an if-chain, not a switch: no jump table in the 8-lane code */
    if (ip == 4) {
      passf4_il_ip_ps(idot, work1, &wa[0], &wa[idot], &wa[2*idot], fsign);
    } else if (ip == 2) {
      passf2_il_ip_ps(idot, work1, &wa[0], fsign);
    } else if (ip == 3) {
      passf3_il_ip_ps(idot, work1, &wa[0], &wa[idot], fsign);
    } else {
      assert(ip == 5);
      passf5_il_ip_ps(idot, work1, &wa[0], &wa[idot], &wa[2*idot], &wa[3*idot], fsign);
    }
    return cfftf1_tail_ps(n, work1, work2, 3, ip, (ip - 1)*idot, wa, ifac, isign);
  }
#endif
  if (input != work1 && ip != 8) {
    switch (ip) {
      case 5:
        passf5_il_ps(idot, input, work1, &wa[0], &wa[idot], &wa[2*idot], &wa[3*idot], fsign);
        break;
      case 4:
        passf4_il_ps(idot, input, work1, &wa[0], &wa[idot], &wa[2*idot], fsign);
        break;
      case 3:
        passf3_il_ps(idot, input, work1, &wa[0], &wa[idot], fsign);
        break;
      default:
        assert(ip == 2);
        passf2_il_ps(idot, input, work1, &wa[0], fsign);
        break;
    }
    return cfftf1_tail_ps(n, work1, work2, 3, ip, (ip - 1)*idot, wa, ifac, isign);
  }
  for (k=0; k < n; ++k) {
    PF_UNINTERLEAVE2_ST(VLD(input, 2*k), VLD(input, 2*k+1), work2, 2*k, 2*k+1);
  }
  return cfftf1_tail_ps(n, work2, work1, 2, 1, 0, wa, ifac, isign);
}
#undef PF_INTERLEAVE2_ST
#undef PF_UNINTERLEAVE2_ST
#endif /* PF_INPLACE_CODELETS */
#endif /* PF_FUSED_LAYOUT */
#if PFFFT_TERMINAL_RADIX8 && !defined(PF_INPLACE_CODELETS)
#undef PF_PASS8
#endif
/* the codelet header macros, at the end of the outer inclusion; nothing
   after the PF_INPLACE_CODELETS inclusion uses them */
#ifndef PF_INPLACE_CODELETS
#undef PF_CODELET
#undef PF_CODELET_NAME
#undef PF_CODELET_RESTRICT
#endif
