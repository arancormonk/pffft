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

/* Intentionally included twice, with PFFFT_REAL_ORDERED set to 0 and 1.
   Both layouts use the same arithmetic, including the DC/Nyquist boundary
   terms. Ordered transforms load/store the final spectrum directly. */

static NEVER_INLINE(void) PFFFT_REAL_FINALIZE(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  int k, dk = Ncvec/SIMD_SZ; /* number of 4x4 matrix blocks */
  /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

#if PFFFT_REAL_ORDERED
  v4sf first[8], previous1, previous3;
#define PFFFT_REAL_FIRST(k) uout[2*dk*((k)/2)].f[(k)%2]
#else
#define PFFFT_REAL_FIRST(k) uout[k].f[0]
#endif
  v4sf_union cr, ci, *uout = (v4sf_union*)out;
  v4sf save = in[7], zero=VZERO();
  float xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3;
  static const float s = (float)(M_SQRT1_2);

  cr.v = in[0]; ci.v = in[Ncvec*2-1];
  assert(in != out);
#if PFFFT_REAL_ORDERED
  FUNC_REAL_FINALIZE_4X4(&zero, &zero, in+1, e, first);
  pffft_real_store_ordered(first, out, dk, 0, &previous1, &previous3);
#else
  FUNC_REAL_FINALIZE_4X4(&zero, &zero, in+1, e, out);
#endif

  /*
    [cr0 cr1 cr2 cr3 ci0 ci1 ci2 ci3]

    [Xr(1)]  ] [1   1   1   1   0   0   0   0]
    [Xr(N/4) ] [0   0   0   0   1   s   0  -s]
    [Xr(N/2) ] [1   0  -1   0   0   0   0   0]
    [Xr(3N/4)] [0   0   0   0   1  -s   0   s]
    [Xi(1)   ] [1  -1   1  -1   0   0   0   0]
    [Xi(N/4) ] [0   0   0   0   0  -s  -1  -s]
    [Xi(N/2) ] [0  -1   0   1   0   0   0   0]
    [Xi(3N/4)] [0   0   0   0   0  -s   1  -s]
  */

  xr0=(cr.f[0]+cr.f[2]) + (cr.f[1]+cr.f[3]); PFFFT_REAL_FIRST(0) = xr0;
  xi0=(cr.f[0]+cr.f[2]) - (cr.f[1]+cr.f[3]); PFFFT_REAL_FIRST(1) = xi0;
  xr2=(cr.f[0]-cr.f[2]);                     PFFFT_REAL_FIRST(4) = xr2;
  xi2=(cr.f[3]-cr.f[1]);                     PFFFT_REAL_FIRST(5) = xi2;
  xr1= ci.f[0] + s*(ci.f[1]-ci.f[3]);        PFFFT_REAL_FIRST(2) = xr1;
  xi1=-ci.f[2] - s*(ci.f[1]+ci.f[3]);        PFFFT_REAL_FIRST(3) = xi1;
  xr3= ci.f[0] - s*(ci.f[1]-ci.f[3]);        PFFFT_REAL_FIRST(6) = xr3;
  xi3= ci.f[2] - s*(ci.f[1]+ci.f[3]);        PFFFT_REAL_FIRST(7) = xi3;
#undef PFFFT_REAL_FIRST
  for (k=1; k < dk; ++k) {
    v4sf save_next = in[8*k+7];
#if PFFFT_REAL_ORDERED
    v4sf block[8];
    FUNC_REAL_FINALIZE_4X4(&save, &in[8*k+0], in + 8*k+1, e + k*6, block);
    pffft_real_store_ordered(block, out, dk, k, &previous1, &previous3);
#else
    FUNC_REAL_FINALIZE_4X4(&save, &in[8*k+0], in + 8*k+1,
                           e + k*6, out + k*8);
#endif
    save = save_next;
  }
#if PFFFT_REAL_ORDERED
  /* The low pairs were written by the boundary formulas above. Complete
     their high pairs without reading back those individual scalar stores.
     Use byte offsets to copy part of each vector's object representation. */
  const size_t pair_bytes = 2*sizeof(float);
  memcpy((unsigned char*)&out[2*dk] + pair_bytes,
         (const unsigned char*)&previous1 + pair_bytes, pair_bytes);
  memcpy((unsigned char*)&out[6*dk] + pair_bytes,
         (const unsigned char*)&previous3 + pair_bytes, pair_bytes);
#endif
}


static NEVER_INLINE(void) PFFFT_REAL_PREPROCESS(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  int k, dk = Ncvec/SIMD_SZ; /* number of 4x4 matrix blocks */
  /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

  v4sf_union Xr, Xi, *uout = (v4sf_union*)out;
  float cr0, ci0, cr1, ci1, cr2, ci2, cr3, ci3;
  static const float s = (float)M_SQRT2;
#if PFFFT_REAL_ORDERED
  v4sf block[8];
  const v4sf *first = block;
  pffft_real_load_ordered(in, block, dk, 0);
#else
  const v4sf *first = in;
#endif
  assert(in != out);
  for (k=0; k < 4; ++k) {
    Xr.f[k] = ((const float*)first)[8*k];
    Xi.f[k] = ((const float*)first)[8*k+4];
  }

  FUNC_REAL_PREPROCESS_4X4(first, e, out+1, 1); /* will write only 6 values */

  /*
    [Xr0 Xr1 Xr2 Xr3 Xi0 Xi1 Xi2 Xi3]

    [cr0] [1   0   2   0   1   0   0   0]
    [cr1] [1   0   0   0  -1   0  -2   0]
    [cr2] [1   0  -2   0   1   0   0   0]
    [cr3] [1   0   0   0  -1   0   2   0]
    [ci0] [0   2   0   2   0   0   0   0]
    [ci1] [0   s   0  -s   0  -s   0  -s]
    [ci2] [0   0   0   0   0  -2   0   2]
    [ci3] [0  -s   0   s   0  -s   0  -s]
  */
  for (k=1; k < dk; ++k) {
#if PFFFT_REAL_ORDERED
    pffft_real_load_ordered(in, block, dk, k);
    FUNC_REAL_PREPROCESS_4X4(block, e + k*6, out-1+k*8, 0);
#else
    FUNC_REAL_PREPROCESS_4X4(in+8*k, e + k*6, out-1+k*8, 0);
#endif
  }

  cr0=(Xr.f[0]+Xi.f[0]) + 2*Xr.f[2]; uout[0].f[0] = cr0;
  cr1=(Xr.f[0]-Xi.f[0]) - 2*Xi.f[2]; uout[0].f[1] = cr1;
  cr2=(Xr.f[0]+Xi.f[0]) - 2*Xr.f[2]; uout[0].f[2] = cr2;
  cr3=(Xr.f[0]-Xi.f[0]) + 2*Xi.f[2]; uout[0].f[3] = cr3;
  ci0= 2*(Xr.f[1]+Xr.f[3]);                       uout[2*Ncvec-1].f[0] = ci0;
  ci1= s*(Xr.f[1]-Xr.f[3]) - s*(Xi.f[1]+Xi.f[3]); uout[2*Ncvec-1].f[1] = ci1;
  ci2= 2*(Xi.f[3]-Xi.f[1]);                       uout[2*Ncvec-1].f[2] = ci2;
  ci3=-s*(Xr.f[1]-Xr.f[3]) - s*(Xi.f[1]+Xi.f[3]); uout[2*Ncvec-1].f[3] = ci3;
}
